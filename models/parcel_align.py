"""ParcelAlign: learnable parcel-level alignment on top of a frozen RemoteCLIP backbone.

Core idea
---------
RemoteCLIP compresses a whole image into ONE vector (global alignment) and therefore
cannot tell WHICH parcel a text refers to (see Figure 1 motivation experiments).

ParcelAlign adds lightweight trainable components on a frozen backbone:

1. ParcelTokenizer: K learnable queries cross-attend to the 49 ViT patch tokens,
   producing K parcel tokens. Unlike fixed 2x2 tiling (which failed), parcel shapes
   are learned soft attention patterns over patches.
2. Mixed retrieval score: global term (text vs CLS) + parcel term (max cosine over
   the K parcel embeddings), FILIP-style, with a learnable mix weight.
3. Losses:
   - global InfoNCE (keeps scene-level retrieval ability)
   - parcel-level InfoNCE (text must pick its own image's best parcel out of all
     B*K parcels in the batch -> parcels become text-matching units)
   - diversity regularizer (mean off-diagonal cosine between parcels, keeps the K
     parcels from collapsing into one)

Only ~2.4M parameters train; the backbone (151M) stays frozen.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ParcelTokenizer(nn.Module):
    """K learnable queries cross-attend to 49 ViT patch tokens -> K parcel tokens."""

    def __init__(self, dim=768, k=8, heads=8):
        super().__init__()
        self.k = k
        self.queries = nn.Parameter(torch.randn(k, dim) * 0.02)
        self.norm_q = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, patch_tokens):                    # (B, 49, 768)
        B = patch_tokens.shape[0]
        q = self.norm_q(self.queries).unsqueeze(0).expand(B, -1, -1)
        parcels, attn = self.cross_attn(q, patch_tokens, patch_tokens)
        return parcels, attn                            # (B,K,768), (B,K,49)


class ParcelAlign(nn.Module):
    """Frozen RemoteCLIP + trainable parcel tokenizer with parcel-level contrastive loss."""

    def __init__(self, clip_model, k=8, heads=8, init_t=0.07, div_weight=0.05):
        super().__init__()
        self.clip = clip_model
        for p in self.clip.parameters():
            p.requires_grad_(False)
        vis = clip_model.visual
        self.dim = vis.proj.shape[0]                    # 768
        self.joint = vis.proj.shape[1]                  # 512
        self.tokenizer = ParcelTokenizer(self.dim, k, heads)
        self.k = k
        self.div_weight = div_weight
        self.logit_g = nn.Parameter(torch.log(torch.tensor(1.0 / init_t)))
        self.logit_p = nn.Parameter(torch.log(torch.tensor(1.0 / init_t)))
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))   # sigmoid -> 0.5 mix
        self._caps = {}
        self._hook = vis.transformer.register_forward_hook(self._grab)

    # ---- internals ------------------------------------------------------
    def _grab(self, module, inputs, output):
        self._caps["tokens"] = output                   # (B, 50, 768)

    @torch.no_grad()
    def encode_backbone(self, images, texts):
        """Frozen backbone features. The hook grabs the token sequence."""
        cls_emb = self.clip.visual(images)              # (B, 512), triggers hook
        tokens = self._caps["tokens"]
        patch_tokens = tokens[:, 1:, :]                 # (B, 49, 768)
        patch_emb = self.clip.visual.ln_post(patch_tokens) @ self.clip.visual.proj
        text_emb = self.clip.encode_text(texts)         # (B, 512)
        return cls_emb, patch_tokens, patch_emb, text_emb

    def _project(self, parcels):
        """Frozen ln_post+proj map (differentiable w.r.t. parcels)."""
        return self.clip.visual.ln_post(parcels) @ self.clip.visual.proj

    # ---- forward / loss --------------------------------------------------
    def forward(self, images, texts):
        cls_emb, patch_tokens, _, text_emb = self.encode_backbone(images, texts)
        # Trainable components run in fp32: fp16 backward produced NaN gradients
        # (GradScaler then skipped every optimizer step -> no learning at all).
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=False):
            patch_tokens = patch_tokens.float()
            parcels, tok_attn = self.tokenizer(patch_tokens)         # (B,K,768),(B,K,49)
            parcel_emb = self._project(parcels)                      # (B,K,512)

            t = F.normalize(text_emb.float(), dim=-1)
            g = F.normalize(cls_emb.float(), dim=-1)
            p = F.normalize(parcel_emb, dim=-1)
            B = t.shape[0]

            Sg = t @ g.T                                    # (B,B) global cosine
            Sp = torch.einsum("id,jkd->ijk", t, p)          # (B,B,K) text x parcel cosine
            Sp_max = Sp.max(dim=2).values                   # (B,B)

            a = torch.sigmoid(self.alpha_logit)
            logits = a * self.logit_g.exp() * Sg + (1 - a) * self.logit_p.exp() * Sp_max
            labels = torch.arange(B, device=t.device)
            loss_global = 0.5 * (F.cross_entropy(logits, labels)
                                 + F.cross_entropy(logits.T, labels))

            # parcel-level InfoNCE: text i must pick the best parcel of ITS image
            flat_p = p.reshape(B * self.k, -1)              # (B*K, 512)
            sims_pt = t @ flat_p.T * self.logit_p.exp()     # (B, B*K)
            pos = Sp.argmax(dim=2).diagonal() + torch.arange(B, device=t.device) * self.k
            loss_parcel = F.cross_entropy(sims_pt, pos)

            # diversity: mean off-diagonal cosine between the K parcels
            sim_pp = torch.einsum("bkd,bjd->bkj", p, p)     # (B,K,K)
            eye = torch.eye(self.k, dtype=torch.bool, device=p.device)
            loss_div = sim_pp[:, ~eye].mean()

            loss = loss_global + loss_parcel + self.div_weight * loss_div
            return {"loss": loss,
                    "loss_global": loss_global,
                    "loss_parcel": loss_parcel,
                    "loss_div": loss_div,
                    "alpha": a.item(),
                    "logits": logits.detach(),
                    "tok_attn": tok_attn.detach()}

    # ---- inference helpers -------------------------------------------------
    @torch.no_grad()
    def encode_image_parcels(self, images):
        """Returns (cls_emb (B,512), parcel_emb (B,K,512)) for retrieval."""
        dummy = torch.zeros(1, 77, dtype=torch.long, device=images.device)
        cls_emb, patch_tokens, _, _ = self.encode_backbone(images, dummy)
        parcels, _ = self.tokenizer(patch_tokens.float())
        return F.normalize(cls_emb.float(), dim=-1), F.normalize(self._project(parcels), dim=-1)

    def score_matrix(self, text_emb_n, cls_n, parcel_n):
        """Mixed retrieval score, same formula as training. Inputs normalized."""
        Sg = text_emb_n @ cls_n.T
        Sp = torch.einsum("id,jkd->ijk", text_emb_n, parcel_n).max(dim=2).values
        a = torch.sigmoid(self.alpha_logit)
        return a * self.logit_g.exp() * Sg + (1 - a) * self.logit_p.exp() * Sp

    @torch.no_grad()
    def text_patch_attention(self, text_emb, parcel_emb, tok_attn):
        """Text-conditioned patch map: which patches does this text look at? -> (B,49)."""
        t = F.normalize(text_emb.float(), dim=-1)
        pr = F.normalize(parcel_emb.float(), dim=-1)
        s = torch.einsum("bd,bkd->bk", t, pr)                          # (B,K)
        w = F.softmax(s, dim=-1)                                       # (B,K)
        return torch.einsum("bk,bkp->bp", w, tok_attn.float())         # (B,49)
