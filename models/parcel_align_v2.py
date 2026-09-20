"""ParcelAlign v2: input-adaptive granularity gating.

v1 finding: parcel-level matching helps object-centric categories (railwaystation +18.3)
but hurts texture-global categories (desert -10.6, farmland -4.1). A single global alpha
cannot serve both. v2 replaces the scalar alpha with a per-image gate that predicts
how much to trust the global vs parcel term for EACH image:

    alpha(x) = Gate(CLS embedding, parcel-parcel diversity, parcel-to-CLS stats)

Expected learned behavior: homogeneous texture images (desert/farmland/forest) -> high
alpha (trust global); heterogeneous object images (railwaystation/storagetanks) -> low
alpha (trust parcels).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.parcel_align import ParcelTokenizer


class GranularityGate(nn.Module):
    """Per-image alpha in (0,1): weight of the global term vs the parcel term."""

    def __init__(self, joint_dim=512, k=8, hidden=128):
        super().__init__()
        self.k = k
        # input: cls (512) + sim_pc stats (3) + sim_pc (k) + parcel diversity (1)
        self.mlp = nn.Sequential(
            nn.Linear(joint_dim + 3 + k + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, cls_n, parcel_n):                    # (B,512), (B,K,512) normalized
        sim_pp = torch.einsum("bkd,bjd->bkj", parcel_n, parcel_n)   # (B,K,K)
        eye = torch.eye(self.k, dtype=torch.bool, device=parcel_n.device)
        div = sim_pp[:, ~eye].mean(dim=-1, keepdim=True)            # (B,1)
        sim_pc = torch.einsum("bkd,bd->bk", parcel_n, cls_n)        # (B,K)
        stats = torch.stack([sim_pc.mean(-1), sim_pc.std(-1), sim_pc.max(-1).values], dim=-1)
        x = torch.cat([cls_n, stats, sim_pc, div], dim=-1)
        raw = self.mlp(x).squeeze(-1)                                # (B,)
        raw = raw - raw.mean()                # centering: gate encodes only RELATIVE per-image differences,
        return torch.sigmoid(raw)             # so it cannot globally collapse alpha (batch mean pinned at 0.5)


class ParcelAlignV2(nn.Module):
    """Frozen RemoteCLIP + parcel tokenizer + input-adaptive granularity gate."""

    def __init__(self, clip_model, k=8, heads=8, init_t=0.07, div_weight=0.05):
        super().__init__()
        self.clip = clip_model
        for p in self.clip.parameters():
            p.requires_grad_(False)
        vis = clip_model.visual
        self.dim = vis.proj.shape[0]
        self.joint = vis.proj.shape[1]
        self.tokenizer = ParcelTokenizer(self.dim, k, heads)
        self.gate = GranularityGate(self.joint, k)
        self.k = k
        self.div_weight = div_weight
        self.logit_g = nn.Parameter(torch.log(torch.tensor(1.0 / init_t)))
        self.logit_p = nn.Parameter(torch.log(torch.tensor(1.0 / init_t)))
        self._caps = {}
        self._hook = vis.transformer.register_forward_hook(self._grab)

    def _grab(self, module, inputs, output):
        self._caps["tokens"] = output

    @torch.no_grad()
    def encode_backbone(self, images, texts):
        cls_emb = self.clip.visual(images)
        tokens = self._caps["tokens"]
        patch_tokens = tokens[:, 1:, :]
        text_emb = self.clip.encode_text(texts)
        return cls_emb, patch_tokens, text_emb

    def _project(self, parcels):
        return self.clip.visual.ln_post(parcels) @ self.clip.visual.proj

    def load_v1(self, v1_ckpt):
        """Warm start tokenizer + temperatures from a v1 checkpoint."""
        ck = torch.load(v1_ckpt, map_location="cpu")
        self.tokenizer.load_state_dict(ck["tokenizer"])
        self.logit_g.data = ck["logit_g"].to(self.logit_g.device)
        self.logit_p.data = ck["logit_p"].to(self.logit_p.device)
        return self

    def forward(self, images, texts):
        cls_emb, patch_tokens, text_emb = self.encode_backbone(images, texts)
        # trainable parts in fp32 (fp16 backward produced NaN gradients in v1)
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=False):
            patch_tokens = patch_tokens.float()
            parcels, tok_attn = self.tokenizer(patch_tokens)
            parcel_emb = self._project(parcels)

            t = F.normalize(text_emb.float(), dim=-1)
            g = F.normalize(cls_emb.float(), dim=-1)
            p = F.normalize(parcel_emb, dim=-1)
            B = t.shape[0]

            a = self.gate(g, p)                              # (B,) per-image alpha

            Sg = t @ g.T                                     # (B,B)
            Sp = torch.einsum("id,jkd->ijk", t, p)
            Sp_max = Sp.max(dim=2).values                    # (B,B)

            # column j (image j) weighted by its own gate value a_j
            aw = a.unsqueeze(0)                              # (1,B)
            logits = aw * self.logit_g.exp() * Sg + (1 - aw) * self.logit_p.exp() * Sp_max
            labels = torch.arange(B, device=t.device)
            loss_global = 0.5 * (F.cross_entropy(logits, labels)
                                 + F.cross_entropy(logits.T, labels))

            flat_p = p.reshape(B * self.k, -1)
            sims_pt = t @ flat_p.T * self.logit_p.exp()
            pos = Sp.argmax(dim=2).diagonal() + torch.arange(B, device=t.device) * self.k
            loss_parcel = F.cross_entropy(sims_pt, pos)

            sim_pp = torch.einsum("bkd,bjd->bkj", p, p)
            eye = torch.eye(self.k, dtype=torch.bool, device=p.device)
            loss_div = sim_pp[:, ~eye].mean()

            loss = loss_global + loss_parcel + self.div_weight * loss_div
            return {"loss": loss, "loss_global": loss_global, "loss_parcel": loss_parcel,
                    "loss_div": loss_div,
                    "alpha_mean": a.mean().item(), "alpha_std": a.std().item(),
                    "alpha_min": a.min().item(), "alpha_max": a.max().item(),
                    "logits": logits.detach(), "tok_attn": tok_attn.detach()}

    @torch.no_grad()
    def encode_image_parcels(self, images):
        """Returns (cls_n, parcel_n, alpha) for retrieval and analysis."""
        dummy = torch.zeros(1, 77, dtype=torch.long, device=images.device)
        cls_emb, patch_tokens, _ = self.encode_backbone(images, dummy)
        parcels, _ = self.tokenizer(patch_tokens.float())
        cls_n = F.normalize(cls_emb.float(), dim=-1)
        parc_n = F.normalize(self._project(parcels), dim=-1)
        alpha = self.gate(cls_n, parc_n)
        return cls_n, parc_n, alpha

    def score_matrix(self, text_n, cls_n, parcel_n):
        Sg = text_n @ cls_n.T
        Sp = torch.einsum("id,jkd->ijk", text_n, parcel_n).max(dim=2).values
        a = self.gate(cls_n, parcel_n).unsqueeze(0)          # (1, n_img)
        return a * self.logit_g.exp() * Sg + (1 - a) * self.logit_p.exp() * Sp

    @torch.no_grad()
    def text_patch_attention(self, text_emb, parcel_emb, tok_attn):
        t = F.normalize(text_emb.float(), dim=-1)
        pr = F.normalize(parcel_emb.float(), dim=-1)
        s = torch.einsum("bd,bkd->bk", t, pr)
        w = F.softmax(s, dim=-1)
        return torch.einsum("bk,bkp->bp", w, tok_attn.float())
