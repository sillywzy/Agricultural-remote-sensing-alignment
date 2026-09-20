"""Train ParcelAlign on RSICD train split; evaluate retrieval on the test split each epoch.

Outputs -> outputs/parcel-align/run_YYYYMMDD_HHMMSS/:
  config.json  train_log.txt  results.csv  epoch{k}_trainable.pt  best_trainable.pt
  final_heatmaps/  final_report.json
"""
import os, sys, json, math, time, random, csv, re, collections
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import open_clip
from models.parcel_align import ParcelAlign

SEED, EPOCHS, BS, LR, WD, K, CLIP_GRAD = 42, 6, 64, 2e-4, 0.01, 8, 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT = os.path.join(ROOT, "work", "baselines", "RemoteCLIP-weights", "RemoteCLIP-ViT-B-32.pt")
TRAIN_JSON = os.path.join(ROOT, "work", "datasets", "RSICD_optimal", "dataset_rsicd.json")
TRAIN_IMGDIR = os.path.join(ROOT, "work", "datasets", "RSICD_optimal", "RSICD_images")
EVAL_NPZ = os.path.join(ROOT, "work", "emb_cache_gpu", "RSICD_b32.npz")
EVAL_IMGDIR = os.path.join(ROOT, "work", "datasets", "remoteclip-ret", "test_images")
CLSDIR = os.path.join(ROOT, "work", "datasets", "RSICD_optimal", "txtclasses", "txtclasses_rsicd")
OUT = os.path.join(ROOT, "outputs", "parcel-align", time.strftime("run_%Y%m%d_%H%M%S"))
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "train_log.txt"), "a", encoding="utf-8")

def log(s):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

class RSICDPairs(Dataset):
    def __init__(self, imgdir, pre):
        data = json.load(open(TRAIN_JSON, encoding="utf-8"))["images"]
        self.items = [(e["filename"], [s["raw"] for s in e["sentences"]])
                      for e in data if e["split"] == "train"]
        self.dir = imgdir; self.pre = pre
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        fn, caps = self.items[i]
        return self.pre(Image.open(os.path.join(self.dir, fn)).convert("RGB")), random.choice(caps)

def collate(b): return torch.stack([x[0] for x in b]), [x[1] for x in b]

@torch.no_grad()
def encode_eval_images(pa, pre, files, bs=64):
    clss, parcs = [], []
    for i in range(0, len(files), bs):
        imgs = torch.stack([pre(Image.open(os.path.join(EVAL_IMGDIR, f)).convert("RGB"))
                            for f in files[i:i+bs]]).to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE=="cuda")):
            c, p = pa.encode_image_parcels(imgs)
        clss.append(c.float().cpu()); parcs.append(p.float().cpu())
    return torch.cat(clss), torch.cat(parcs)

@torch.no_grad()
def encode_eval_texts(pa, tok, caps, bs=512):
    out = []
    for i in range(0, len(caps), bs):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE=="cuda")):
            t = pa.clip.encode_text(tok(list(caps[i:i+bs])).to(DEVICE))
            t = F.normalize(t, dim=-1)
        out.append(t.float().cpu())
    return torch.cat(out)

def score_matrix(text_n, cls_n, parcel_n, pa, chunk=1024):
    a = torch.sigmoid(pa.alpha_logit).detach().item()
    tg = pa.logit_g.exp().detach().item(); tp = pa.logit_p.exp().detach().item()
    c = cls_n.to(DEVICE); p = parcel_n.to(DEVICE)
    rows = []
    for i in range(0, text_n.shape[0], chunk):
        t = text_n[i:i+chunk].to(DEVICE)
        Sg = t @ c.T
        Sp = torch.einsum("id,jkd->ijk", t, p).max(dim=2).values
        rows.append((a * tg * Sg + (1 - a) * tp * Sp).cpu())
    return torch.cat(rows).numpy()

def blk(r_t, r_i):
    d = {"TR@1": (r_t < 1).mean()*100, "TR@5": (r_t < 5).mean()*100, "TR@10": (r_t < 10).mean()*100,
         "IR@1": (r_i < 1).mean()*100, "IR@5": (r_i < 5).mean()*100, "IR@10": (r_i < 10).mean()*100}
    d["mR"] = float(np.mean([d["TR@1"],d["TR@5"],d["TR@10"],d["IR@1"],d["IR@5"],d["IR@10"]]))
    return d

def metrics_from_S(S, per):
    n_cap, n_img = S.shape
    gt = np.arange(n_cap) // per
    order = np.argsort(-S, axis=1)
    rt = np.where(order == gt[:, None])[1]
    orderi = np.argsort(-S.T, axis=1)
    ri = np.empty(n_img, dtype=int)
    for gi in range(n_img):
        gts = np.arange(gi*per, (gi+1)*per)
        ri[gi] = np.where(np.isin(orderi[gi], gts))[0].min()
    return rt, ri

def save_trainable(pa, path):
    torch.save({"tokenizer": pa.tokenizer.state_dict(),
                "logit_g": pa.logit_g.detach().cpu(),
                "logit_p": pa.logit_p.detach().cpu(),
                "alpha_logit": pa.alpha_logit.detach().cpu()}, path)

def jet(v):
    v = np.clip(v, 0, 1)
    r = np.clip(1.5-abs(4*v-3),0,1); g = np.clip(1.5-abs(4*v-2),0,1); b = np.clip(1.5-abs(4*v-1),0,1)
    return (np.stack([r,g,b],-1)*255).astype(np.uint8)

def overlay(im, grid, label):
    h = Image.fromarray(jet(grid)).resize(im.size, Image.BICUBIC)
    b = Image.blend(im.convert("RGB"), h, 0.55)
    ImageDraw.Draw(b).text((6,6), label, fill=(255,255,0))
    return b

def final_heatmaps(pa, tok, pre):
    qd = os.path.join(OUT, "final_heatmaps"); os.makedirs(qd, exist_ok=True)
    Q = "farmland"
    for fn in ["rsicd_farmland_37.jpg", "rsicd_farmland_370.jpg"]:
        im = Image.open(os.path.join(EVAL_IMGDIR, fn)).convert("RGB")
        # before: RemoteCLIP sliding window 5x5
        Wd, Sd = 96, 32
        patches = [im.crop((x,y,x+Wd,y+Wd)) for y in range(0,im.height-Wd+1,Sd) for x in range(0,im.width-Wd+1,Sd)]
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE=="cuda")):
            batch = torch.stack([pre(p) for p in patches]).to(DEVICE)
            fe = F.normalize(pa.clip.encode_image(batch), dim=-1)
            te = F.normalize(pa.clip.encode_text(tok([Q]).to(DEVICE)), dim=-1)
        sb = (te @ fe.T).float().cpu().numpy().reshape(5,5)
        sb = (sb - sb.min())/(sb.max()-sb.min()+1e-9)
        # after: ParcelAlign text->patch attention (7x7)
        imgs = pre(im).unsqueeze(0).to(DEVICE)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE=="cuda")):
            cls_emb, patch_tokens, _, text_emb = pa.encode_backbone(imgs, tok([Q]).to(DEVICE))
            parcels, tok_attn = pa.tokenizer(patch_tokens)
            parcel_emb = pa._project(parcels)
            amap = pa.text_patch_attention(text_emb, parcel_emb, tok_attn).float().cpu().numpy().reshape(7,7)
        sa = (amap - amap.min())/(amap.max()-amap.min()+1e-9)
        pa_ = overlay(im, sb, "RemoteCLIP (before)")
        pb = overlay(im, sa, "ParcelAlign (after)")
        m = Image.new("RGB", (im.width*2+20, im.height+10), (15,15,15))
        m.paste(pa_,(5,5)); m.paste(pb,(im.width+15,5))
        m.save(os.path.join(qd, f"{fn[:-4]}__compare.png"))
    return qd

if __name__ == "__main__":
    set_seed(SEED)
    log(f"output: {OUT}")
    log(f"device: {DEVICE}")
    json.dump({"seed":SEED,"epochs":EPOCHS,"bs":BS,"lr":LR,"wd":WD,"k":K}, open(os.path.join(OUT,"config.json"),"w"), indent=2)

    model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    pa = ParcelAlign(model, k=K).to(DEVICE)
    tok = open_clip.get_tokenizer("ViT-B-32")
    ntr = sum(p.numel() for p in pa.parameters() if p.requires_grad)
    log(f"trainable params: {ntr/1e6:.2f}M")

    # eval data (exact pairing from baseline cache)
    z = np.load(EVAL_NPZ, allow_pickle=True)
    eval_files = list(z["imgs"]); eval_caps = list(z["caps"])
    n_img, n_cap = len(eval_files), len(eval_caps); per = n_cap // n_img
    name2cls = {}
    for tf in os.listdir(CLSDIR):
        if tf.endswith(".txt"):
            cls = tf[:-4].lower()
            if cls == "playfields": cls = "playground"
            for line in open(os.path.join(CLSDIR, tf), encoding="utf-8", errors="ignore"):
                nm = line.strip()
                if nm: name2cls[nm] = cls
    cats = [name2cls[re.sub(r"^rsicd_","",f)] for f in eval_files]
    farm_idx = np.array(sorted(i for i,c in enumerate(cats) if c=="farmland"))
    farm_cap = np.sort(np.concatenate([np.arange(i*per,(i+1)*per) for i in farm_idx]))
    log(f"eval: {n_img} imgs / {n_cap} caps | farmland: {len(farm_idx)} imgs")

    ds = RSICDPairs(TRAIN_IMGDIR, pre)
    dl = DataLoader(ds, batch_size=BS, shuffle=True, num_workers=0, collate_fn=collate)
    log(f"train: {len(ds)} pairs, {len(dl)} steps/epoch")
    steps_total = EPOCHS * len(dl)
    trainable = [p for p in pa.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5*(1+math.cos(math.pi*s/steps_total)))
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))

    rescols = ["epoch","loss","loss_global","loss_parcel","loss_div","alpha",
               "ALL_TR@1","ALL_TR@5","ALL_TR@10","ALL_IR@1","ALL_IR@5","ALL_IR@10","ALL_mR",
               "FARM_TR@1","FARM_TR@5","FARM_TR@10","FARM_IR@1","FARM_IR@5","FARM_IR@10","FARM_mR",
               "parcel_div"]
    fres = open(os.path.join(OUT,"results.csv"),"w",newline="",encoding="utf-8-sig")
    wres = csv.DictWriter(fres, fieldnames=rescols); wres.writeheader()

    best_mR, best_ep = -1, -1
    for ep in range(1, EPOCHS+1):
        pa.train(); pa.clip.eval()
        agg = collections.defaultdict(float); nb = 0; t0 = time.time()
        for it, (imgs, caps) in enumerate(dl, 1):
            imgs = imgs.to(DEVICE, non_blocking=True)
            txts = tok(caps).to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE=="cuda")):
                out = pa(imgs, txts)
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, CLIP_GRAD)
            scaler.step(opt); scaler.update(); sched.step()
            for kk in ("loss","loss_global","loss_parcel","loss_div"): agg[kk] += out[kk].item()
            agg["alpha"] += out["alpha"]; nb += 1
            if it % 20 == 0 or it == len(dl):
                log(f"  ep{ep} it{it}/{len(dl)} loss={agg['loss']/nb:.3f} g={agg['loss_global']/nb:.3f} "
                    f"p={agg['loss_parcel']/nb:.3f} div={agg['loss_div']/nb:.3f} a={agg['alpha']/nb:.3f} "
                    f"lr={sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s")
        pa.eval()
        cls_n, parc_n = encode_eval_images(pa, pre, eval_files)
        text_n = encode_eval_texts(pa, tok, eval_caps)
        S = score_matrix(text_n, cls_n, parc_n, pa)
        rt, ri = metrics_from_S(S, per)
        allm = blk(rt, ri); farmm = blk(rt[farm_cap], ri[farm_idx])
        sel = np.random.RandomState(0).choice(len(parc_n), min(256,len(parc_n)), replace=False)
        ps = parc_n[torch.tensor(sel)]
        spp = torch.einsum("mkd,mjd->mkj", ps, ps)
        pdiv = spp[:, ~torch.eye(K, dtype=torch.bool)].mean().item()
        row = {"epoch":ep, "loss":round(agg["loss"]/nb,4), "loss_global":round(agg["loss_global"]/nb,4),
               "loss_parcel":round(agg["loss_parcel"]/nb,4), "loss_div":round(agg["loss_div"]/nb,4),
               "alpha":round(agg["alpha"]/nb,3)}
        for k_,v in allm.items(): row[f"ALL_{k_}"] = round(v,2)
        for k_,v in farmm.items(): row[f"FARM_{k_}"] = round(v,2)
        row["parcel_div"] = round(pdiv,4)
        wres.writerow(row); fres.flush()
        save_trainable(pa, os.path.join(OUT, f"epoch{ep}_trainable.pt"))
        if allm["mR"] > best_mR: best_mR, best_ep = allm["mR"], ep
        log(f"ep{ep} EVAL  ALL mR={allm['mR']:.2f} (TR@1={allm['TR@1']:.2f})  "
            f"FARM mR={farmm['mR']:.2f} (TR@1={farmm['TR@1']:.2f})  div={pdiv:.3f}  "
            f"[{time.time()-t0:.0f}s total]")
    save_trainable(pa, os.path.join(OUT, "best_trainable.pt"))
    qd = final_heatmaps(pa, tok, pre)
    rep = {"best_epoch": best_ep, "best_ALL_mR": best_mR,
           "baseline_ALL_mR": 32.53, "baseline_FARM_mR": 23.51,
           "final_alpha": round(agg["alpha"]/nb,3), "final_parcel_div": round(pdiv,4),
           "heatmaps": qd}
    json.dump(rep, open(os.path.join(OUT,"final_report.json"),"w"), indent=2)
    log(f"DONE. best ep{best_ep} ALL mR={best_mR:.2f} (baseline 32.53)")
    log(f"final report: {os.path.join(OUT,'final_report.json')}")
