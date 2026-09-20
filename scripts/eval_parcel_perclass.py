"""Per-category comparison: trained ParcelAlign vs RemoteCLIP baseline on RSICD test.
Also: bootstrap 95% CI for the farmland subset (both models)."""
import sys, os, json, csv, re
sys.path.insert(0, r"work")
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
import open_clip
from models.parcel_align import ParcelAlign

RUN = r"outputs/parcel-align/run_20260919_220949"
CKPT = r"work/baselines/RemoteCLIP-weights/RemoteCLIP-ViT-B-32.pt"
EVAL_IMGDIR = r"work/datasets/remoteclip-ret/test_images"
CLSDIR = r"work/datasets/RSICD_optimal/txtclasses/txtclasses_rsicd"
NPZ = r"work/emb_cache_gpu/RSICD_b32.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- load trained model ----
model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
pa = ParcelAlign(model, k=8).to(DEVICE)
ck = torch.load(os.path.join(RUN, "best_trainable.pt"), map_location="cpu")
pa.tokenizer.load_state_dict(ck["tokenizer"])
pa.logit_g.data = ck["logit_g"].to(DEVICE); pa.logit_p.data = ck["logit_p"].to(DEVICE)
pa.alpha_logit.data = ck["alpha_logit"].to(DEVICE)
pa.eval()
tok = open_clip.get_tokenizer("ViT-B-32")

z = np.load(NPZ, allow_pickle=True)
files = list(z["imgs"]); caps = list(z["caps"])
n_img, n_cap = len(files), len(caps); per = n_cap // n_img

# ---- encode with ParcelAlign ----
clss, parcs = [], []
for i in range(0, n_img, 64):
    imgs = torch.stack([pre(Image.open(os.path.join(EVAL_IMGDIR, f)).convert("RGB")) for f in files[i:i+64]]).to(DEVICE)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE=="cuda")):
        c, p = pa.encode_image_parcels(imgs)
    clss.append(c.float().cpu()); parcs.append(p.float().cpu())
cls_n = torch.cat(clss); parc_n = torch.cat(parcs)
texts = []
for i in range(0, n_cap, 512):
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE=="cuda")):
        t = pa.clip.encode_text(tok(list(caps[i:i+512])).to(DEVICE))
        t = F.normalize(t, dim=-1)
    texts.append(t.float().cpu())
text_n = torch.cat(texts)

# ---- score matrices: ParcelAlign (mixed) and baseline (pure cls cosine) ----
a = torch.sigmoid(pa.alpha_logit).item(); tg = pa.logit_g.exp().item(); tp = pa.logit_p.exp().item()
c_gpu = cls_n.to(DEVICE); p_gpu = parc_n.to(DEVICE)
rows = []
for i in range(0, n_cap, 1024):
    t = text_n[i:i+1024].to(DEVICE)
    Sg = t @ c_gpu.T
    Sp = torch.einsum("id,jkd->ijk", t, p_gpu).max(dim=2).values
    rows.append((a * tg * Sg + (1 - a) * tp * Sp).cpu())
S_pa = torch.cat(rows).numpy()
S_bl = (text_n @ cls_n.T).numpy()          # baseline: pure global cosine

def ranks_from_S(S):
    gt = np.arange(S.shape[0]) // per
    order = np.argsort(-S, axis=1)
    rt = np.where(order == gt[:, None])[1]
    orderi = np.argsort(-S.T, axis=1)
    ri = np.empty(S.shape[1], dtype=int)
    for gi in range(S.shape[1]):
        gts = np.arange(gi*per, (gi+1)*per)
        ri[gi] = np.where(np.isin(orderi[gi], gts))[0].min()
    return rt, ri

rt_pa, ri_pa = ranks_from_S(S_pa)
rt_bl, ri_bl = ranks_from_S(S_bl)

def blk(r_t, r_i):
    d = {"TR@1":(r_t<1).mean()*100,"TR@5":(r_t<5).mean()*100,"TR@10":(r_t<10).mean()*100,
         "IR@1":(r_i<1).mean()*100,"IR@5":(r_i<5).mean()*100,"IR@10":(r_i<10).mean()*100}
    d["mR"] = float(np.mean(list(d.values())))
    return d

# ---- category map ----
name2cls = {}
for tf in os.listdir(CLSDIR):
    if tf.endswith(".txt"):
        cls = tf[:-4].lower()
        if cls == "playfields": cls = "playground"
        for line in open(os.path.join(CLSDIR, tf), encoding="utf-8", errors="ignore"):
            nm = line.strip()
            if nm: name2cls[nm] = cls
cats = [name2cls[re.sub(r"^rsicd_","",f)] for f in files]
import collections
by = collections.defaultdict(list)
for i, c in enumerate(cats): by[c].append(i)

def cat_metrics(rt, ri):
    out = {}
    for c, idxi in by.items():
        idxi = np.array(sorted(idxi))
        idxc = np.sort(np.concatenate([np.arange(i*per,(i+1)*per) for i in idxi]))
        out[c] = blk(rt[idxc], ri[idxi])
    out["ALL"] = blk(rt, ri)
    return out

M_pa = cat_metrics(rt_pa, ri_pa)
M_bl = cat_metrics(rt_bl, ri_bl)

# ---- farmland bootstrap CI ----
farm_idx = np.array(sorted(by["farmland"]))
farm_cap = np.sort(np.concatenate([np.arange(i*per,(i+1)*per) for i in farm_idx]))
rng = np.random.RandomState(42)
def boot_ci(rt, ri, n=2000):
    vals = []
    for _ in range(n):
        samp = rng.choice(farm_idx, size=len(farm_idx), replace=True)
        sampc = np.concatenate([np.arange(i*per,(i+1)*per) for i in samp])
        vals.append(blk(rt[sampc], ri[samp]).values() if False else blk(rt[sampc], ri[samp])["mR"])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
ci_pa = boot_ci(rt_pa, ri_pa); ci_bl = boot_ci(rt_bl, ri_bl)

# ---- save + print ----
order = sorted([c for c in M_pa if c != "ALL"], key=lambda c: M_pa[c]["mR"] - M_bl[c]["mR"])
cols = ["category","n","BL_mR","PA_mR","delta","BL_TR@1","PA_TR@1","d_TR@1","BL_IR@1","PA_IR@1","d_IR@1"]
with open(os.path.join(RUN,"perclass_compare.csv"),"w",newline="",encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(cols)
    for c in order + ["ALL"]:
        b, p_ = M_bl[c], M_pa[c]; n = len(by[c]) if c != "ALL" else n_img
        w.writerow([c, n, round(b["mR"],2), round(p_["mR"],2), round(p_["mR"]-b["mR"],2),
                    round(b["TR@1"],2), round(p_["TR@1"],2), round(p_["TR@1"]-b["TR@1"],2),
                    round(b["IR@1"],2), round(p_["IR@1"],2), round(p_["IR@1"]-b["IR@1"],2)])

print(f"{'category':20s}{'n':>4s}{'BL_mR':>7s}{'PA_mR':>7s}{'delta':>7s}")
for c in order + ["ALL"]:
    b, p_ = M_bl[c], M_pa[c]; n = len(by[c]) if c != "ALL" else n_img
    mark = " <<< FARMLAND" if c == "farmland" else ""
    print(f"{c:20s}{n:4d}{b['mR']:7.2f}{p_['mR']:7.2f}{p_['mR']-b['mR']:+7.2f}{mark}")

print(f"\nFARMLAND bootstrap 95% CI (2000 resamples):")
print(f"  baseline:   mR 23.51? actual {M_bl['farmland']['mR']:.2f}  CI [{ci_bl[0]:.2f}, {ci_bl[1]:.2f}]")
print(f"  ParcelAlign: mR {M_pa['farmland']['mR']:.2f}  CI [{ci_pa[0]:.2f}, {ci_pa[1]:.2f}]")
overlap = not (ci_pa[1] < ci_bl[0] or ci_bl[1] < ci_pa[0])
print(f"  CI overlap: {overlap} -> {'NOT significant' if overlap else 'significant'}")

json.dump({"farmland_baseline_ci": ci_bl, "farmland_parcelalign_ci": ci_pa,
           "farmland_bl_mR": M_bl["farmland"]["mR"], "farmland_pa_mR": M_pa["farmland"]["mR"],
           "ALL_bl_mR": M_bl["ALL"]["mR"], "ALL_pa_mR": M_pa["ALL"]["mR"]},
          open(os.path.join(RUN,"farmland_ci.json"),"w"), indent=2)
print(f"\nsaved: perclass_compare.csv, farmland_ci.json -> {RUN}")
