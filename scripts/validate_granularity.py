"""Validation experiment: does finer spatial granularity (tiles) improve retrieval?
Strategies: whole-only (baseline) / tiles-max / whole+tiles-max / whole+tiles-avg
Metrics on full RSICD test + farmland subset. Ranks computed over full 1093-image gallery."""
import os, csv, re, json, time, collections
import numpy as np, torch
from PIL import Image
import open_clip

CKPT = r"work/baselines/RemoteCLIP-weights/RemoteCLIP-ViT-B-32.pt"
IMG_DIR = r"work/datasets/remoteclip-ret/test_images"
CLSDIR = r"work/datasets/RSICD_optimal/txtclasses/txtclasses_rsicd"
OUT = r"outputs/validation-granularity"
os.makedirs(OUT, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, flush=True)

# ---- model ----
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval().to(device)

# ---- load cache (whole-image + text embeddings, image list) ----
z = np.load(r"work/emb_cache_gpu/RSICD_b32.npz", allow_pickle=True)
whole_emb = torch.tensor(z["img"])          # (1093,512)
tx_emb = torch.tensor(z["txt"])             # (5465,512)
imgs = list(z["imgs"])                       # filenames with rsicd_ prefix
n_img, n_cap = len(imgs), len(z["caps"])
per = n_cap // n_img                          # 5
print("images:", n_img, "captions:", n_cap, "per:", per, flush=True)

# ---- encode 2x2 tiles ----
t0 = time.time()
tile_embs = []   # list of (n_img,512) per tile position, 4 tiles
for pos in range(4):
    out = []
    for i in range(0, n_img, 64):
        files = imgs[i:i+64]
        batch = []
        for f in files:
            im = Image.open(os.path.join(IMG_DIR, f)).convert("RGB")
            w, h = im.size
            x0 = 0 if pos in (0,2) else w//2
            y0 = 0 if pos in (1,3) else h//2   # pos: 0=TL,1=BL,2=TR,3=BR
            batch.append(preprocess(im.crop((x0, y0, x0+w//2, y0+h//2))))
        batch = torch.stack(batch).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
            f = model.encode_image(batch)
            f = f / f.norm(dim=-1, keepdim=True)
        out.append(f.float().cpu())
    tile_embs.append(torch.cat(out))
    print(f"  tile {pos} done {time.time()-t0:.1f}s", flush=True)
print("tile encoding total %.1fs" % (time.time()-t0), flush=True)

# ---- similarity strategies: S[caption, image] ----
S_whole = (tx_emb @ whole_emb.T).numpy()                          # (5465,1093)
S_tiles = [ (tx_emb @ te.T).numpy() for te in tile_embs ]          # 4 x (5465,1093)
S_tilesmax = np.max(np.stack(S_tiles), axis=0)
views = np.stack([S_whole] + S_tiles)                              # (5,5465,1093)
S_wtmax = np.max(views, axis=0)
avg_emb = (whole_emb + sum(tile_embs)) / 5
avg_emb = avg_emb / avg_emb.norm(dim=-1, keepdim=True)
S_avg = (tx_emb @ avg_emb.T).numpy()

strategies = {"whole_only": S_whole, "tiles_max": S_tilesmax,
              "whole+tiles_max": S_wtmax, "whole+tiles_avg": S_avg}

# ---- metrics ----
def ranks_from_S(S):
    gt_img = np.arange(n_cap) // per
    order = np.argsort(-S, axis=1)
    rt = np.where(order == gt_img[:, None])[1]
    orderi = np.argsort(-S.T, axis=1)
    ri = np.empty(n_img, dtype=int)
    for gi in range(n_img):
        gts = np.arange(gi*per, (gi+1)*per)
        ri[gi] = np.where(np.isin(orderi[gi], gts))[0].min()
    return rt, ri

def block(idxi, rt, ri):
    r_t = rt[np.concatenate([np.arange(i*per,(i+1)*per) for i in idxi])]
    r_i = ri[idxi]
    d = {"n_images": len(idxi),
         "TR@1":(r_t<1).mean()*100,"TR@5":(r_t<5).mean()*100,"TR@10":(r_t<10).mean()*100,
         "IR@1":(r_i<1).mean()*100,"IR@5":(r_i<5).mean()*100,"IR@10":(r_i<10).mean()*100}
    d["mR"] = float(np.mean([d["TR@1"],d["TR@5"],d["TR@10"],d["IR@1"],d["IR@5"],d["IR@10"]]))
    return d

# ---- farmland indices (same rule as per-class script) ----
name2cls = {}
for tf in os.listdir(CLSDIR):
    if not tf.endswith(".txt"): continue
    cls = tf[:-4].lower()
    if cls == "playfields": cls = "playground"
    for line in open(os.path.join(CLSDIR, tf), encoding="utf-8", errors="ignore"):
        nm = line.strip()
        if nm: name2cls[nm] = cls
cats = [name2cls[re.sub(r"^rsicd_","",f)] for f in imgs]
farm_idx = np.array(sorted(i for i,c in enumerate(cats) if c=="farmland"))
all_idx = np.arange(n_img)

cols = ["strategy","subset","n_images","TR@1","TR@5","TR@10","IR@1","IR@5","IR@10","mR"]
results = []
for sname, S in strategies.items():
    rt, ri = ranks_from_S(S)
    for subname, idx in [("ALL", all_idx), ("farmland", farm_idx)]:
        d = block(idx, rt, ri)
        results.append({"strategy":sname,"subset":subname, **d})

with open(os.path.join(OUT,"granularity_validation.csv"),"w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in results:
        w.writerow({k:(round(r[k],2) if isinstance(r[k],float) else r[k]) for k in cols})
json.dump(results, open(os.path.join(OUT,"granularity_validation.json"),"w"), indent=2)

print("\n===== RESULTS =====")
print(f"{'strategy':18s}{'subset':10s}{'n':>4s}{'TR@1':>7s}{'TR@5':>7s}{'TR@10':>7s}{'IR@1':>7s}{'IR@5':>7s}{'IR@10':>7s}{'mR':>7s}")
for r in results:
    print(f"{r['strategy']:18s}{r['subset']:10s}{r['n_images']:4d}{r['TR@1']:7.2f}{r['TR@5']:7.2f}{r['TR@10']:7.2f}{r['IR@1']:7.2f}{r['IR@5']:7.2f}{r['IR@10']:7.2f}{r['mR']:7.2f}")

