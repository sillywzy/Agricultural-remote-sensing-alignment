"""RemoteCLIP baseline retrieval on RSITMD / RSICD test sets (GPU if available)."""
import os, csv, json, time, numpy as np, torch
from PIL import Image
import open_clip

CKPT = r"work/baselines/RemoteCLIP-weights/RemoteCLIP-ViT-B-32.pt"
IMG_DIR = r"work/datasets/remoteclip-ret/test_images"
CSV = {"RSITMD": r"work/datasets/remoteclip-ret/rsitmd_test.csv",
       "RSICD": r"work/datasets/remoteclip-ret/rsicd_test.csv"}
CACHE = "work/emb_cache_gpu"
OUT = r"outputs/baseline-results"
os.makedirs(CACHE, exist_ok=True); os.makedirs(OUT, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, flush=True)

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval().to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

def load_pairs(name):
    rows = list(csv.DictReader(open(CSV[name], encoding="utf-8-sig"), delimiter="\t"))
    by = {}
    for r in rows:
        fn = r["filename"].strip()
        if os.path.exists(os.path.join(IMG_DIR, fn)):
            by.setdefault(fn, []).append(r["title"].strip())
    imgs = sorted(by)
    caps = []
    for fn in imgs:
        caps.extend(by[fn][:5])
    return imgs, caps

def encode_images(imgs, bs=128):
    out = []
    t0 = time.time()
    for i in range(0, len(imgs), bs):
        files = imgs[i:i+bs]
        batch = torch.stack([preprocess(Image.open(os.path.join(IMG_DIR, f)).convert("RGB")) for f in files]).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
            f = model.encode_image(batch)
            f = f / f.norm(dim=-1, keepdim=True)
        out.append(f.float().cpu())
        print(f"  images {i+len(files)}/{len(imgs)}", flush=True)
    print(f"  image encoding total {time.time()-t0:.1f}s", flush=True)
    return torch.cat(out)

def encode_texts(caps, bs=1024):
    out = []
    for i in range(0, len(caps), bs):
        with torch.no_grad():
            t = model.encode_text(tokenizer(caps[i:i+bs]).to(device))
            t = t / t.norm(dim=-1, keepdim=True)
        out.append(t.float().cpu())
    return torch.cat(out)

def metrics(imgs, caps, im_emb, tx_emb):
    n_img, n_cap = len(imgs), len(caps)
    per = n_cap // n_img
    S = (tx_emb @ im_emb.T).numpy()
    gt_img = np.arange(n_cap) // per
    order = np.argsort(-S, axis=1)
    rank_ti = np.where(order == gt_img[:, None])[1]
    tr = {f"TR@{k}": float((rank_ti < k).mean()*100) for k in (1, 5, 10)}
    tr["TR_meanRank"] = float(rank_ti.mean()+1); tr["TR_medianRank"] = float(np.median(rank_ti)+1)
    Si = S.T
    orderi = np.argsort(-Si, axis=1)
    best_rank = np.full(n_img, 10**9)
    for gi in range(n_img):
        gts = list(range(gi*per, (gi+1)*per))
        ranks = np.where(np.isin(orderi[gi], gts))[0]
        best_rank[gi] = ranks.min()
    ir = {f"IR@{k}": float((best_rank < k).mean()*100) for k in (1, 5, 10)}
    ir["IR_meanRank"] = float(best_rank.mean()+1); ir["IR_medianRank"] = float(np.median(best_rank)+1)
    mr = float(np.mean([tr["TR@1"], tr["TR@5"], tr["TR@10"], ir["IR@1"], ir["IR@5"], ir["IR@10"]]))
    return {**tr, **ir, "mR": mr, "n_images": n_img, "n_captions": n_cap}

allres = {}
for name in ["RSITMD", "RSICD"]:
    print(f"=== {name} ===", flush=True)
    imgs, caps = load_pairs(name)
    print(f"  images: {len(imgs)}, captions: {len(caps)}", flush=True)
    cache = os.path.join(CACHE, f"{name}_b32.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        im_emb = torch.tensor(z["img"]); tx_emb = torch.tensor(z["txt"])
    else:
        im_emb = encode_images(imgs)
        tx_emb = encode_texts(caps)
        np.savez(cache, img=im_emb.numpy(), txt=tx_emb.numpy(),
                 imgs=np.array(imgs), caps=np.array(caps))
    res = metrics(imgs, caps, im_emb, tx_emb)
    allres[name] = res
    print(json.dumps(res, indent=1), flush=True)

json.dump(allres, open(os.path.join(OUT, "retrieval_baseline.json"), "w"), indent=2)
with open(os.path.join(OUT, "retrieval_baseline.csv"), "w", newline="", encoding="utf-8-sig") as f:
    cols = ["dataset","TR@1","TR@5","TR@10","IR@1","IR@5","IR@10","mR","TR_meanRank","TR_medianRank","IR_meanRank","IR_medianRank","n_images","n_captions"]
    w = csv.writer(f); w.writerow(cols)
    for k, v in allres.items(): w.writerow([k]+[round(v[c],3) if isinstance(v[c],float) else v[c] for c in cols[1:]])
print("SAVED to", OUT, flush=True)
