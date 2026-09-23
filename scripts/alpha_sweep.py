"""Fixed-alpha sweep (efficient): one full-gallery ranking per alpha, then subset slicing."""
import sys, os, json, re, csv, collections
sys.path.insert(0, r"work")
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
import open_clip
from models.parcel_align_v2 import ParcelAlignV2

RUN = r"outputs/parcel-align/run_v2_20260920_211323"
CKPT = r"work/baselines/RemoteCLIP-weights/RemoteCLIP-ViT-B-32.pt"
NPZ = r"work/emb_cache_gpu/RSICD_b32.npz"
IMGDIR = r"work/datasets/remoteclip-ret/test_images"
CLSDIR = r"work/datasets/RSICD_optimal/txtclasses/txtclasses_rsicd"
OUT = r"outputs/paper-figures"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
pa = ParcelAlignV2(model, k=8).to(DEVICE)
ck = torch.load(os.path.join(RUN, "best_trainable.pt"), map_location="cpu")
pa.tokenizer.load_state_dict(ck["tokenizer"]); pa.gate.load_state_dict(ck["gate"])
pa.logit_g.data = ck["logit_g"].to(DEVICE); pa.logit_p.data = ck["logit_p"].to(DEVICE)
pa.eval()
tok = open_clip.get_tokenizer("ViT-B-32")

z = np.load(NPZ, allow_pickle=True)
files = list(z["imgs"]); caps = list(z["caps"])
n_img, n_cap = len(files), len(caps); per = n_cap // n_img

clss, parcs = [], []
for i in range(0, n_img, 64):
    imgs = torch.stack([pre(Image.open(os.path.join(IMGDIR, f)).convert("RGB")) for f in files[i:i+64]]).to(DEVICE)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        c, p, _ = pa.encode_image_parcels(imgs)
    clss.append(c.float().cpu()); parcs.append(p.float().cpu())
cls_n = torch.cat(clss).cuda(); parc_n = torch.cat(parcs).cuda()
texts = []
for i in range(0, n_cap, 512):
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        t = pa.clip.encode_text(tok(list(caps[i:i+512])).to(DEVICE)); t = F.normalize(t, dim=-1)
    texts.append(t.float().cpu())
text_n = torch.cat(texts).cuda()

with torch.no_grad():
    Sg = (text_n @ cls_n.T).cpu().numpy()
    Sp = torch.einsum("id,jkd->ijk", text_n, parc_n).max(dim=2).values.cpu().numpy()

name2cls = {}
for tf in os.listdir(CLSDIR):
    if tf.endswith(".txt"):
        c = tf[:-4].lower()
        if c == "playfields": c = "playground"
        for line in open(os.path.join(CLSDIR, tf), encoding="utf-8", errors="ignore"):
            if line.strip(): name2cls[line.strip()] = c
cats = [name2cls[re.sub(r"^rsicd_","",f)] for f in files]
by = collections.defaultdict(list)
for i, c in enumerate(cats): by[c].append(i)
farm_idx = np.array(sorted(by["farmland"]))
farm_cap = np.sort(np.concatenate([np.arange(i*per,(i+1)*per) for i in farm_idx]))

alphas = np.round(np.arange(0, 1.0001, 0.05), 2)
ranks_cache = {}
def ranks_at(a):
    S = a * Sg + (1-a) * Sp
    order = np.argsort(-S, axis=1)
    rt = np.where(order == (np.arange(n_cap)//per)[:, None])[1]
    orderi = np.argsort(-S.T, axis=1)
    ri = np.argmax(np.isin(orderi, np.arange(n_img*per).reshape(n_img,1,per)//per*0 + np.arange(n_img)[:,None]) * 0, axis=1)  # placeholder
    # image->text best rank:
    ri = np.empty(n_img, dtype=int)
    for gi in range(n_img):
        ri[gi] = np.where(np.isin(orderi[gi], np.arange(gi*per,(gi+1)*per)))[0].min()
    return rt, ri
def mR(r_t, r_i): return float(np.mean([(r_t<k).mean()*100 for k in (1,5,10)] + [(r_i<k).mean()*100 for k in (1,5,10)]))

sweep = []
cat_series = {c: [] for c in by}
print(f"{'alpha':>6s}{'ALL_mR':>9s}{'FARM_mR':>9s}")
for a in alphas:
    rt, ri = ranks_at(a)
    ranks_cache[a] = (rt, ri)
    allv, farmv = mR(rt, ri), mR(rt[farm_cap], ri[farm_idx])
    sweep.append((float(a), round(allv,2), round(farmv,2)))
    for c, idxi in by.items():
        idxc = np.sort(np.concatenate([np.arange(i*per,(i+1)*per) for i in idxi]))
        cat_series[c].append(round(mR(rt[idxc], ri[np.array(idxi)]),2))
    print(f"{a:6.2f}{allv:9.2f}{farmv:9.2f}")

# oracle: best alpha per category
oracle_rows = []
alpha_star = {}
for c in by:
    j = int(np.argmax(cat_series[c])); alpha_star[c] = float(alphas[j])
    oracle_rows.append((c, len(by[c]), float(alphas[j]), cat_series[c][j]))
S_or = np.zeros_like(Sg)
for c, idxi in by.items():
    a = alpha_star[c]
    S_or[:, idxi] = a*Sg[:, idxi] + (1-a)*Sp[:, idxi]
rt_o, ri_o = ranks_at(-1) if False else (None, None)
# compute ranks on oracle matrix directly
order = np.argsort(-S_or, axis=1)
rt_o = np.where(order == (np.arange(n_cap)//per)[:, None])[1]
orderi = np.argsort(-S_or.T, axis=1)
ri_o = np.empty(n_img, dtype=int)
for gi in range(n_img):
    ri_o[gi] = np.where(np.isin(orderi[gi], np.arange(gi*per,(gi+1)*per)))[0].min()
or_all = mR(rt_o, ri_o); or_farm = mR(rt_o[farm_cap], ri_o[farm_idx])
print(f"\nper-category oracle: ALL mR={or_all:.2f}, FARM mR={or_farm:.2f}, farmland alpha*={alpha_star['farmland']:.2f}")
print("top alpha* (global-preferring):", sorted(oracle_rows, key=lambda r:-r[2])[:5])
print("bottom alpha* (parcel-preferring):", sorted(oracle_rows, key=lambda r:r[2])[:5])

with open(os.path.join(OUT,"alpha_sweep.csv"),"w",newline="",encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["alpha","ALL_mR","FARM_mR"]); w.writerows(sweep)
with open(os.path.join(OUT,"alpha_oracle_per_category.csv"),"w",newline="",encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["category","n","oracle_alpha","oracle_mR"])
    for c,n,a,m in sorted(oracle_rows, key=lambda r:-r[2]): w.writerow([c,n,a,m])
json.dump({"oracle_ALL": round(or_all,2), "oracle_FARM": round(or_farm,2),
           "farmland_alpha_star": alpha_star["farmland"],
           "baseline_FARM": 23.51, "v2_FARM": 21.26, "v2_ALL": 34.49},
          open(os.path.join(OUT,"alpha_oracle.json"),"w"), indent=2)
print("saved.")
