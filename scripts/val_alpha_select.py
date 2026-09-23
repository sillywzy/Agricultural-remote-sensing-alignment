"""Val-selected alpha (no test leakage): sweep alpha on RSICD val split, then report
the selected alpha on the held-out test split, with full per-category breakdown."""
import sys, os, json, re, csv, collections
sys.path.insert(0, r"work")
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
import open_clip
from models.parcel_align_v2 import ParcelAlignV2

RUN = r"outputs/parcel-align/run_v2_20260920_211323"
CKPT = r"work/baselines/RemoteCLIP-weights/RemoteCLIP-ViT-B-32.pt"
DATA_JSON = r"work/datasets/RSICD_optimal/dataset_rsicd.json"
IMG = r"work/datasets/RSICD_optimal/RSICD_images"
NPZ_TEST = r"work/emb_cache_gpu/RSICD_b32.npz"
IMG_TEST = r"work/datasets/remoteclip-ret/test_images"
CLSDIR = r"work/datasets/RSICD_optimal/txtclasses/txtclasses_rsicd"
OUT = r"outputs/paper-figures"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

data = json.load(open(DATA_JSON, encoding="utf-8"))["images"]
val = [e for e in data if e["split"] == "val"]

model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
pa = ParcelAlignV2(model, k=8).to(DEVICE)
ck = torch.load(os.path.join(RUN, "best_trainable.pt"), map_location="cpu")
pa.tokenizer.load_state_dict(ck["tokenizer"]); pa.gate.load_state_dict(ck["gate"])
pa.logit_g.data = ck["logit_g"].to(DEVICE); pa.logit_p.data = ck["logit_p"].to(DEVICE)
pa.eval()
tok = open_clip.get_tokenizer("ViT-B-32")

def encode_split(entries, imgdir):
    files = [e["filename"] for e in entries]
    caps = []
    for e in entries:
        caps.extend(s["raw"] for s in e["sentences"][:5])
    clss, parcs = [], []
    for i in range(0, len(files), 64):
        imgs = torch.stack([pre(Image.open(os.path.join(imgdir, f)).convert("RGB")) for f in files[i:i+64]]).to(DEVICE)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            c, p, _ = pa.encode_image_parcels(imgs)
        clss.append(c.float().cpu()); parcs.append(p.float().cpu())
    texts = []
    for i in range(0, len(caps), 512):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            t = pa.clip.encode_text(tok(caps[i:i+512]).to(DEVICE)); t = F.normalize(t, dim=-1)
        texts.append(t.float().cpu())
    return files, torch.cat(clss).cuda(), torch.cat(parcs).cuda(), torch.cat(texts).cuda()

def terms(cls_n, parc_n, text_n):
    with torch.no_grad():
        return (text_n @ cls_n.T).cpu().numpy(), torch.einsum("id,jkd->ijk", text_n, parc_n).max(dim=2).values.cpu().numpy()

def ranks(S, per):
    n_cap = S.shape[0]
    order = np.argsort(-S, axis=1)
    rt = np.where(order == (np.arange(n_cap)//per)[:, None])[1]
    orderi = np.argsort(-S.T, axis=1)
    n_img = S.shape[1]; ri = np.empty(n_img, dtype=int)
    for gi in range(n_img):
        ri[gi] = np.where(np.isin(orderi[gi], np.arange(gi*per,(gi+1)*per)))[0].min()
    return rt, ri
def mR(rt, ri): return float(np.mean([(rt<k).mean()*100 for k in (1,5,10)] + [(ri<k).mean()*100 for k in (1,5,10)]))

name2cls = {}
for tf in os.listdir(CLSDIR):
    if tf.endswith(".txt"):
        c = tf[:-4].lower()
        if c == "playfields": c = "playground"
        for line in open(os.path.join(CLSDIR, tf), encoding="utf-8", errors="ignore"):
            if line.strip(): name2cls[line.strip()] = c

def subset_idxs(files, per):
    cats = [name2cls[f] for f in files]
    by = collections.defaultdict(list)
    for i,c in enumerate(cats): by[c].append(i)
    fi = np.array(sorted(by["farmland"]))
    fc = np.sort(np.concatenate([np.arange(i*per,(i+1)*per) for i in fi]))
    return cats, by, fi, fc

print("encoding val...")
vfiles, vc, vp, vt = encode_split(val, IMG)
Sg_v, Sp_v = terms(vc, vp, vt)
vcats, vby, vfi, vfc = subset_idxs(vfiles, 5)

alphas = np.round(np.arange(0, 1.0001, 0.05), 2)
val_rows = []
print(f"{'alpha':>6s}{'VAL_ALL':>9s}{'VAL_FARM':>9s}")
for a in alphas:
    rt, ri = ranks(a*Sg_v + (1-a)*Sp_v, 5)
    va, vf = mR(rt,ri), mR(rt[vfc], ri[vfi])
    val_rows.append((float(a), round(va,2), round(vf,2)))
    print(f"{a:6.2f}{va:9.2f}{vf:9.2f}")
a_all = float(alphas[int(np.argmax([r[1] for r in val_rows]))])
a_farm = float(alphas[int(np.argmax([r[2] for r in val_rows]))])
print(f"\nval-selected: alpha*_ALL={a_all:.2f}  alpha*_FARM={a_farm:.2f}")

print("encoding test...")
zt = np.load(NPZ_TEST, allow_pickle=True)
tfiles = list(zt["imgs"]); tcap = list(zt["caps"])
# strip rsicd_ prefix to map disk names
clss, parcs = [], []
for i in range(0, len(tfiles), 64):
    imgs = torch.stack([pre(Image.open(os.path.join(IMG_TEST, f)).convert("RGB")) for f in tfiles[i:i+64]]).to(DEVICE)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        c, p, _ = pa.encode_image_parcels(imgs)
    clss.append(c.float().cpu()); parcs.append(p.float().cpu())
texts = []
for i in range(0, len(tcap), 512):
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        t = pa.clip.encode_text(tok(tcap[i:i+512]).to(DEVICE)); t = F.normalize(t, dim=-1)
    texts.append(t.float().cpu())
Sg_t, Sp_t = terms(torch.cat(clss).cuda(), torch.cat(parcs).cuda(), torch.cat(texts).cuda())
tcats, tby, tfi, tfc = subset_idxs([re.sub(r"^rsicd_","",f) for f in tfiles], 5)
n_img = len(tfiles)

def eval_alpha(a):
    rt, ri = ranks(a*Sg_t + (1-a)*Sp_t, 5)
    return mR(rt,ri), mR(rt[tfc], ri[tfi]), rt, ri
t_all, t_farm, _, _ = eval_alpha(a_all)
print(f"\nTEST report:")
print(f"  baseline (a=1.00):       ALL 32.53  FARM 23.51")
print(f"  learned gate (~0.50):    ALL 34.49  FARM 21.26")
print(f"  val-alpha*_ALL={a_all:.2f}:      ALL {t_all:.2f}  FARM {t_farm:.2f}")
ta2, tf2, rt_f, ri_f = eval_alpha(a_farm)
print(f"  val-alpha*_FARM={a_farm:.2f}:     ALL {ta2:.2f}  FARM {tf2:.2f}")

# per-category at val-selected ALL alpha
rt, ri = ranks(a_all*Sg_t + (1-a_all)*Sp_t, 5)
per_cat = []
for c, idxi in sorted(tby.items()):
    idxi = np.array(sorted(idxi)); idxc = np.sort(np.concatenate([np.arange(i*5,(i+1)*5) for i in idxi]))
    per_cat.append((c, len(idxi), round(mR(rt[idxc], ri[idxi]),2)))

# bootstrap farmland CI at val-selected alpha
rng = np.random.RandomState(42); vals = []
for _ in range(2000):
    samp = rng.choice(tfi, size=len(tfi), replace=True)
    sampc = np.concatenate([np.arange(i*5,(i+1)*5) for i in samp])
    vals.append(mR(rt[sampc], ri[samp]))
ci = (round(float(np.percentile(vals,2.5)),2), round(float(np.percentile(vals,97.5)),2))

with open(os.path.join(OUT,"val_alpha_selection.csv"),"w",newline="",encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["split","alpha","ALL_mR","FARM_mR"]); w.writerows(val_rows)
with open(os.path.join(OUT,"val_selected_test_per_category.csv"),"w",newline="",encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["category","n","test_mR_at_val_alpha"]); w.writerows(per_cat)
rep = {"val_alpha_ALL": a_all, "val_alpha_FARM": a_farm,
       "test_at_alpha_ALL": {"ALL_mR": round(t_all,2), "FARM_mR": round(t_farm,2), "farm_CI95": ci},
       "test_at_alpha_FARM": {"ALL_mR": round(ta2,2), "FARM_mR": round(tf2,2)},
       "baseline": {"ALL_mR":32.53,"FARM_mR":23.51},
       "v2_learned_gate": {"ALL_mR":34.49,"FARM_mR":21.26}}
json.dump(rep, open(os.path.join(OUT,"val_selection_report.json"),"w"), indent=2)
print(f"\nfarmland 95% CI at val-selected alpha: {ci}")
print("saved:", OUT)
