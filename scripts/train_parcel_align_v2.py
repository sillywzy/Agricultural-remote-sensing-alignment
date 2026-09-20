"""Train ParcelAlign v2 (adaptive granularity gate) on RSICD; warm start from v1.
Final analysis: per-category metrics vs baseline, per-category learned alpha,
farmland bootstrap CI. One run produces everything."""
import os, sys, json, math, time, random, csv, re, collections
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import open_clip
from models.parcel_align_v2 import ParcelAlignV2

SEED, EPOCHS, BS, LR, WD, K, CLIP_GRAD = 42, 6, 64, 1e-4, 0.01, 8, 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT = os.path.join(ROOT, "work", "baselines", "RemoteCLIP-weights", "RemoteCLIP-ViT-B-32.pt")
INIT_FROM = os.path.join(ROOT, "outputs", "parcel-align", "run_20260919_220949", "best_trainable.pt")
TRAIN_JSON = os.path.join(ROOT, "work", "datasets", "RSICD_optimal", "dataset_rsicd.json")
TRAIN_IMGDIR = os.path.join(ROOT, "work", "datasets", "RSICD_optimal", "RSICD_images")
EVAL_NPZ = os.path.join(ROOT, "work", "emb_cache_gpu", "RSICD_b32.npz")
EVAL_IMGDIR = os.path.join(ROOT, "work", "datasets", "remoteclip-ret", "test_images")
CLSDIR = os.path.join(ROOT, "work", "datasets", "RSICD_optimal", "txtclasses", "txtclasses_rsicd")
OUT = os.path.join(ROOT, "outputs", "parcel-align", "run_v2_" + time.strftime("%Y%m%d_%H%M%S"))
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
def encode_eval(pa, pre, tok, files, caps):
    clss, parcs = [], []
    for i in range(0, len(files), 64):
        imgs = torch.stack([pre(Image.open(os.path.join(EVAL_IMGDIR, f)).convert("RGB"))
                            for f in files[i:i+64]]).to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE=="cuda")):
            c, p, al = pa.encode_image_parcels(imgs)
        clss.append(c.float().cpu()); parcs.append(p.float().cpu())
    texts = []
    for i in range(0, len(caps), 512):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE=="cuda")):
            t = pa.clip.encode_text(tok(list(caps[i:i+512])).to(DEVICE))
            t = F.normalize(t, dim=-1)
        texts.append(t.float().cpu())
    return torch.cat(clss), torch.cat(parcs), torch.cat(texts)

def score_all(pa, text_n, cls_n, parc_n, chunk=1024):
    with torch.no_grad():
        a = pa.gate(cls_n.to(DEVICE), parc_n.to(DEVICE)).cpu()      # (n_img,)
    tg = pa.logit_g.exp().item(); tp = pa.logit_p.exp().item()
    c = cls_n.to(DEVICE); p = parc_n.to(DEVICE); aw = a.to(DEVICE).unsqueeze(0)
    rows = []
    for i in range(0, text_n.shape[0], chunk):
        t = text_n[i:i+chunk].to(DEVICE)
        Sg = t @ c.T
        Sp = torch.einsum("id,jkd->ijk", t, p).max(dim=2).values
        rows.append((aw * tg * Sg + (1 - aw) * tp * Sp).cpu())
    return torch.cat(rows).numpy(), a.numpy()

def ranks_from_S(S, per):
    gt = np.arange(S.shape[0]) // per
    order = np.argsort(-S, axis=1)
    rt = np.where(order == gt[:, None])[1]
    orderi = np.argsort(-S.T, axis=1)
    ri = np.empty(S.shape[1], dtype=int)
    for gi in range(S.shape[1]):
        gts = np.arange(gi*per, (gi+1)*per)
        ri[gi] = np.where(np.isin(orderi[gi], gts))[0].min()
    return rt, ri

def blk(r_t, r_i):
    d = {"TR@1":(r_t<1).mean()*100,"TR@5":(r_t<5).mean()*100,"TR@10":(r_t<10).mean()*100,
         "IR@1":(r_i<1).mean()*100,"IR@5":(r_i<5).mean()*100,"IR@10":(r_i<10).mean()*100}
    d["mR"] = float(np.mean(list(d.values())))
    return d

def save_trainable(pa, path):
    torch.save({"tokenizer": pa.tokenizer.state_dict(),
                "gate": pa.gate.state_dict(),
                "logit_g": pa.logit_g.detach().cpu(),
                "logit_p": pa.logit_p.detach().cpu()}, path)

if __name__ == "__main__":
    set_seed(SEED)
    log(f"output: {OUT}\ndevice: {DEVICE}")
    json.dump({"seed":SEED,"epochs":EPOCHS,"bs":BS,"lr":LR,"wd":WD,"k":K,"init_from":INIT_FROM},
              open(os.path.join(OUT,"config.json"),"w"), indent=2)
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    pa = ParcelAlignV2(model, k=K).to(DEVICE)
    if os.path.exists(INIT_FROM):
        pa.load_v1(INIT_FROM); log(f"warm start from v1: {INIT_FROM}")
    tok = open_clip.get_tokenizer("ViT-B-32")
    log(f"trainable params: {sum(p.numel() for p in pa.parameters() if p.requires_grad)/1e6:.2f}M")

    z = np.load(EVAL_NPZ, allow_pickle=True)
    files = list(z["imgs"]); caps = list(z["caps"])
    n_img, n_cap = len(files), len(caps); per = n_cap // n_img
    name2cls = {}
    for tf in os.listdir(CLSDIR):
        if tf.endswith(".txt"):
            cls = tf[:-4].lower()
            if cls == "playfields": cls = "playground"
            for line in open(os.path.join(CLSDIR, tf), encoding="utf-8", errors="ignore"):
                nm = line.strip()
                if nm: name2cls[nm] = cls
    cats = [name2cls[re.sub(r"^rsicd_","",f)] for f in files]
    by = collections.defaultdict(list)
    for i, c in enumerate(cats): by[c].append(i)
    farm_idx = np.array(sorted(by["farmland"]))
    farm_cap = np.sort(np.concatenate([np.arange(i*per,(i+1)*per) for i in farm_idx]))
    log(f"eval: {n_img} imgs / {n_cap} caps | farmland {len(farm_idx)}")

    ds = RSICDPairs(TRAIN_IMGDIR, pre)
    dl = DataLoader(ds, batch_size=BS, shuffle=True, num_workers=0, collate_fn=collate)
    steps_total = EPOCHS * len(dl)
    trainable = [p for p in pa.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5*(1+math.cos(math.pi*s/steps_total)))
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))

    rescols = ["epoch","loss","loss_global","loss_parcel","loss_div",
               "alpha_mean","alpha_std","alpha_min","alpha_max",
               "ALL_TR@1","ALL_TR@5","ALL_TR@10","ALL_IR@1","ALL_IR@5","ALL_IR@10","ALL_mR",
               "FARM_TR@1","FARM_TR@5","FARM_TR@10","FARM_IR@1","FARM_IR@5","FARM_IR@10","FARM_mR","parcel_div"]
    fres = open(os.path.join(OUT,"results.csv"),"w",newline="",encoding="utf-8-sig")
    wres = csv.DictWriter(fres, fieldnames=rescols); wres.writeheader()

    best_mR, best_ep = -1, -1
    for ep in range(1, EPOCHS+1):
        pa.train(); pa.clip.eval()
        agg = collections.defaultdict(float); nb = 0; t0 = time.time()
        for it, (imgs, caps_b) in enumerate(dl, 1):
            imgs = imgs.to(DEVICE, non_blocking=True)
            txts = tok(caps_b).to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE=="cuda")):
                out = pa(imgs, txts)
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, CLIP_GRAD)
            scaler.step(opt); scaler.update(); sched.step()
            for kk in ("loss","loss_global","loss_parcel","loss_div",
                       "alpha_mean","alpha_std","alpha_min","alpha_max"):
                v = out[kk]
                agg[kk] += v.item() if torch.is_tensor(v) else v
            nb += 1
            if it % 40 == 0 or it == len(dl):
                log(f"  ep{ep} it{it}/{len(dl)} loss={agg['loss']/nb:.3f} p={agg['loss_parcel']/nb:.3f} "
                    f"a[{agg['alpha_min']/nb:.2f},{agg['alpha_max']/nb:.2f}] mean={agg['alpha_mean']/nb:.3f} "
                    f"lr={sched.get_last_lr()[0]:.1e} {time.time()-t0:.0f}s")
        pa.eval()
        cls_n, parc_n, text_n = encode_eval(pa, pre, tok, files, caps)
        S, alphas = score_all(pa, text_n, cls_n, parc_n)
        rt, ri = ranks_from_S(S, per)
        allm = blk(rt, ri); farmm = blk(rt[farm_cap], ri[farm_idx])
        sel = np.random.RandomState(0).choice(len(parc_n), min(256,len(parc_n)), replace=False)
        ps = parc_n[torch.tensor(sel)]
        spp = torch.einsum("mkd,mjd->mkj", ps, ps)
        pdiv = spp[:, ~torch.eye(K, dtype=torch.bool)].mean().item()
        row = {"epoch": ep}
        for kk in ("loss","loss_global","loss_parcel","loss_div","alpha_mean","alpha_std","alpha_min","alpha_max"):
            row[kk] = round(agg[kk]/nb, 4)
        for k_, v in allm.items(): row[f"ALL_{k_}"] = round(v, 2)
        for k_, v in farmm.items(): row[f"FARM_{k_}"] = round(v, 2)
        row["parcel_div"] = round(pdiv, 4)
        wres.writerow(row); fres.flush()
        save_trainable(pa, os.path.join(OUT, f"epoch{ep}_trainable.pt"))
        if allm["mR"] > best_mR: best_mR, best_ep = allm["mR"], ep
        log(f"ep{ep} EVAL  ALL mR={allm['mR']:.2f}  FARM mR={farmm['mR']:.2f}  "
            f"alpha: mean={alphas.mean():.3f} farm={alphas[farm_idx].mean():.3f}  div={pdiv:.3f}")
    save_trainable(pa, os.path.join(OUT, "best_trainable.pt"))

    # ---------- final analysis ----------
    # baseline ranks from cached embeddings
    S_bl = (text_n @ cls_n.T).numpy() if False else None
    zb = np.load(EVAL_NPZ, allow_pickle=True)
    S_bl = (torch.tensor(zb["txt"]) @ torch.tensor(zb["img"]).T).numpy()
    rt_bl, ri_bl = ranks_from_S(S_bl, per)

    M_pa = {c: blk(rt[np.sort(np.concatenate([np.arange(i*per,(i+1)*per) for i in np.array(sorted(v))]))],
                   ri[np.array(sorted(v))]) for c, v in by.items()}
    M_pa["ALL"] = blk(rt, ri)
    M_bl = {c: blk(rt_bl[np.sort(np.concatenate([np.arange(i*per,(i+1)*per) for i in np.array(sorted(v))]))],
                   ri_bl[np.array(sorted(v))]) for c, v in by.items()}
    M_bl["ALL"] = blk(rt_bl, ri_bl)

    # per-category learned alpha
    alpha_by_cat = {c: float(alphas[np.array(sorted(v))].mean()) for c, v in by.items()}

    # farmland bootstrap CI
    rng = np.random.RandomState(42)
    def boot(rt_, ri_):
        vals = []
        for _ in range(2000):
            samp = rng.choice(farm_idx, size=len(farm_idx), replace=True)
            sampc = np.concatenate([np.arange(i*per,(i+1)*per) for i in samp])
            vals.append(blk(rt_[sampc], ri_[samp])["mR"])
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    ci_pa = boot(rt, ri); ci_bl = boot(rt_bl, ri_bl)

    order = sorted([c for c in M_pa if c != "ALL"], key=lambda c: M_pa[c]["mR"] - M_bl[c]["mR"])
    with open(os.path.join(OUT,"perclass_compare.csv"),"w",newline="",encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["category","n","BL_mR","PA_mR","delta","alpha_mean"])
        for c in order + ["ALL"]:
            n = len(by[c]) if c != "ALL" else n_img
            w.writerow([c, n, round(M_bl[c]["mR"],2), round(M_pa[c]["mR"],2),
                        round(M_pa[c]["mR"]-M_bl[c]["mR"],2),
                        round(alpha_by_cat.get(c, alphas.mean()),3) if c != "ALL" else round(alphas.mean(),3)])

    print(f"\n{'category':20s}{'n':>4s}{'BL_mR':>7s}{'V2_mR':>7s}{'delta':>7s}{'alpha':>7s}")
    for c in order + ["ALL"]:
        n = len(by[c]) if c != "ALL" else n_img
        al = alpha_by_cat.get(c, alphas.mean())
        mk = " <<< FARMLAND" if c == "farmland" else ""
        print(f"{c:20s}{n:4d}{M_bl[c]['mR']:7.2f}{M_pa[c]['mR']:7.2f}{M_pa[c]['mR']-M_bl[c]['mR']:+7.2f}{al:7.3f}{mk}")
    print(f"\nFARMLAND CI: v2 [{ci_pa[0]:.2f},{ci_pa[1]:.2f}] vs baseline [{ci_bl[0]:.2f},{ci_bl[1]:.2f}]")
    json.dump({"best_epoch":best_ep,"best_ALL_mR":best_mR,
               "final_ALL_mR":M_pa["ALL"]["mR"],"final_FARM_mR":M_pa["farmland"]["mR"],
               "baseline_ALL_mR":M_bl["ALL"]["mR"],"baseline_FARM_mR":M_bl["farmland"]["mR"],
               "farmland_ci_v2":ci_pa,"farmland_ci_baseline":ci_bl,
               "alpha_by_category":alpha_by_cat,"alpha_overall_mean":float(alphas.mean()),
               "alpha_farmland_mean":float(alphas[farm_idx].mean())},
              open(os.path.join(OUT,"final_report.json"),"w"), indent=2)
    log(f"DONE. best ep{best_ep} ALL mR={best_mR:.2f} (v1: 34.20, baseline: 32.53)")
    log(f"final report: {os.path.join(OUT,'final_report.json')}")
