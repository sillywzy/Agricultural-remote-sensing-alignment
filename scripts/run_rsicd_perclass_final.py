"""Final per-category RSICD retrieval results; authoritative class map from txtclasses."""
import os, csv, json, re, numpy as np, torch, collections

CLSDIR = r"work/datasets/RSICD_optimal/txtclasses/txtclasses_rsicd"
OUT = r"outputs/baseline-results"

# 1) filename(without rsicd_ prefix) -> class
name2cls = {}; collisions = []
for tf in os.listdir(CLSDIR):
    if not tf.endswith(".txt"): continue
    cls = tf[:-4].lower()
    if cls == "playfields": cls = "playground"
    for line in open(os.path.join(CLSDIR, tf), encoding="utf-8", errors="ignore"):
        nm = line.strip()
        if not nm: continue
        if nm in name2cls and name2cls[nm] != cls: collisions.append((nm, name2cls[nm], cls))
        name2cls[nm] = cls

z = np.load(r"work/emb_cache_gpu/RSICD_b32.npz", allow_pickle=True)
im_emb = torch.tensor(z["img"]); tx_emb = torch.tensor(z["txt"])
imgs = list(z["imgs"]); caps = list(z["caps"])
n_img, n_cap = len(imgs), len(caps); per = n_cap // n_img

S = (tx_emb @ im_emb.T).numpy()
gt_img = np.arange(n_cap) // per
rank_ti = np.where(np.argsort(-S, axis=1) == gt_img[:, None])[1]
orderi = np.argsort(-S.T, axis=1)
best_rank = np.empty(n_img, dtype=int)
for gi in range(n_img):
    gts = np.arange(gi*per, (gi+1)*per)
    best_rank[gi] = np.where(np.isin(orderi[gi], gts))[0].min()

def inner(fn):
    return re.sub(r"^rsicd_", "", fn)

missing = [f for f in imgs if inner(f) not in name2cls]
print("class collisions:", set(collisions), "| unmapped test images:", len(missing), missing[:5])
cats = [name2cls[inner(f)] for f in imgs]

rows_csv = list(csv.DictReader(open(r"work/datasets/remoteclip-ret/rsicd_test.csv", encoding="utf-8-sig"), delimiter="\t"))
caps_text = []
i_cap = 0
for fn in imgs:
    g = [r["title"].strip() for r in rows_csv if r["filename"] == fn][:per]
    caps_text.append(g)

def block(idxi, idxc):
    rt = rank_ti[idxc]; ri = best_rank[idxi]
    d = {"n_images": len(idxi), "TR@1":(rt<1).mean()*100,"TR@5":(rt<5).mean()*100,"TR@10":(rt<10).mean()*100,
         "IR@1":(ri<1).mean()*100,"IR@5":(ri<5).mean()*100,"IR@10":(ri<10).mean()*100,
         "TR_medianRank":float(np.median(rt)+1),"IR_medianRank":float(np.median(ri)+1)}
    d["mR"] = float(np.mean([d["TR@1"],d["TR@5"],d["TR@10"],d["IR@1"],d["IR@5"],d["IR@10"]]))
    return d

by = collections.defaultdict(list)
for i,c in enumerate(cats): by[c].append(i)
rows = {c: block(np.array(sorted(v)), np.sort(np.concatenate([np.arange(i*per,(i+1)*per) for i in v]))) for c,v in by.items()}
rows["ALL"] = block(np.arange(n_img), np.arange(n_cap))

cols = ["category","n_images","TR@1","TR@5","TR@10","IR@1","IR@5","IR@10","mR","TR_medianRank","IR_medianRank"]
order = sorted([c for c in rows if c!="ALL"], key=lambda c: -rows[c]["mR"]) + ["ALL"]
with open(os.path.join(OUT,"rsicd_per_category.csv"),"w",newline="",encoding="utf-8-sig") as f:
    w=csv.writer(f); w.writerow(cols)
    for c in order:
        v=rows[c]; w.writerow([c,v["n_images"],round(v["TR@1"],2),round(v["TR@5"],2),round(v["TR@10"],2),
                              round(v["IR@1"],2),round(v["IR@5"],2),round(v["IR@10"],2),round(v["mR"],2),
                              v["TR_medianRank"],v["IR_medianRank"]])
json.dump(rows, open(os.path.join(OUT,"rsicd_per_category.json"),"w"), indent=2)

# 2) farmland-specific outputs
farm = [i for i,c in enumerate(cats) if c=="farmland"]
farm_idx = np.array(sorted(farm))
farm_cap_idx = np.sort(np.concatenate([np.arange(i*per,(i+1)*per) for i in farm]))
summary = block(farm_idx, farm_cap_idx); summary["note"]="farmland subset, ranks computed over the full 1093-image gallery"
json.dump(summary, open(os.path.join(OUT,"rsicd_farmland_summary.json"),"w"), indent=2)
with open(os.path.join(OUT,"rsicd_farmland_per_image.csv"),"w",newline="",encoding="utf-8-sig") as f:
    w=csv.writer(f); w.writerow(["filename","IR_rank(best of 5 captions)","caption_idx","caption","TR_rank(caption->image)"])
    for gi in farm_idx:
        ranks = rank_ti[gi*per:(gi+1)*per]
        for k,(t,rr) in enumerate(zip(caps_text[gi], ranks)):
            w.writerow([imgs[gi], int(best_rank[gi]+1), k+1, t, int(rr+1)])

print(f"{'category':20s}{'n':>4s}{'TR@1':>7s}{'TR@5':>7s}{'TR@10':>7s}{'IR@1':>7s}{'IR@5':>7s}{'IR@10':>7s}{'mR':>7s}")
for c in order:
    v=rows[c]; mark=" <<< FARMLAND" if c=="farmland" else ""
    print(f"{c:20s}{v['n_images']:4d}{v['TR@1']:7.2f}{v['TR@5']:7.2f}{v['TR@10']:7.2f}{v['IR@1']:7.2f}{v['IR@5']:7.2f}{v['IR@10']:7.2f}{v['mR']:7.2f}{mark}")
print("\nFARMLAND summary saved. n =", summary["n_images"], "| mR =", round(summary["mR"],2),
      "| TR@1/5/10 =", [round(summary[k],2) for k in ("TR@1","TR@5","TR@10")],
      "| IR@1/5/10 =", [round(summary[k],2) for k in ("IR@1","IR@5","IR@10")])
