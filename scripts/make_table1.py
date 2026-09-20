"""Generate paper Table 1 (main results) as CSV + markdown."""
import csv, os
OUT = r"outputs/paper-figures"
rows = [
    # method, ALL_mR, FARM_mR, note
    ("RemoteCLIP ViT-B/32 (baseline)", 32.53, 23.51, "global alignment, zero-shot"),
    ("Naive 2x2 tiling max",           18.63, 16.31, "ablation: fixed grid, no learning"),
    ("ParcelAlign v1 (fixed alpha)",   34.20, 19.55, "learned parcels, scalar alpha=0.48"),
    ("v2 gate w/o centering",          32.07, 18.20, "ablation: gate collapses to alpha->0"),
    ("ParcelAlign v2 (centered gate)", 34.49, 21.26, "adaptive per-image alpha (ours)"),
]
with open(os.path.join(OUT, "table1_main_results.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["method","ALL_mR","FARM_mR","note"])
    for r in rows: w.writerow(r)
md = ["| Method | ALL mR | FARM mR | Note |", "|---|---|---|---|"]
for m, a, fm, n in rows:
    md.append(f"| {m} | {a:.2f} | {fm:.2f} | {n} |")
md += ["", "FARM = farmland test subset (37 images). mR = mean of TR@1/5/10 and IR@1/5/10.",
       "v2 farmland 95% bootstrap CI: [13.42, 29.55]; baseline: [16.22, 30.99] (overlap, not significant)."]
open(os.path.join(OUT, "table1_main_results.md"), "w", encoding="utf-8").write("\n".join(md))
print("\n".join(md))
