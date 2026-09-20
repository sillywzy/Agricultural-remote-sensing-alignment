"""Paper Figure 2: per-category learned alpha (granularity gate behavior).
Bars colored by whether the category benefited from parcels in v1
(red = declined with parcels / texture-global, blue = improved / object-centric).
If the gate learned the right thing, red bars should sit higher."""
import json, csv, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

V2 = r"outputs/parcel-align/run_v2_20260920_211323"
V1 = r"outputs/parcel-align/run_20260919_220949"
OUT = r"outputs/paper-figures"; os.makedirs(OUT, exist_ok=True)

alpha = json.load(open(os.path.join(V2, "final_report.json"), encoding="utf-8"))["alpha_by_category"]
v1 = {r["category"]: float(r["delta"]) for r in csv.DictReader(open(os.path.join(V1, "perclass_compare.csv"), encoding="utf-8-sig"))}

cats = sorted(alpha.items(), key=lambda kv: kv[1])
FB = r"C:\Windows\Fonts\arialbd.ttf"; FR = r"C:\Windows\Fonts\arial.ttf"
f_cat = ImageFont.truetype(FR, 22); f_val = ImageFont.truetype(FR, 20)
f_title = ImageFont.truetype(FB, 30); f_leg = ImageFont.truetype(FR, 20)

W, H = 1150, 300 + len(cats) * 34
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((30, 18), "Learned granularity gate: per-category alpha (v2)", font=f_title, fill=(0, 0, 0))
d.text((30, 62), "alpha > 0.5: trust global   |   alpha < 0.5: trust parcels   |   color: red = hurt by parcels in v1, blue = helped",
       font=f_leg, fill=(90, 90, 90))

L, R = 250, 1060
top = 110
for i, (c, a) in enumerate(cats):
    y = top + i * 34
    v1d = v1.get(c, 0.0)
    col = (214, 69, 65) if v1d < 0 else (66, 118, 200)
    lbl = c + ("  [FARMLAND]" if c == "farmland" else "")
    d.text((L - 12, y + 4), lbl, font=f_cat, fill=(0, 0, 0), anchor="ra")
    x0 = L
    x1 = L + (a - 0.40) / (0.60 - 0.40) * (R - L)
    d.rectangle([x0, y, x1, y + 26], fill=col)
    d.text((x1 + 8, y + 2), f"{a:.3f}", font=f_val, fill=(60, 60, 60))

# axis
d.line([L, top - 8, L, top + len(cats) * 34], fill=(120, 120, 120), width=2)
for v in (0.40, 0.45, 0.50, 0.55, 0.60):
    x = L + (v - 0.40) / 0.20 * (R - L)
    d.line([x, top + len(cats) * 34, x, top + len(cats) * 34 + 8], fill=(120, 120, 120), width=2)
    d.text((x, top + len(cats) * 34 + 12), f"{v:.2f}", font=f_val, fill=(60, 60, 60), anchor="ma")
d.text((L, H - 34), "mean alpha = 0.500 (centered gate); higher = more trust in the global representation",
       font=f_leg, fill=(120, 120, 120))

img.save(os.path.join(OUT, "figure2_alpha_category.png"))
print("saved:", os.path.join(OUT, "figure2_alpha_category.png"), f"({W}x{H})")

# also correlation stat
va = np.array([alpha[c] for c in alpha]); vd = np.array([v1.get(c, 0.0) for c in alpha])
m = (vd != 0)
corr = np.corrcoef(va[m], vd[m])[0, 1]
print(f"corr(alpha, v1_delta) = {corr:.3f}  (negative = gate correctly raises alpha where parcels hurt)")
