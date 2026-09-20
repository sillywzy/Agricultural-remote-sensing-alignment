"""Text-conditioned localization heatmaps: does RemoteCLIP know WHERE the farmland is?
For each farmland test image: sliding-window (5x5) similarity with text queries,
render heatmap overlays, and quantify spatial sensitivity (sim variance across patches).
"""
import os, csv, json, re
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
import open_clip

CKPT = r"work/baselines/RemoteCLIP-weights/RemoteCLIP-ViT-B-32.pt"
IMG_DIR = r"work/datasets/remoteclip-ret/test_images"
OUT = r"outputs/localization-heatmaps"
os.makedirs(OUT, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval().to(device)
tok = open_clip.get_tokenizer("ViT-B-32")

# ---- farmland test images + captions ----
rows = list(csv.DictReader(open(r"work/datasets/remoteclip-ret/rsicd_test.csv", encoding="utf-8-sig"), delimiter="\t"))
farm_files = sorted({r["filename"] for r in rows if re.sub(r"^rsicd_", "", r["filename"]).startswith("farmland_")})
print("farmland test images:", len(farm_files), flush=True)

QUERIES = ["farmland", "a large area of farmland", "road", "water", "buildings"]

def encode_texts(qs):
    with torch.no_grad():
        t = model.encode_text(tok(qs).to(device))
        return (t / t.norm(dim=-1, keepdim=True)).float().cpu()

txt_emb = encode_texts(QUERIES)

# sliding window 5x5 on 224 image: window 96, stride 32
def windows(im, W=96, S=32):
    patches, coords = [], []
    for y in range(0, im.height - W + 1, S):
        for x in range(0, im.width - W + 1, S):
            patches.append(im.crop((x, y, x+W, y+W))); coords.append((x, y, W))
    return patches, coords

def encode_patches(patches, bs=64):
    out = []
    for i in range(0, len(patches), bs):
        batch = torch.stack([preprocess(p) for p in patches[i:i+bs]]).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
            f = model.encode_image(batch); f = f / f.norm(dim=-1, keepdim=True)
        out.append(f.float().cpu())
    return torch.cat(out)

# jet-like colormap via numpy
def jet(v):  # v in [0,1] -> RGB uint8 HxWx3
    v = np.clip(v, 0, 1); r = np.clip(1.5 - abs(4*v - 3), 0, 1); g = np.clip(1.5 - abs(4*v - 2), 0, 1); b = np.clip(1.5 - abs(4*v - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)

def heatmap_overlay(im, sim_grid, qname):
    h = Image.fromarray(jet(sim_grid)).resize(im.size, Image.BICUBIC)
    base = im.convert("RGB").copy()
    blend = Image.blend(base, h, 0.55)
    d = ImageDraw.Draw(blend); d.text((6, 6), qname, fill=(255,255,0))
    return blend

summary = []
W, S = 96, 32; G = 5
for k, fn in enumerate(farm_files):
    im = Image.open(os.path.join(IMG_DIR, fn)).convert("RGB")
    patches, coords = windows(im, W, S)
    pe = encode_patches(patches)                     # (25,512)
    sims = (txt_emb @ pe.T).numpy()                  # (5,25)
    # per-query stats + heatmap for selected queries
    stats = {}
    for qi, q in enumerate(QUERIES):
        g = sims[qi].reshape(G, G)
        stats[q] = (float(g.min()), float(g.max()), float(g.std()))
        if q in ("farmland", "road", "water"):
            heatmap_overlay(im, g, q).save(os.path.join(OUT, f"{fn[:-4]}__{q.replace(' ','_')}.png"))
    row = {"filename": fn}
    for q in QUERIES:
        mn, mx, sd = stats[q]
        row[f"{q}_min"] = round(mn,4); row[f"{q}_max"] = round(mx,4); row[f"{q}_std"] = round(sd,4)
        row[f"{q}_range"] = round(mx-mn,4)
    summary.append(row)
    if (k+1) % 10 == 0: print(f"  {k+1}/{len(farm_files)}", flush=True)

with open(os.path.join(OUT,"localization_stats.csv"),"w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

# aggregate stats
print("\n===== SPATIAL SENSITIVITY (25 patches per image, 37 images) =====")
for q in QUERIES:
    rngs = [r[f"{q}_range"] for r in summary]; stds = [r[f"{q}_std"] for r in summary]
    print(f"{q:32s} sim-range mean={np.mean(rngs):.4f}  patch-std mean={np.mean(stds):.4f}")
print("\nheatmaps saved to", OUT, "| files:", len([f for f in os.listdir(OUT) if f.endswith('.png')]))
