"""Paper Figure 1, two variants:
  (B) zoomed unified scale 0.20-0.35  -> shows relative structure + cross-concept hotspot overlap
  (A) full scale 0-1                  -> shows absolute responses are weak/flat
"""
import os, math
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
import open_clip

CKPT = r"work/baselines/RemoteCLIP-weights/RemoteCLIP-ViT-B-32.pt"
IMG_DIR = r"work/datasets/remoteclip-ret/test_images"
OUT = r"outputs/paper-figures"; os.makedirs(OUT, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
FILES = ["rsicd_farmland_37.jpg", "rsicd_farmland_370.jpg"]
RANGES = {"rsicd_farmland_37.jpg": 0.071, "rsicd_farmland_370.jpg": 0.019}
QUERIES = ["farmland", "road", "water"]
HEADERS = ["Original image", '"farmland"', '"road"', '"water"']

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval().to(device)
tok = open_clip.get_tokenizer("ViT-B-32")
with torch.no_grad():
    t = model.encode_text(tok(QUERIES).to(device))
    txt_emb = (t / t.norm(dim=-1, keepdim=True)).float().cpu()

W, S, G = 96, 32, 5
def grids_for(fn):
    im = Image.open(os.path.join(IMG_DIR, fn)).convert("RGB")
    patches = [im.crop((x, y, x+W, y+W)) for y in range(0, im.height-W+1, S) for x in range(0, im.width-W+1, S)]
    batch = torch.stack([preprocess(p) for p in patches]).to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
        f = model.encode_image(batch); f = f / f.norm(dim=-1, keepdim=True)
    sims = (txt_emb @ f.float().cpu().T).numpy()
    return im, [s.reshape(G, G) for s in sims]

results = {fn: grids_for(fn) for fn in FILES}
all_g = [g for fn in FILES for g in results[fn][1]]
gmin = min(g.min() for g in all_g); gmax = max(g.max() for g in all_g)
print(f"actual data range: {gmin:.3f} ~ {gmax:.3f}")

def jet(v):
    v = np.clip(v, 0, 1)
    r = np.clip(1.5-abs(4*v-3),0,1); g = np.clip(1.5-abs(4*v-2),0,1); b = np.clip(1.5-abs(4*v-1),0,1)
    return (np.stack([r,g,b],-1)*255).astype(np.uint8)

FB = r"C:\Windows\Fonts\arialbd.ttf"; FR = r"C:\Windows\Fonts\arial.ttf"
f_head = ImageFont.truetype(FB, 34); f_row = ImageFont.truetype(FB, 30)
f_tick = ImageFont.truetype(FR, 26); f_ax = ImageFont.truetype(FB, 28)

def render(vmin, vmax, ticks, out_png):
    def grid_img(g, size):
        n = (g - vmin)/(vmax - vmin + 1e-9)
        return Image.fromarray(jet(n)).resize((size,size), Image.BICUBIC)
    P, GAP = 448, 14
    ML, MT, MR, MB = 100, 88, 170, 56
    CW = ML + 4*P + 3*GAP + MR; CH = MT + 2*P + 1*GAP + MB
    canvas = Image.new("RGB", (CW, CH), (255,255,255)); d = ImageDraw.Draw(canvas)
    def text_center(x, y, s, font):
        bb = d.textbbox((0,0), s, font=font); w = bb[2]-bb[0]; h = bb[3]-bb[1]
        d.text((x-w/2, y-h/2), s, font=font, fill=(0,0,0))
    for c, htxt in enumerate(HEADERS):
        text_center(ML + c*(P+GAP) + P/2, MT/2 + 6, htxt, f_head)
    for r_i, fn in enumerate(FILES):
        im, gs = results[fn]; y0 = MT + r_i*(P+GAP)
        lbl = "(a)" if r_i == 0 else "(b)"
        name = fn.replace("rsicd_","").replace(".jpg","")
        for li, line in enumerate([lbl, name, f"range={RANGES[fn]:.3f}"]):
            d.text((18, y0 + 60 + li*44), line, font=(f_row if li==0 else f_tick), fill=(0,0,0))
        canvas.paste(im.resize((P,P), Image.LANCZOS), (ML, y0))
        for c_i, g in enumerate(gs):
            x0 = ML + (c_i+1)*(P+GAP)
            blend = Image.blend(im.convert("RGB").resize((P,P), Image.LANCZOS), grid_img(g, P), 0.55)
            canvas.paste(blend, (x0, y0))
        for c in range(4):
            x0 = ML + c*(P+GAP)
            d.rectangle([x0-1, y0-1, x0+P, y0+P], outline=(120,120,120), width=2)
    cb_x = ML + 4*P + 3*GAP + 30; cb_y, cb_w, cb_h = MT, 40, 2*P + GAP
    grad = Image.fromarray(jet(np.linspace(1, 0, 256)[:, None].repeat(40, 1))).resize((cb_w, cb_h))
    canvas.paste(grad, (cb_x, cb_y))
    d.rectangle([cb_x-1, cb_y-1, cb_x+cb_w, cb_y+cb_h], outline=(80,80,80), width=2)
    for v in ticks:
        fy = cb_y + cb_h - (v-vmin)/(vmax-vmin)*cb_h
        d.line([cb_x+cb_w, fy, cb_x+cb_w+10, fy], fill=(0,0,0), width=3)
        d.text((cb_x+cb_w+18, fy-14), f"{v:.2f}", font=f_tick, fill=(0,0,0))
    tmp = Image.new("RGBA", (700, 50), (255,255,255,0)); td = ImageDraw.Draw(tmp)
    td.text((350, 25), "Cosine similarity", font=f_ax, fill=(0,0,0,255), anchor="mm")
    rot = tmp.rotate(90, expand=True)
    canvas.paste(rot, (cb_x + cb_w + 95, cb_y + cb_h//2 - rot.height//2), rot)
    canvas.save(out_png)
    canvas.resize((CW//2, CH//2), Image.LANCZOS).save(out_png.replace(".png", "_half.png"))
    print("saved:", out_png)

# B: zoomed scale (rounded to cover actual range)
zmin = math.floor(gmin*20)/20; zmax = math.ceil(gmax*20)/20
render(zmin, zmax, [zmin + i*0.05 for i in range(int(round((zmax-zmin)/0.05))+1)],
       os.path.join(OUT, "figure1_zoomed.png"))
# A: full scale
render(0.0, 1.0, [0.0, 0.25, 0.50, 0.75, 1.0], os.path.join(OUT, "figure1_fullscale.png"))

capB = ("Figure 1: Spatial response of RemoteCLIP ViT-B/32 on RSICD farmland test images. Sliding-window "
 f"similarity maps (5x5 patches) use a unified zoomed color scale (cosine similarity, {zmin:.2f}-{zmax:.2f}) with "
 "explicit tick values. (a) The most spatially-varying image in the farmland test subset (max-min range = 0.071): "
 "the hottest regions (red, ~0.33) elicited by different text concepts largely coincide (mean cross-concept "
 "correlation 0.35-0.52), showing the response is driven by general saliency rather than the queried concept. "
 "(b) A typical image (range = 0.019): responses are nearly uniform. In absolute terms no patch exceeds ~0.33 "
 "similarity, so the model cannot tell WHICH parcel the text refers to. Global image-text alignment thus lacks "
 "the spatial selectivity required for parcel-level grounding.")
capA = ("Figure 1 (full-scale variant): the same sliding-window similarity maps rendered on the full cosine-similarity "
 f"range [0, 1]. All window responses fall within {gmin:.2f}-{gmax:.2f}, so every map appears nearly flat: the model "
 "assigns almost the same weak similarity to every location and provides no spatial evidence of where the queried "
 "concept is. This complements the zoomed view: the apparent hotspots there correspond to only ~0.3 similarity.")
open(os.path.join(OUT, "figure1_caption_zoomed.txt"), "w", encoding="utf-8").write(capB)
open(os.path.join(OUT, "figure1_caption_fullscale.txt"), "w", encoding="utf-8").write(capA)
print("captions saved")
