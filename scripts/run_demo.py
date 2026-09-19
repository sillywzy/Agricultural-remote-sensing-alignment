"""
最小演示：RemoteCLIP 图文对齐结果
输出：
  1) similarity_matrix.csv  —— 每张图 vs 每句文本的余弦相似度（越大越匹配）
  2) heatmap_<图片>.png     —— 文本短语在图像上的对齐热力图（7x7 patch 上采样叠加）
"""
import os, csv, torch, open_clip, numpy as np
from PIL import Image, ImageDraw
os.environ.setdefault("PYTHONIOENCODING","utf-8")

IMG_DIR = r"work/baselines/GeoChat/demo_images"
OUT_DIR = r"outputs/demo-results"
CKPT = r"work/baselines/RemoteCLIP-weights/RemoteCLIP-ViT-B-32.pt"

TEXTS = [
    "an aerial image of agricultural farmland with fields",
    "an aerial image of buildings in a town",
    "an aerial image of a railway station with trains",
    "an aerial image of a church building",
    "a satellite image of water or coastline",
]
HM_TEXTS = [TEXTS[0], TEXTS[1]]  # 每张图只画“农田”和“建筑”两个描述

device = "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval().to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

images = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith((".png",".jpg",".jpeg"))])

# ---------- patch token hook ----------
captured = {}
def hook(module, inp, out):
    captured["tokens"] = out[0] if isinstance(out, tuple) else out
h = model.visual.transformer.register_forward_hook(hook)

def jet_cmap(x):
    # x: HxW in [0,1] -> RGB uint8
    r = np.clip(1.5 - np.abs(4*x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4*x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4*x - 1), 0, 1)
    return (np.stack([r,g,b],-1)*255).astype(np.uint8)

def encode_patches(pil_img):
    img = preprocess(pil_img).unsqueeze(0).to(device)
    v = model.visual
    with torch.no_grad():
        x = v.conv1(img); gh, gw = x.shape[-2:]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0,2,1)
        cls = v.class_embedding.to(x.dtype) + torch.zeros(x.shape[0],1,x.shape[-1],dtype=x.dtype,device=x.device)
        x = torch.cat([cls, x], dim=1) + v.positional_embedding.to(x.dtype)
        norm_pre = getattr(v, "norm_pre", None) or getattr(v, "ln_pre", None)
        if norm_pre is not None: x = norm_pre(x)
        x = v.transformer(x)
        patch = x[:,1:,:]
        patch = v.ln_post(patch)
        if v.proj is not None: patch = patch @ v.proj
    return img, patch.reshape(gh,gw,-1).cpu().numpy()

with torch.no_grad():
    tok = tokenizer(TEXTS).to(device)
    tf = model.encode_text(tok)
    tf = tf / tf.norm(dim=-1, keepdim=True)

rows, matrix = [], []
for name in images:
    pil = Image.open(os.path.join(IMG_DIR,name)).convert("RGB")
    img_t, patch = encode_patches(pil)
    with torch.no_grad():
        gf = model.encode_image(img_t)
        gf = gf / gf.norm(dim=-1, keepdim=True)
        sims = (gf @ tf.T).squeeze(0).cpu().numpy()
    rows.append({"image": name, **{f"top{i+1}": "" for i in range(3)}})
    matrix.append([name] + [round(float(s),4) for s in sims])

    # ---- heatmaps ----
    pflat = patch.reshape(-1, patch.shape[-1])
    pflat = pflat / (np.linalg.norm(pflat, axis=1, keepdims=True)+1e-8)
    panels = []
    for t in HM_TEXTS:
        with torch.no_grad():
            te = model.encode_text(tokenizer([t]).to(device))
            te = (te/te.norm(dim=-1,keepdim=True)).squeeze(0).cpu().numpy()
        m = (pflat @ te).reshape(patch.shape[0], patch.shape[1])
        m = (m - m.min()) / (m.ptp()+1e-8)
        hm = Image.fromarray(jet_cmap(m)).resize((pil.width, pil.height), Image.BILINEAR)
        blend = Image.blend(pil, hm, 0.45)
        panels.append((t, blend))
    pad = 24
    canvas = Image.new("RGB", (pil.width*2, pil.height+pad*2), "white")
    d = ImageDraw.Draw(canvas)
    for i,(t,b) in enumerate(panels):
        canvas.paste(b, (i*pil.width, pad))
        d.text((i*pil.width+8, 6), ("farmland" if i==0 else "buildings")+": "+t[:42], fill="black")
    canvas.save(os.path.join(OUT_DIR, f"heatmap_{os.path.splitext(name)[0]}.png"))

# ---- CSV ----
with open(os.path.join(OUT_DIR,"similarity_matrix.csv"),"w",newline="",encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["image"] + TEXTS)
    w.writerows(matrix)

h.remove()
print("images:", len(images))
print("saved to", OUT_DIR)
for r in matrix:
    best = int(np.argmax(r[1:]))
    print(f"{r[0]:28s} best-match -> {TEXTS[best][:50]}  (sim={r[1:][best]:.3f})")
print("DONE")
