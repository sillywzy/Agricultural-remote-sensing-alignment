"""Quantitative localization analysis + montage generation.
Q1: Do different concepts produce DIFFERENT heatmaps? (corr between farmland/road/water grids)
Q2: Is the response spatially concentrated or diffuse? (fraction of near-max patches)
Q3: Center bias? (top patch position distribution)
Also builds side-by-side montages: original | farmland | road | water
"""
import os, csv, re, json
import numpy as np, torch
from PIL import Image, ImageDraw
import open_clip

CKPT = r"work/baselines/RemoteCLIP-weights/RemoteCLIP-ViT-B-32.pt"
IMG_DIR = r"work/datasets/remoteclip-ret/test_images"
OUT = r"outputs/localization-heatmaps"
MON = os.path.join(OUT, "montage"); os.makedirs(MON, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval().to(device)
tok = open_clip.get_tokenizer("ViT-B-32")

rows = list(csv.DictReader(open(r"work/datasets/remoteclip-ret/rsicd_test.csv", encoding="utf-8-sig"), delimiter="\t"))
farm_files = sorted({r["filename"] for r in rows if re.sub(r"^rsicd_","",r["filename"]).startswith("farmland_")})
QUERIES = ["farmland", "road", "water"]
with torch.no_grad():
    t = model.encode_text(tok(QUERIES).to(device)); txt_emb = (t/t.norm(dim=-1,keepdim=True)).float().cpu()

def jet(v):
    v = np.clip(v,0,1); r=np.clip(1.5-abs(4*v-3),0,1); g=np.clip(1.5-abs(4*v-2),0,1); b=np.clip(1.5-abs(4*v-1),0,1)
    return (np.stack([r,g,b],-1)*255).astype(np.uint8)

W,S,G = 96,32,5
res = []
for fn in farm_files:
    im = Image.open(os.path.join(IMG_DIR, fn)).convert("RGB")
    patches = [im.crop((x,y,x+W,y+W)) for y in range(0,im.height-W+1,S) for x in range(0,im.width-W+1,S)]
    batch = torch.stack([preprocess(p) for p in patches]).to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
        f = model.encode_image(batch); f = f/f.norm(dim=-1,keepdim=True)
    pe = f.float().cpu()
    sims = (txt_emb @ pe.T).numpy()          # (3,25)
    grids = [s.reshape(G,G) for s in sims]
    fq, rq, wq = grids

    # Q1 concept separability: 1 - corr (higher = concepts fire in different places)
    def corr(a,b):
        a=a.flatten(); b=b.flatten()
        if a.std()<1e-6 or b.std()<1e-6: return 0.0
        return float(np.corrcoef(a,b)[0,1])
    c_fr = corr(fq,rq); c_fw = corr(fq,wq); c_rw = corr(rq,wq)

    # Q2 diffuseness: fraction of patches within 80% of range from max
    def diffuse(g):
        g=g.flatten(); rng=g.max()-g.min()
        if rng<1e-6: return 1.0
        return float(((g.max()-g)/rng <= 0.2).mean())
    d_f = diffuse(fq)

    # Q3 top-1 patch position (5x5 grid, 0-indexed; center=12)
    top = int(np.argmax(fq.flatten()))

    # montage: orig + 3 heatmaps
    panels = [im.convert("RGB")]
    for g,q in zip(grids,QUERIES):
        h = Image.fromarray(jet(g)).resize(im.size, Image.BICUBIC)
        b = Image.blend(im.convert("RGB"), h, 0.55)
        ImageDraw.Draw(b).text((6,6), q, fill=(255,255,0))
        panels.append(b)
    m = Image.new("RGB", (im.width*4+30, im.height+10), (20,20,20))
    for i,p in enumerate(panels): m.paste(p,(i*(im.width+10)+5,5))
    m.save(os.path.join(MON, f"{fn[:-4]}__montage.png"))

    res.append({"filename":fn,"corr_farmland_road":round(c_fr,3),"corr_farmland_water":round(c_fw,3),
                "corr_road_water":round(c_rw,3),"diffuse_frac_farmland":round(d_f,2),"top1_patch":top})

with open(os.path.join(OUT,"localization_quant.json"),"w",encoding="utf-8") as f: json.dump(res,f,indent=2)

cf = np.mean([r["corr_farmland_road"] for r in res])
cw = np.mean([r["corr_farmland_water"] for r in res])
cr = np.mean([r["corr_road_water"] for r in res])
df_ = np.mean([r["diffuse_frac_farmland"] for r in res])
tops = [r["top1_patch"] for r in res]
center_frac = sum(1 for t in tops if t in (6,7,8,11,12,13,16,17,18))/len(tops)

print("===== LOCALIZATION QUANTITATIVE RESULT (37 farmland test images) =====")
print(f"mean corr(farmland, road)   = {cf:.3f}   (1=identical heatmap, 0=uncorrelated, neg=complementary)")
print(f"mean corr(farmland, water)  = {cw:.3f}")
print(f"mean corr(road, water)      = {cr:.3f}")
print(f"diffuse fraction (farmland) = {df_:.2f}  (fraction of patches within 80% of max; 0.2=sharp, 1.0=totally flat)")
print(f"top-1 patch in center 3x3   = {center_frac:.2f}  (9/25=0.36 expected if uniform)")
print(f"montages saved: {len(os.listdir(MON))} -> {MON}")
