"""Paper Figure 3: alpha sweep trade-off curve (ALL vs farmland) with key annotations."""
import csv, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = r"outputs/paper-figures"
rows = list(csv.DictReader(open(os.path.join(OUT,"alpha_sweep.csv"), encoding="utf-8-sig")))
a = np.array([float(r["alpha"]) for r in rows])
allm = np.array([float(r["ALL_mR"]) for r in rows])
farm = np.array([float(r["FARM_mR"]) for r in rows])

W, H = 1200, 760
L, Rm, T, Bm = 110, 1180, 90, 90
PW, PH = Rm-L, H-T-Bm
img = Image.new("RGB",(W,H),(255,255,255)); d = ImageDraw.Draw(img)
FB=r"C:\Windows\Fonts\arialbd.ttf"; FR=r"C:\Windows\Fonts\arial.ttf"
ft=ImageFont.truetype(FB,34); fax=ImageFont.truetype(FR,26); flab=ImageFont.truetype(FB,24); fleg=ImageFont.truetype(FR,24); fan=ImageFont.truetype(FR,20)

ymin, ymax = 15, 38
def X(v): return L + v*PW
def Y(v): return T + (ymax-v)/(ymax-ymin)*PH

d.text((L, 20), "Granularity trade-off: global vs parcel weight (alpha)", font=ft, fill=(0,0,0))
# grid + axes
for gy in range(16, 39, 2):
    yy = Y(gy); d.line([L,yy,Rm,yy], fill=(235,235,235), width=1)
    d.text((L-14,yy-12), str(gy), font=fan, fill=(110,110,110), anchor="ra")
d.line([L,T,L,T+PH], fill=(60,60,60), width=2); d.line([L,T+PH,Rm,T+PH], fill=(60,60,60), width=2)
for gx in np.arange(0,1.01,0.1):
    xx = X(gx); d.line([xx,T+PH,xx,T+PH+7], fill=(60,60,60), width=2)
    d.text((xx,T+PH+14), f"{gx:.1f}", font=fan, fill=(60,60,60), anchor="ma")
d.text((PW//2+L, H-24), "alpha (weight of global term)", font=fax, fill=(0,0,0), anchor="mm")
d.text((28, T+PH//2), "mR", font=fax, fill=(0,0,0), anchor="mm")

def line(vals, col, label_xy):
    pts = [(X(x),Y(y)) for x,y in zip(a,vals)]
    for p0,p1 in zip(pts[:-1],pts[1:]): d.line([p0,p1], fill=col, width=4)
    for x,y in pts: d.ellipse([x-5,y-5,x+5,y+5], fill=col)
line(allm,(66,118,200),None)
line(farm,(214,69,65),None)
# baseline lines
d.line([L,Y(32.53),Rm,Y(32.53)], fill=(66,118,200), width=2)
d.line([L,Y(23.51),Rm,Y(23.51)], fill=(214,69,65), width=2)
d.text((Rm-4,Y(32.53)-26),"ALL baseline 32.53",font=fan,fill=(66,118,200),anchor="ra")
d.text((Rm-4,Y(23.51)+8),"farmland baseline 23.51",font=fan,fill=(214,69,65),anchor="ra")
# key points
def mark(xv,yv,txt,col,dy=-30):
    x,y=X(xv),Y(yv); d.ellipse([x-8,y-8,x+8,y+8],outline=col,width=3)
    d.text((x+10,y+dy),txt,font=fleg,fill=col)
mark(0.70,24.41,"farmland best a*=0.70 -> 24.41",(150,20,20),-42)
mark(0.54,21.26,"learned gate (a=0.54) -> 21.26",(110,110,110),30)
# legend
lx,ly=130,T+14
d.rectangle([lx,ly,lx+34,ly+18],fill=(66,118,200)); d.text((lx+44,ly-3),"ALL categories mR",font=fleg,fill=(0,0,0))
d.rectangle([lx,ly+32,lx+34,ly+50],fill=(214,69,65)); d.text((lx+44,ly+29),"farmland subset mR (n=37)",font=fleg,fill=(0,0,0))
# caption line
d.text((L,H-58),"Oracle per-category alpha ceiling: ALL 37.15 / FARM 24.14  (vs baseline 32.53 / 23.51)",
       font=fleg,fill=(70,70,70))
img.save(os.path.join(OUT,"figure3_alpha_sweep.png"))
print("saved figure3_alpha_sweep.png")
