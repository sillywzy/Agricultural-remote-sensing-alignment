import zipfile, os, csv, json, collections
root=r"work/datasets/RSITMD"
os.mirs = None
os.makedirs(os.path.join(root,"images"), exist_ok=True)

# 1) extract 452 test + 20 train originals, strip rsitmd_ prefix
n=0
for zpath in [r"work/datasets/remoteclip-ret/Ret-3_test.zip", r"work/datasets/remoteclip-ret/Ret-3_train.zip"]:
    z=zipfile.ZipFile(zpath)
    for x in z.namelist():
        if x.startswith("rsitmd_") and not x.endswith("/"):
            orig=x.replace("rsitmd_","",1)
            # train zip also contains augmented high-index files; RSITMD originals are indices 0..471
            import re
            m=re.search(r"_(\d+)\.png$", orig)
            if m and int(m.group(1)) <= 471:
                dst=os.path.join(root,"images",orig)
                if not os.path.exists(dst):
                    open(dst,"wb").write(z.read(x)); n+=1
print("images assembled:", n, "| total on disk:", len(os.listdir(os.path.join(root,"images"))))

# 2) captions
test_rows=list(csv.DictReader(open(r"work/datasets/remoteclip-ret/rsitmd_test.csv",encoding="utf-8-sig"), delimiter="\t"))
train_all=list(csv.DictReader(open(r"work/datasets/remoteclip-ret/Ret-3_train.csv",encoding="utf-8-sig"), delimiter="\t"))
train_names={"rsitmd_"+f for f in os.listdir(os.path.join(root,"images")) if int(re.search(r"_(\d+)\.",f).group(1))>=452}
train_rows=[r for r in train_all if r["filename"] in train_names]
# dedupe train captions (csv may repeat)
seen=set(); train_rows=[r for r in train_rows if not (r["filename"]+r["title"] in seen or seen.add(r["filename"]+r["title"]))]
print("test caption rows:",len(test_rows),"| train caption rows:",len(train_rows))

# 3) build dataset_RSITMD.json (AMFMN-style: images->sentences)
def build(rows, split):
    by=collections.OrderedDict()
    for r in rows:
        fn=r["filename"].replace("rsitmd_","",1)
        by.setdefault(fn,[]).append(r["title"].strip())
    out=[]
    for i,(fn,caps) in enumerate(by.items()):
        out.append({"filename":fn,"imgid":i,"split":split,
                    "sentences":[{"raw":c,"tokens":c.lower().replace(".","").split()} for c in caps]})
    return out
images=build(train_rows,"train")+build(test_rows,"test")
data={"images":images,"dataset":"RSITMD"}
json.dump(data, open(os.path.join(root,"dataset_RSITMD.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("json images:",len(images),"| train:",sum(1 for x in images if x['split']=='train'),"| test:",sum(1 for x in images if x['split']=='test'))

# 4) unified caption csv
with open(os.path.join(root,"captions_all.csv"),"w",newline="",encoding="utf-8-sig") as f:
    w=csv.writer(f, delimiter="\t"); w.writerow(["title","filename","split"])
    for r in train_rows: w.writerow([r["title"].strip(), r["filename"].replace("rsitmd_","",1),"train"])
    for r in test_rows: w.writerow([r["title"].strip(), r["filename"].replace("rsitmd_","",1),"test"])
print("DONE")
