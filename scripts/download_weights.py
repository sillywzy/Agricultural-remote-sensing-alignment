import urllib.request, ssl, os, sys, time
url = "https://hf-mirror.com/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-B-32.pt"
dst = r"work/baselines/RemoteCLIP-weights/RemoteCLIP-ViT-B-32.pt"
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=60, context=ctx) as r, open(dst, "wb") as f:
    total = int(r.getheader("Content-Length") or 0); got = 0; t0=time.time(); last=0
    while True:
        chunk = r.read(1024*1024)
        if not chunk: break
        f.write(chunk); got += len(chunk)
        if time.time()-last > 5:
            last=time.time()
            sp = got/1024/1024/max(time.time()-t0,1e-9)
            print(f"{got/1024/1024:.1f}/{total/1024/1024:.1f} MB  {sp:.1f} MB/s", flush=True)
print("DONE", os.path.getsize(dst), flush=True)
