import torch, open_clip
from PIL import Image
import numpy as np
print("torch", torch.__version__, "| open_clip", open_clip.__version__)
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
ckpt = torch.load(r"work/baselines/RemoteCLIP-weights/RemoteCLIP-ViT-B-32.pt", map_location="cpu")
msg = model.load_state_dict(ckpt, strict=False)
print("missing:", len(msg.missing_keys), "unexpected:", len(msg.unexpected_keys))
model.eval()
tok = open_clip.get_tokenizer("ViT-B-32")
img = preprocess(Image.fromarray((np.random.rand(224,224,3)*255).astype("uint8"))).unsqueeze(0)
texts = tok(["a photo of farmland", "a photo of a river"])
with torch.no_grad():
    fi = model.encode_image(img); ft = model.encode_text(texts)
    fi = fi/fi.norm(dim=-1,keepdim=True); ft = ft/ft.norm(dim=-1,keepdim=True)
    sim = (fi @ ft.T).squeeze(0)
print("similarity:", [round(float(x),4) for x in sim])
print("SMOKE TEST PASSED")
