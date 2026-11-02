"""Phase 2: run ONCE on Google Colab with a GPU runtime (T4 is enough).

Writes into ./data:
  test_labels.npy      (10000,)       CIFAR-10 test labels (the single label per image)
  test_probs.npy       (10000, 10)    ResNet20 softmax probabilities on the test set
  resnet_test.npy      (10000, 64)    ResNet20 penultimate features (for ablation E6)
  clip_test.npy        (10000, 512)   CLIP ViT-B/32 image features, L2-normalised
  cifar10h-probs.npy   (10000, 10)    answer key: human label distribution (eval only)
  cifar10_mturk.json                  answer key: validated label errors (eval only)
and web/data/sprite.jpg (100 x 100 grid of 32x32 test thumbnails, row-major by index).
Only the test set is needed: neighbours are found among the other test images.

Afterwards download data/*.npy, data/*.json and sprite.jpg to the shared drive / your laptop.
"""
import os
import urllib.request

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "data"
WEB = "web/data"
os.makedirs(OUT, exist_ok=True)
os.makedirs(WEB, exist_ok=True)
print("device:", DEV)

# ---------------------------------------------------------------- datasets
raw_test = torchvision.datasets.CIFAR10(OUT, train=False, download=True)
np.save(f"{OUT}/test_labels.npy", np.array(raw_test.targets))

# ---------------------------------------------------------------- ResNet20
# Normalisation used in the model author's training config. Forgetting it ruins accuracy.
norm = T.Compose([T.ToTensor(),
                  T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20",
                       pretrained=True).to(DEV).eval()

_feat = {}
model.fc.register_forward_hook(lambda m, inp, out: _feat.__setitem__("x", inp[0].detach()))


@torch.no_grad()
def run_resnet(train):
    ds = torchvision.datasets.CIFAR10(OUT, train=train, transform=norm)
    probs, feats = [], []
    for x, _ in DataLoader(ds, batch_size=500, num_workers=2):
        logits = model(x.to(DEV))
        probs.append(logits.softmax(1).cpu())
        feats.append(_feat["x"].cpu())
    return torch.cat(probs).numpy(), torch.cat(feats).numpy()


test_probs, resnet_test = run_resnet(train=False)
acc = (test_probs.argmax(1) == np.array(raw_test.targets)).mean()
print(f"ResNet20 test accuracy: {acc:.4f}  (expect ~0.92; much lower => check normalisation)")
np.save(f"{OUT}/test_probs.npy", test_probs.astype(np.float32))
np.save(f"{OUT}/resnet_test.npy", resnet_test.astype(np.float32))

# ---------------------------------------------------------------- CLIP
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEV).eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


@torch.no_grad()
def run_clip(dataset, bs=256):
    out = []
    for start in range(0, len(dataset), bs):
        imgs = [dataset[i][0] for i in range(start, min(start + bs, len(dataset)))]
        inputs = proc(images=imgs, return_tensors="pt").to(DEV)
        f = clip.get_image_features(**inputs)
        out.append(torch.nn.functional.normalize(f, dim=-1).cpu())
    return torch.cat(out).numpy().astype(np.float32)


np.save(f"{OUT}/clip_test.npy", run_clip(raw_test))
print("CLIP features saved")

# ---------------------------------------------------------------- thumbnail sprite
sprite = Image.new("RGB", (3200, 3200))
for i in range(10000):
    sprite.paste(raw_test[i][0], ((i % 100) * 32, (i // 100) * 32))
sprite.save(f"{WEB}/sprite.jpg", quality=90)

# ---------------------------------------------------------------- answer keys (eval only)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/jcpeterson/cifar-10h/master/data/cifar10h-probs.npy",
    f"{OUT}/cifar10h-probs.npy")
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/cleanlab/label-errors/main/mturk/cifar10_mturk.json",
    f"{OUT}/cifar10_mturk.json")
print("done")
