# Triage Squares: Is It the Model or the Data?

DS-GA 3001 Visualization for Machine Learning · Final project
Qixin Fu (qf2078) · Wenshan Zhang (wz3191)

Triage Squares extends **Squares** (Ren et al., IEEE TVCG 2017) so that every misclassified
instance is attributed to a likely cause — **label error**, **inherent ambiguity**, or
**model weakness (residual)** — using a neighbourhood vote in an independent CLIP embedding
space. The main analysis set is the CIFAR-10 test set (10,000 images).

## Pipeline

```
CIFAR-10 test ──► ResNet20 ──► predicted class p, score s ─┐
        │                                                  ├─► kNN + rules ─► web/data/cifar10.json ─► D3 view
        └────────► CLIP ViT-B/32 ◄── CIFAR-10 train (neighbour pool)
Answer keys (CIFAR-10H, confirmed label errors, synthetic flips) ──► evaluation only
```

The system never sees the answer keys; they are used only in `04_eval`.

## Repository layout

| Path | What it is |
| --- | --- |
| `notebooks/01_mnist_export.py` | MNIST (7×7) random forest + SVM → JSON, only to validate our Squares implementation |
| `notebooks/02_cifar_features.py` | Run once on Colab (GPU): ResNet20 probabilities, CLIP features, thumbnail sprite, answer keys |
| `notebooks/03_triage.py` | kNN in CLIP space + decision rules → `web/data/cifar10.json`; quick checks and cleanlab baseline |
| `notebooks/04_eval.ipynb` | Evaluation E1–E6 (to be added) |
| `web/` | D3 Squares view (`index.html`, `squares.js`, `style.css`) and the JSON it reads |
| `data/` | Raw data and `.npy` features — **not tracked by git** (shared drive) |
| `results/` | Figures for the report |

## Decision rule (initial version)

For each misclassified test image with label `y` and prediction `p`, find the `k` nearest
CIFAR-10 *training* images in CLIP space (cosine; near-duplicates with similarity > 0.95 removed).
Let `f_y` / `f_p` be the fraction of neighbours labelled `y` / `p`. The neighbour majority
must reach the vote threshold `tau` (default: at least 6 of 10 neighbours).

```
if f_p >= tau    -> label error       (neighbour majority = prediction)
elif f_y >= tau  -> model weakness    (neighbour majority = label; residual)
else             -> ambiguous         (no class reaches tau, a tie, or the majority is a third class)
defaults: k = 10, tau = 0.6
```

`f_p` is also the suspicion score used to order label errors for review.

Sensitivity is checked over k ∈ {5, 10, 20, 50} × tau ∈ {0.5, 0.6, 0.7, 0.8}. With tau > 0.5 the
three categories are mutually exclusive regardless of the order of checks; at tau = 0.5 a 5–5 tie
between label and prediction counts as ambiguous.

## How to run

```bash
pip install -r requirements.txt

# Phase 1 – MNIST baseline (local)
# python notebooks/01_mnist_export.py

# Phase 2 – upload 02_cifar_features.py to Colab (GPU runtime), run it,
#           then download the files it writes into data/ and web/data/sprite.jpg

# Phase 3 – cause assignment (local)
# python notebooks/03_triage.py --k 10 --tau 0.6

# View
cd web && python -m http.server 8000   # open http://localhost:8000
```

## Data sources

- CIFAR-10 — https://www.cs.toronto.edu/~kriz/cifar.html
- Pretrained ResNet20 — https://github.com/chenyaofo/pytorch-cifar-models
- CLIP ViT-B/32 — https://huggingface.co/openai/clip-vit-base-patch32
- CIFAR-10H — https://github.com/jcpeterson/cifar-10h
- Validated label errors — https://github.com/cleanlab/label-errors

## Status

- [x] Phase 1 · Baseline Squares on MNIST
- [ ] Phase 2 · CIFAR-10 predictions and CLIP features
- [ ] Phase 3 · Cause assignment + cleanlab baseline
- [ ] Phase 4 · Cause-aware view
- [ ] Phase 5 · Evaluation (E1–E6) and case studies
- [ ] Phase 6 · Final report and demo
