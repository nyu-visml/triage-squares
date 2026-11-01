"""Phase 1: MNIST predictions for validating our Squares implementation.

Downsamples MNIST to 7x7 (as in the Squares usage scenario), trains a random forest
and an SVM, and writes one JSON per model to web/data/. Each item has:
  i = index, y = label, p = predicted class, s = score of the predicted class.

Goal is the qualitative pattern from the paper (RF scores lower and more scattered,
SVM scores higher and more concentrated), not the paper's exact accuracies.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

OUT = Path(__file__).resolve().parents[1] / "web" / "data"
OUT.mkdir(parents=True, exist_ok=True)

X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
X = X.reshape(-1, 7, 4, 7, 4).mean(axis=(2, 4)).reshape(-1, 49) / 255  # 28x28 -> 7x7
y = y.astype(int)
Xtr, ytr, Xte, yte = X[:60000], y[:60000], X[60000:], y[60000:]

models = {
    "rf": RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1).fit(Xtr, ytr),
    "svm": SVC(probability=True, random_state=0).fit(Xtr, ytr),
}

for name, m in models.items():
    P = m.predict_proba(Xte)
    pred = P.argmax(1)
    print(f"{name} accuracy: {(pred == yte).mean():.4f}")
    items = [
        {"i": i, "y": int(yte[i]), "p": int(pred[i]), "s": round(float(P[i, pred[i]]), 3)}
        for i in range(len(yte))
    ]
    with open(OUT / f"mnist_{name}.json", "w") as f:
        json.dump({"classes": [str(c) for c in range(10)], "items": items}, f)
    print(f"wrote {OUT / f'mnist_{name}.json'}")
