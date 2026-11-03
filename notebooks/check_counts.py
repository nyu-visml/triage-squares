"""Phase 1 consistency check: per-class TP / FP / FN from sklearn's confusion matrix.
Compare with the table the web page logs in the browser console (Cmd+Option+J)."""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
files = sys.argv[1:] or [ROOT / "web/data/mnist_rf.json", ROOT / "web/data/mnist_svm.json"]

for f in files:
    d = json.load(open(f))
    y = np.array([it["y"] for it in d["items"]])
    p = np.array([it["p"] for it in d["items"]])
    cm = confusion_matrix(y, p, labels=range(len(d["classes"])))   # rows = label, cols = prediction
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp        # predicted as c, labelled otherwise  (right side, striped)
    fn = cm.sum(axis=1) - tp        # labelled c, predicted otherwise     (left side, outlined)
    print(f"\n{Path(f).name}  (n = {len(y)}, accuracy = {tp.sum() / len(y):.4f})")
    print(f"{'class':>12} {'TP':>6} {'FP':>6} {'FN':>6}")
    for c, name in enumerate(d["classes"]):
        print(f"{name:>12} {tp[c]:>6} {fp[c]:>6} {fn[c]:>6}")
    print(f"{'total':>12} {tp.sum():>6} {fp.sum():>6} {fn.sum():>6}")