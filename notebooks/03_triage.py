"""Phase 3: assign a likely cause to every misclassified CIFAR-10 test image.

Inputs (from 02_cifar_features.py, in ./data):
  test_labels.npy, train_labels.npy, test_probs.npy, clip_test.npy, clip_train.npy
Output:
  web/data/cifar10.json  -- one item per test image: i, y, p, s; errors also get
                            cause, f_y, f_p and nbrs (indices into the TRAIN set)

Decision rule (fixed before evaluation; see README). Among the k neighbours, the
majority class must receive at least tau * k votes (default 6 of 10):
  if f_p >= tau    -> label      (neighbour majority = prediction)
  elif f_y >= tau  -> model      (neighbour majority = label; residual: errors not explained by the data)
  else             -> ambiguous  (no class reaches tau, a label/prediction tie, or the majority is a third class)
f_p doubles as the suspicion score used to order label errors.
The classifier score s is NOT used for assignment, only for ordering inside the view.

Usage:
  python notebooks/03_triage.py                          # defaults k=10 tau=0.6
  python notebooks/03_triage.py --emb resnet --no-export # ablation E6, no JSON written
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WEB = ROOT / "web" / "data"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
CAUSES = ("label", "ambiguous", "model")


# ------------------------------------------------------------------ core
def l2norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def knn(query, pool, k, dup_thresh=0.95, batch=1000):
    """Cosine kNN of each query row in pool. Neighbours with similarity > dup_thresh
    (near-duplicates between CIFAR-10 test and train) are dropped."""
    query, pool = l2norm(query), l2norm(pool)
    idx = np.empty((len(query), k), dtype=np.int64)
    for s in range(0, len(query), batch):
        S = query[s:s + batch] @ pool.T
        S[S > dup_thresh] = -np.inf
        part = np.argpartition(-S, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(S, part, 1), axis=1)
        idx[s:s + batch] = np.take_along_axis(part, order, 1)
    return idx


def assign(y, p, nbr_labels, tau=0.6):
    """Return (cause array, f_y, f_p) for errors with labels y, predictions p.
    f_p is also the suspicion score: the higher, the more likely the label is wrong."""
    f_y = (nbr_labels == y[:, None]).mean(1)
    f_p = (nbr_labels == p[:, None]).mean(1)
    cause = np.full(len(y), "ambiguous", dtype=object)
    # For tau > 0.5 at most one class can reach the threshold, so the order of checks does not matter.
    # For tau <= 0.5 a tie between label and prediction (e.g. 5 vs 5 of 10) counts as ambiguous.
    cause[(f_y >= tau) & (f_y > f_p)] = "model"
    cause[(f_p >= tau) & (f_p > f_y)] = "label"
    return cause, f_y, f_p


def cause_ranking(err_idx, cause, f_y, f_p):
    """Review order for 'find wrong labels': label errors first (strongest evidence first),
    then ambiguous, then model weakness. Within a cause, higher suspicion (f_p) first."""
    prio = np.array([CAUSES.index(c) for c in cause])
    order = np.lexsort((f_y, -f_p, prio))
    return err_idx[order]


def recall_at(ranking, truth, budget):
    return len(set(ranking[:budget].tolist()) & truth) / max(len(truth), 1)


# ------------------------------------------------------------------ answer key (eval only)
def load_confirmed_errors(path=DATA / "cifar10_mturk.json"):
    """Confirmed CIFAR-10 test label errors: candidates where fewer than 3 of 5 MTurk
    workers chose the given label. Verify the field names against the file once;
    the expected structure is a list of records with 'id' and 'mturk' -> {'given': n}."""
    if not path.exists():
        return None
    recs = json.load(open(path))
    try:
        return {int(r["id"]) for r in recs if r["mturk"]["given"] < 3}
    except (KeyError, TypeError):
        print("! cifar10_mturk.json has an unexpected structure; first record:",
              recs[0] if isinstance(recs, list) else list(recs)[:3])
        return None


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--emb", choices=["clip", "resnet"], default="clip")
    ap.add_argument("--no-export", action="store_true")
    a = ap.parse_args()

    y = np.load(DATA / "test_labels.npy")
    y_train = np.load(DATA / "train_labels.npy")
    probs = np.load(DATA / "test_probs.npy")
    emb_test = np.load(DATA / f"{a.emb}_test.npy")
    emb_train = np.load(DATA / f"{a.emb}_train.npy")

    p, s = probs.argmax(1), probs.max(1)
    err = np.flatnonzero(p != y)
    print(f"accuracy {1 - len(err) / len(y):.4f}, {len(err)} errors")

    nbrs = knn(emb_test[err], emb_train, a.k)
    cause, f_y, f_p = assign(y[err], p[err], y_train[nbrs], a.tau)
    for c in CAUSES:
        print(f"  {c:9s} {np.sum(cause == c):5d}")

    # quick checks against the real label errors (eval only, never fed back into the rule)
    truth = load_confirmed_errors()
    if truth is not None:
        hit = [i for i in truth if i in set(err.tolist())]
        print(f"confirmed label errors: {len(truth)}, of which misclassified by ResNet: {len(hit)}")
        pos = {int(i): j for j, i in enumerate(err)}
        print("  their assigned causes:",
              {c: int(sum(cause[pos[i]] == c for i in hit)) for c in CAUSES})
        ours = cause_ranking(err, cause, f_y, f_p)
        print(f"  recall@50  cause-based {recall_at(ours, truth, 50):.3f}")
        print(f"  recall@50  score-based {recall_at(err[np.argsort(-s[err])], truth, 50):.3f}")
        try:
            from cleanlab.filter import find_label_issues
            cl = find_label_issues(y, probs, return_indices_ranked_by="self_confidence")
            print(f"  recall@50  cleanlab    {recall_at(np.asarray(cl), truth, 50):.3f}")
        except ImportError:
            print("  (pip install cleanlab for the cleanlab baseline)")

    if a.no_export:
        return
    items = [{"i": int(i), "y": int(y[i]), "p": int(p[i]), "s": round(float(s[i]), 3)}
             for i in range(len(y))]
    for j, i in enumerate(err):
        items[i].update(cause=cause[j], f_y=round(float(f_y[j]), 2),
                        f_p=round(float(f_p[j]), 2), nbrs=nbrs[j].tolist())
    WEB.mkdir(parents=True, exist_ok=True)
    meta = {"k": a.k, "tau": a.tau, "emb": a.emb}
    with open(WEB / "cifar10.json", "w") as f:
        json.dump({"classes": CLASSES, "params": meta, "items": items}, f)
    print(f"wrote {WEB / 'cifar10.json'}")


if __name__ == "__main__":
    main()
