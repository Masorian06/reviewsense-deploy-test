#!/usr/bin/env python3
"""Stage 4 of the pipeline: score the model on the test split.

deps     models/model.pkl, data/features/, src/evaluate.py
metrics  eval/metrics.json   (small, kept in Git; compare runs with `dvc metrics diff`)

f1_macro is the headline number. Precision and recall of the negative class are
what the support team cares about: unhappy customers that the triage misses.
"""
import json
import pickle
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def main() -> None:
    model = pickle.load(open("models/model.pkl", "rb"))
    X_test, y_test = pickle.load(open("data/features/test.pkl", "rb"))
    pred = model.predict(X_test)
    metrics = {
        "f1_macro": round(float(f1_score(y_test, pred, average="macro")), 4),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision_negative": round(float(precision_score(y_test, pred, pos_label=0)), 4),
        "recall_negative": round(float(recall_score(y_test, pred, pos_label=0)), 4),
        "n_test": int(len(y_test)),
    }
    Path("eval").mkdir(exist_ok=True)
    with open("eval/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    print("evaluate:", json.dumps(metrics))


if __name__ == "__main__":
    main()
