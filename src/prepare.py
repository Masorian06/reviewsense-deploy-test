#!/usr/bin/env python3
"""Stage 1 of the pipeline: label, clean and split the reviews.

deps     data/reviews.csv, src/prepare.py
params   prepare.split, prepare.seed          (params.yaml)
outs     data/prepared/train.csv, data/prepared/test.csv

Label: 4-5 stars -> 1 (positive), 1-2 stars -> 0 (negative). 3-star reviews are
dropped as neutral. The document is the review title followed by its text.
"""
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def main() -> None:
    p = yaml.safe_load(open("params.yaml"))["prepare"]
    df = pd.read_csv("data/reviews.csv", keep_default_na=False, dtype={"title": str, "text": str})
    df = df[df.rating != 3].copy()
    df["label"] = (df.rating >= 4).astype(int)
    df["document"] = (df.title.str.strip() + ". " + df.text.str.strip()).str.strip(". ")
    df["document"] = df.document.str.replace(r"\s+", " ", regex=True)
    df = df[df.document.str.len() >= p.get("min_chars", 5)]
    cols = ["review_id", "category", "label", "document"]
    train, test = train_test_split(df[cols], test_size=p["split"], random_state=p["seed"], stratify=df.label)
    out = Path("data/prepared")
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False, lineterminator="\n")
    test.to_csv(out / "test.csv", index=False, lineterminator="\n")
    print(f"prepare: {len(df):,} labelled reviews -> train {len(train):,} / test {len(test):,}, "
          f"positive share {df.label.mean():.1%}")


if __name__ == "__main__":
    main()
