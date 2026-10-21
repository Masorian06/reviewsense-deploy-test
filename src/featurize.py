#!/usr/bin/env python3
"""Stage 2 of the pipeline: turn documents into TF-IDF features.

deps     data/prepared/, src/featurize.py
params   featurize.max_features, featurize.ngrams   (params.yaml)
outs     data/features/  (train.pkl, test.pkl, vectorizer.pkl)
"""
import pickle
from pathlib import Path

import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer


def main() -> None:
    p = yaml.safe_load(open("params.yaml"))["featurize"]
    train = pd.read_csv("data/prepared/train.csv", keep_default_na=False)
    test = pd.read_csv("data/prepared/test.csv", keep_default_na=False)
    vec = TfidfVectorizer(max_features=p["max_features"], ngram_range=(1, p["ngrams"]),
                          min_df=2, sublinear_tf=True)
    X_train = vec.fit_transform(train.document)
    X_test = vec.transform(test.document)
    out = Path("data/features")
    out.mkdir(parents=True, exist_ok=True)
    pickle.dump((X_train, train.label.to_numpy()), open(out / "train.pkl", "wb"))
    pickle.dump((X_test, test.label.to_numpy()), open(out / "test.pkl", "wb"))
    pickle.dump(vec, open(out / "vectorizer.pkl", "wb"))
    print(f"featurize: {X_train.shape[0]:,} x {X_train.shape[1]:,} train matrix, ngrams 1-{p['ngrams']}")


if __name__ == "__main__":
    main()
