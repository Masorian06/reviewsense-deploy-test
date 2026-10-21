#!/usr/bin/env python3
"""Stage 3 of the pipeline: train the classifier.

deps     data/features/, src/train.py
params   train.C, train.seed                  (params.yaml)
outs     models/model.pkl

Experiment tracking (block 03) is optional and switched on by the environment:

    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000      # or sqlite:///mlflow.db

Every run is then logged to MLflow with its parameters, the test F1, the Git
commit and the hash of the data it was trained on (read from data/reviews.csv.dvc),
which joins the DVC lineage to the MLflow run table.
"""
import os
import pickle
import subprocess
from pathlib import Path

import yaml
from sklearn.linear_model import LogisticRegression


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    p = params["train"]
    X_train, y_train = pickle.load(open("data/features/train.pkl", "rb"))
    model = LogisticRegression(C=p["C"], max_iter=1000, random_state=p["seed"])
    model.fit(X_train, y_train)
    Path("models").mkdir(exist_ok=True)
    pickle.dump(model, open("models/model.pkl", "wb"))
    print(f"train: logistic regression, C={p['C']}, {X_train.shape[0]:,} documents")
    if os.environ.get("MLFLOW_TRACKING_URI"):
        track(model, params)
    else:
        print("train: MLFLOW_TRACKING_URI not set, run not logged")


def track(model, params: dict) -> None:
    import mlflow
    import mlflow.sklearn
    import pandas as pd
    from sklearn.metrics import f1_score

    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    data_md5 = yaml.safe_load(open("data/reviews.csv.dvc"))["outs"][0]["md5"]
    X_test, y_test = pickle.load(open("data/features/test.pkl", "rb"))
    mlflow.set_experiment("reviewsense")
    tags = {"git_commit": commit, "data_md5": data_md5}                      # the join to DVC
    if os.environ.get("DVC_EXP_NAME"):                                       # set by `dvc exp run`
        tags["dvc_exp"] = os.environ["DVC_EXP_NAME"]
    with mlflow.start_run(run_name=tags.get("dvc_exp", f"logreg-C{params['train']['C']}")):
        mlflow.set_tags(tags)
        mlflow.log_params({**params["featurize"], **params["train"]})
        train_df = pd.read_csv("data/prepared/train.csv", keep_default_na=False)
        mlflow.log_input(mlflow.data.from_pandas(train_df, source="data/reviews.csv",
                                                 name=f"reviews-{data_md5[:6]}"), context="training")
        mlflow.log_metric("f1_macro", f1_score(y_test, model.predict(X_test), average="macro"))
        info = mlflow.sklearn.log_model(model, name="model", input_example=X_test[:5])
        mlflow.log_artifact("data/features/vectorizer.pkl")               # the model needs its vectorizer
        print(f"train: logged to MLflow, run {mlflow.active_run().info.run_id[:8]}, model {info.model_id}")


if __name__ == "__main__":
    main()
