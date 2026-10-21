#!/usr/bin/env python3
"""ReviewSense REST API: the champion model behind two HTTP endpoints.

    export MLFLOW_TRACKING_URI=...                       # where the registry lives
    flask --app app run --debug --port 8000              # development server, this laptop only
    gunicorn -w 4 -b 127.0.0.1:8000 "app:create_app()"   # production server

GET  /health       -> {"status": "ok", "model": "reviewsense", "version": "3"}
POST /v1/predict   <- {"texts": ["Great product", "Broke after two days"]}
                   -> {"predictions": [{"label": "positive", "p_positive": 0.962}, ...],
                       "model": "reviewsense", "version": "3", "latency_ms": 3.8}
"""

import os
import pickle
import time

import mlflow
import mlflow.sklearn
from flask import Flask, jsonify, request
from mlflow import MlflowClient
from werkzeug.exceptions import HTTPException

NAME = "reviewsense"
MODEL_URI = os.environ.get("MODEL_URI", f"models:/{NAME}@champion")  # MLFLOW_TRACKING_URI: environment too
MAX_TEXTS = 100


def load_model(uri=MODEL_URI):
    """Fetch the aliased model, its vectorizer (same run) and its version number."""
    model = mlflow.sklearn.load_model(uri)  # sklearn flavor: we need predict_proba
    v = MlflowClient().get_model_version_by_alias(NAME, uri.split("@")[1])
    path = mlflow.artifacts.download_artifacts(run_id=v.run_id, artifact_path="vectorizer.pkl")
    with open(path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer, str(v.version)  # model + its vectorizer, same run


def create_app(loader=load_model):
    """Application factory: tests pass a fake loader, production uses the registry."""
    app = Flask(__name__)
    model, vectorizer, version = loader()  # once, here; never inside a request
    app.config.update(MODEL=model, VECTORIZER=vectorizer, VERSION=version)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": NAME, "version": app.config["VERSION"]}  # a dict becomes JSON

    @app.post("/v1/predict")
    def predict():
        body = request.get_json(silent=True)  # None if the body is not JSON
        texts = body.get("texts") if isinstance(body, dict) else None
        if not isinstance(texts, list) or not texts or not all(isinstance(t, str) and t.strip() for t in texts):
            return jsonify(error='body must be {"texts": [non-empty strings]}'), 400
        if len(texts) > MAX_TEXTS:
            return jsonify(error=f"at most {MAX_TEXTS} texts per request"), 413
        t0 = time.perf_counter()
        X = app.config["VECTORIZER"].transform(texts)  # the fitted vectorizer from featurize.py
        p = app.config["MODEL"].predict_proba(X)[:, 1]  # probability of the positive class
        return jsonify(
            model=NAME,
            version=app.config["VERSION"],
            predictions=[
                {"label": "positive" if q >= 0.5 else "negative", "p_positive": round(float(q), 3)} for q in p
            ],
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    @app.errorhandler(Exception)
    def unhandled(e):
        if isinstance(e, HTTPException):  # 404, 405, 415 ...: Flask's own answer, with its status code
            return jsonify(error=e.description), e.code
        app.logger.exception("unhandled error")  # traceback to the log ...
        return jsonify(error="internal error"), 500  # ... never to the client

    return app
