"""The API's contract, tested with a fake model.

These tests check status codes, shapes and messages, not the model's accuracy:
model quality is the metric gate's job (scripts/gate.py). The fake loader is why
create_app takes a parameter: the tests never touch MLflow, the network or a port.

    pytest -q tests/
"""

import numpy as np
import pytest

from app import create_app


class Fake:  # stands in for both the model and the vectorizer
    def transform(self, texts):
        return np.zeros((len(texts), 1))

    def predict_proba(self, X):
        return np.tile([0.1, 0.9], (X.shape[0], 1))


@pytest.fixture
def client():
    app = create_app(loader=lambda: (Fake(), Fake(), "test"))
    return app.test_client()  # no server, no MLflow, no network


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json()["version"] == "test"


def test_one_label_per_text(client):
    r = client.post("/v1/predict", json={"texts": ["good", "bad"]})
    assert r.status_code == 200 and len(r.get_json()["predictions"]) == 2


def test_empty_list_is_400(client):
    assert client.post("/v1/predict", json={"texts": []}).status_code == 400


def test_get_predict_is_405(client):
    assert client.get("/v1/predict").status_code == 405


def test_not_json_is_400(client):
    r = client.post("/v1/predict", data="just a string", content_type="text/plain")
    assert r.status_code == 400 and "texts" in r.get_json()["error"]


def test_too_many_texts_is_413(client):
    assert client.post("/v1/predict", json={"texts": ["x"] * 101}).status_code == 413
