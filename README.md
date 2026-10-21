# ReviewSense: served and shipped

CMPT-2500 Machine Learning Deployment and Software Development, Unit 6.

ReviewSense classifies Amazon product reviews as positive or negative so a support team
can triage unhappy customers first. In Unit 5 its data, pipeline and experiments went under
version control (DVC) and tracking (MLflow), and a logistic regression on TF-IDF bigrams
became version 3, `@champion`, in the Model Registry. This repository is that end state,
and the starting point for Lab 02: put the champion behind a REST API with Flask, test the
API's contract, and let GitHub Actions test every push, retrain and measure every change to
the pipeline, and promote a model only through an approved button. The lab is `LAB.md`.

## What is in the box

```
reviewsense-deploy/
├── params.yaml              the champion's settings: bigrams, 10,000 features, C = 1
├── dvc.yaml, dvc.lock       the four-stage pipeline and the record of the champion's run
├── eval/metrics.json        the champion's scores (F1 macro 0.9251), the gate's baseline
├── data/reviews.csv.dvc     pointer to the 160,000-review dataset (md5 6bbc136b...)
├── requirements.txt         dvc[s3], scikit-learn, pandas, mlflow, huggingface_hub, pandera, flask, gunicorn
├── requirements-dev.txt     pytest, pytest-cov, ruff: what the CI job runs
├── pyproject.toml           ruff and pytest configuration
├── scripts/
│   ├── bootstrap.sh         rebuilds the Unit 5 end state in five minutes (data, outputs, runs, registry)
│   └── validate_data.py     pandera schema for the prepared splits; runs before the gate
├── src/                     the Unit 5 pipeline: get_data, merge_sources, prepare, featurize, train,
│                            evaluate, register_model, list_models
└── LAB.md                   the lab, step by step
```

What you add in the lab: `app.py`, `tests/test_api.py`, `scripts/gate.py` and three
workflows under `.github/workflows/` (`ci.yml`, `train.yml`, `release.yml`).

## Quick start (WSL 2 Ubuntu 24.04, Linux or macOS)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
export MLFLOW_TRACKING_URI=...        # your DagsHub repository's .mlflow address, see LAB.md § 0
bash scripts/bootstrap.sh             # data, dvc repro (F1 0.9251), four MLflow runs, versions 1-3, @champion on 3
python src/list_models.py
```

## The API contract (what the lab builds)

| Method and path | Body | Answer |
| --- | --- | --- |
| `GET /health` | – | `{"status": "ok", "model": "reviewsense", "version": "3"}` |
| `POST /v1/predict` | `{"texts": ["...", "..."]}` (1 to 100 non-empty strings) | `{"predictions": [{"label": "positive", "p_positive": 0.962}, ...], "model": "reviewsense", "version": "3", "latency_ms": 3.8}` |
| `POST /v1/predict` | anything else | `400 {"error": "body must be {\"texts\": [non-empty strings]}"}` |

The API loads `models:/reviewsense@champion` from the registry at startup. A promotion
moves the alias; the API needs a restart and no code change.

## The pipeline (from Unit 5)

```
data/reviews.csv ──▶ prepare ──▶ data/prepared/ ──▶ featurize ──▶ data/features/ ──▶ train ──▶ models/model.pkl ──▶ evaluate ──▶ eval/metrics.json
                     split, seed                    max_features, ngrams              C, seed
                     └──────────────────────── params.yaml ────────────────────────┘
```

`train.py` logs each run to MLflow when `MLFLOW_TRACKING_URI` is set, tagging it with the
Git commit and the hash of `data/reviews.csv`. In this unit that server is the free MLflow
server DagsHub attaches to your repository, so a GitHub runner can reach it too.

## The data

Amazon Reviews 2023 (Hou et al., 2024): 120,000 All_Beauty reviews (month 1) plus 40,000
Digital_Music reviews (month 2), a seeded sample that everyone rebuilds byte for byte with
`src/get_data.py`. Research and teaching use; cite the paper:
Hou, Y., Li, J., He, Z., Yan, A., Chen, X. and McAuley, J. (2024). *Bridging Language and
Items for Retrieval and Recommendation*, arXiv:2403.03952. https://amazon-reviews-2023.github.io/
