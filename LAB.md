# Lab 02: ReviewSense, served and shipped

CMPT-2500 · Unit 6 · Model deployment fundamentals and CI/CD pipelines · about 3 hours

Unit 5 left ReviewSense reproducible and useless: version 3 is `@champion` in the registry
and the only way to get a prediction is to run Python on one laptop. By the end of this lab
your copy answers over HTTP, every push is tested, every change to the pipeline is retrained
and measured against the champion by a machine, and a model reaches `@champion` only through
a button that someone has to approve. Every step shows the output you should see, taken
from a real run on 16 October 2026 (Flask 3.1, MLflow 3.17, DVC 3.67, GitHub Actions
`checkout@v7`, `setup-python@v6`).

Hand in (see the checklist at the end): the URL of your repository, the URLs of two pull
requests (one the gate accepted, one it refused) and a screenshot of the Promote run.

## 0. Set up (25 min)

You need Git, the GitHub CLI (`gh auth login` once), Python 3.11 or newer, and a DagsHub
account: go to https://dagshub.com and **Sign in with GitHub**. DagsHub gives every
repository a free DVC remote and a free MLflow server; that is what the GitHub runner will
pull data from and register models to. On Windows use WSL 2 (Ubuntu 24.04) and keep the
repository under `~/`, not `/mnt/c/...`.

### 0.1 Your repository

```bash
gh repo create reviewsense-deploy --template Masorian06/reviewsense-deploy --public --clone
cd reviewsense-deploy
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

Public, because the approval step in Part E (a required reviewer on an environment) is
only available on public repositories unless the account has an Enterprise plan. If you
prefer `--private`, everything works except that step: the Promote workflow then runs
without pausing for approval.

The template is Unit 5's end state: the four-stage pipeline, `params.yaml` at the champion's
settings (bigrams, 10,000 features, C = 1), the committed `eval/metrics.json` (F1 0.9251)
and the pointer to the 160,000-review dataset. Nothing from Unit 6 is in it yet.

### 0.2 Your DagsHub repository and token

1. On dagshub.com click **Create +** → **New Repository**, name it `reviewsense-deploy`,
   leave it private, **Create**. (Or **Import Repository** → **Connect** GitHub and pick
   your new repository: the result is the same, plus a mirror of your code.)
2. **Settings → Tokens** (https://dagshub.com/user/settings/tokens) → **Generate new token**.
   Copy it: it is shown once.

The repository gives you two addresses. Put them, your user name and the token in a file
outside the repository and load it in every terminal you open for this lab:

```bash
cat > ~/.config/reviewsense-deploy.env <<'EOF'
export DAGSHUB_USER=<your-dagshub-user>
export DAGSHUB_TOKEN=<your-token>
export MLFLOW_TRACKING_URI=https://dagshub.com/$DAGSHUB_USER/reviewsense-deploy.mlflow
export MLFLOW_TRACKING_USERNAME=$DAGSHUB_USER
export MLFLOW_TRACKING_PASSWORD=$DAGSHUB_TOKEN
EOF
chmod 600 ~/.config/reviewsense-deploy.env
source ~/.config/reviewsense-deploy.env
```

`train.py` logs to MLflow whenever `MLFLOW_TRACKING_URI` is set (Unit 5, Part D); the
user name and password are how the MLflow client signs in to DagsHub. The token is also
the key pair of the DVC remote, which is S3-compatible:

```bash
dvc remote add -d origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/$DAGSHUB_USER/reviewsense-deploy.s3
dvc remote modify --local origin access_key_id $DAGSHUB_TOKEN
dvc remote modify --local origin secret_access_key $DAGSHUB_TOKEN
cat .dvc/config
git commit -am "DVC remote: DagsHub storage"
```

```
[core]
    analytics = false
    remote = origin
['remote "origin"']
    url = s3://dvc
    endpointurl = https://dagshub.com/<you>/reviewsense-deploy.s3
```

The two `--local` lines went to `.dvc/config.local`, which DVC's own `.gitignore` keeps out
of Git. Credentials never enter a repository.

### 0.3 Rebuild the Unit 5 end state (5 min)

```bash
bash scripts/bootstrap.sh
```

The script downloads the raw reviews (311 MB + 75 MB, once, cached in `data/downloads/`),
rebuilds `data/reviews.csv` and checks that its hash matches the committed pointer, runs
`dvc repro` at the champion's settings, then reruns the four experiments of the Unit 5 lab
with MLflow tracking on and registers them: versions 1, 2, 3 and the promotion of version 3.
Expected (about five minutes the first time; the four training runs are two of them):

```
== 1/4  data: month 1 (120k rows), then month 2 appended (160k rows)
   data/reviews.csv matches the committed pointer (6bbc136bf2dc2c692fc27289a2d5864c)
== 2/4  pipeline: dvc repro at the committed params (expect f1_macro 0.9251)
  "f1_macro": 0.9251,
== 3/4  the four runs of the Unit 5 lab (each 20-30 s; -f so the run cache never skips the training)
== 4/4  registry: versions 1-3, then the promotion of version 3
registered run 09e88658 as reviewsense version 1 (tags: git_commit, data_md5)
registered run 8f26de1a as reviewsense version 2 (tags: git_commit, data_md5)
alias @champion -> reviewsense version 2; load it with models:/reviewsense@champion
registered run 6e481a51 as reviewsense version 3 (tags: git_commit, data_md5)
alias @challenger -> reviewsense version 3; load it with models:/reviewsense@challenger
alias @champion -> reviewsense version 3; load it with models:/reviewsense@champion
version 1  -                       run 09e88658  commit 5de31ad  data 1c0848
version 2  -                       run 8f26de1a  commit 5de31ad  data 6bbc13
version 3  @challenger @champion   run 6e481a51  commit 5de31ad  data 6bbc13
done: version 3 is @champion on https://dagshub.com/<you>/reviewsense-deploy.mlflow; ...
```

Run ids and the commit differ; the hash, the F1 values and the three versions do not.
Open your DagsHub repository → **Experiments**: the four runs are there, with their
`git_commit` and `data_md5` tags. Finally push the bytes and the pointers:

```bash
dvc push            # 9 files: the dataset, the splits, the features, the model
git push
```

Check yourself: `python src/list_models.py` prints the three versions with `@champion`
on 3, and `git status` is clean.

## Part A · The API (45 min)

### Step 1 · `app.py`: load the champion once, answer many times

Create `app.py` at the repository root:

```python
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
```

Read it top to bottom; it is about forty lines of logic.

- **`load_model`** asks the registry for whatever `@champion` points at. Nothing in this
  file says "version 3". It downloads `vectorizer.pkl` from the *same run*, because a
  model is only meaningful with the vectorizer that produced its features (case problem 2:
  Omar's Monday crash was a vocabulary that no longer matched the model). It returns the
  version number so every answer can carry it.
- **`create_app` is an application factory.** It builds the app and returns it, which is
  what Flask recommends and what makes testing possible: the loader is a parameter, so a
  test can pass a fake instead of touching MLflow. Loading happens *once*, here, into
  `app.config`; a request only reads. If loading fails, the process never starts, and that
  is the right failure: whoever supervises the process sees it immediately.
- **`/health`** is what a load balancer, a container platform or you with `curl` asks to
  know that the service is up and *which model* it holds. Returning a dict is enough; Flask
  turns it into JSON.
- **`/v1/predict`** validates before it touches the model: not JSON, not a dict, no
  `texts`, an empty list, a non-string, an empty string, more than 100 texts. Each gets a
  4xx and a sentence. Then two lines are the model: transform, `predict_proba`. The `v1` in
  the path is the contract's version; if the response shape ever changes, that becomes `v2`
  and the old path keeps working for old clients.
- **The error handler** is the last line of defence. Flask's own HTTP errors (404 for a
  wrong path, 405 for a GET on `/v1/predict`) pass through with their code and a JSON
  body. Anything else is a bug: the traceback goes to the log, the client gets a plain
  500. Tracebacks are for you, not for the internet.

### Step 2 · Run it, talk to it

Terminal 1, the development server (the `--debug` flag reloads on edits and shows an
interactive debugger on errors; it is for this laptop only, never for a server):

```bash
source ~/.config/reviewsense-deploy.env
flask --app app run --debug --port 8000
```

Terminal 2:

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/v1/predict -H 'Content-Type: application/json' \
     -d '{"texts":["Great product, exactly as described","Broke after two days, waste of money"]}'
curl -s -w ' [%{http_code}]\n' -X POST localhost:8000/v1/predict -H 'Content-Type: application/json' -d '{"texts":[]}'
curl -s -w ' [%{http_code}]\n' -X POST localhost:8000/v1/predict -d 'hello'
curl -s -w ' [%{http_code}]\n' localhost:8000/v1/predict
```

```
{"model":"reviewsense","status":"ok","version":"3"}
{"latency_ms":10.5,"model":"reviewsense","predictions":[{"label":"positive","p_positive":1.0},{"label":"negative","p_positive":0.005}],"version":"3"}
{"error":"body must be {\"texts\": [non-empty strings]}"} [400]
{"error":"body must be {\"texts\": [non-empty strings]}"} [400]
{"error":"The method is not allowed for the requested URL."} [405]
```

Two reviews in, two labels out, in about ten milliseconds: the model and the vectorizer
were loaded at startup, so a request does no I/O. A bad body gets a 400 and a sentence,
never a traceback. Now the experiment that explains the rest of the lab. In terminal 2:

```bash
python src/register_model.py --alias champion --version 2
curl -s localhost:8000/health
python src/register_model.py --alias champion --version 3
```

```
alias @champion -> reviewsense version 2; load it with models:/reviewsense@champion
{"model":"reviewsense","status":"ok","version":"3"}
alias @champion -> reviewsense version 3; load it with models:/reviewsense@champion
```

The registry says 2, the running API still says 3. The model loads at startup, so a
promotion needs a restart. Part E makes that restart the last step of a promotion.

### Step 3 · The same app on a production server

Stop the dev server (Ctrl-C) and start gunicorn, the server you would run on a Linux host
(on Windows outside WSL: `waitress-serve --port 8000 --call app:create_app`):

```bash
gunicorn -w 4 -b 127.0.0.1:8000 "app:create_app()"
```

Four workers, four copies of the app, each holding the champion: four requests at a time,
and if one worker dies the master starts another while three keep answering. Edit a file
and nothing happens: you redeploy on purpose, when the tests pass. `curl localhost:8000/health`
answers as before. Stop it with Ctrl-C and commit:

```bash
git add app.py && git commit -m "Flask API: /health and /v1/predict, the champion loaded once at startup"
```

## Part B · Tests (20 min)

### Step 4 · `tests/test_api.py`: the contract, with a fake model

```python
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
```

```bash
pytest -q --cov=app tests/
ruff check .
```

```
......                                                                   [100%]
Name     Stmts   Miss  Cover
----------------------------
app.py      43      6    86%
----------------------------
TOTAL       43      6    86%
6 passed in 1.40s
All checks passed!
```

- `test_client()` calls the app directly: no port, no process, a second for the whole file.
- The fixture builds the app with a **fake loader**, a fake model and a fake vectorizer.
  That is the reason `create_app` takes a parameter. The tests run anywhere, including on
  a GitHub runner with no MLflow server and no 50 MB download.
- They test the **contract**: status codes, one label per text, the message on a bad body,
  405 on a GET. They say nothing about accuracy. Model quality is checked by the metric
  gate in Part D, on real data, where the registry is reachable. Unit tests should fail
  only when *your code* is wrong.
- The six lines `load_model` is not covered (86 %): it is the part that needs a registry.
  A smoke test against the real champion belongs in the training workflow, not here.

```bash
git add tests/test_api.py && git commit -m "API contract tests with a fake model"
```

## Part C · CI: every push, every pull request (25 min)

### Step 5 · `.github/workflows/ci.yml`

```yaml
# .github/workflows/ci.yml — lint and test on every pull request and every push to main
name: CI
on:
  push: { branches: [main] }
  pull_request:
permissions:
  contents: read                      # least privilege: this token cannot write, comment or publish
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix: { python: ["3.11", "3.12"] }
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: ${{ matrix.python }}
          cache: pip                  # keyed on requirements*.txt: a 90-second install becomes ten
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: ruff check .             # seconds; fail fast
      - run: pytest -q --cov=app tests/
```

```bash
mkdir -p .github/workflows
git add .github/workflows/ci.yml && git commit -m "CI workflow: ruff and pytest on Python 3.11 and 3.12"
git push
```

Open your repository on GitHub → **Actions**. The push started a run named `CI` with two
jobs, `test (3.11)` and `test (3.12)`, green in about a minute. Click one and read the
steps top to bottom; it is the same sequence you ran in Step 4, on a fresh Ubuntu machine.

- **`on`**: every pull request, and every push to `main`. Nothing merges untested.
- **`permissions: contents: read`**: the job's token can read the repository and nothing
  else. A workflow that only tests has no business writing.
- **The matrix** runs the same steps on two Python versions in parallel: the version on
  your laptop and the one on the server are rarely the same.
- **`cache: pip`** keeps the installed wheels between runs, keyed on `requirements*.txt`.
- **Order**: cheap checks first. Lint fails in seconds; the tests take a minute.

This is where case problem 2 ends, if a test covers what broke. Omar's preprocessing
change would have reached a pull request, the tests would have run, and the red X would
have been on the screen before Monday.

## Part D · Train, validate, gate (45 min)

### Step 6 · Three secrets

The training workflow needs to reach DagsHub: the DVC remote for the data and the MLflow
server for logging and registering. It must never find the token in the repository, so the
token goes into GitHub's encrypted secrets (Settings → Secrets and variables → Actions,
or from your terminal):

```bash
source ~/.config/reviewsense-deploy.env
gh secret set MLFLOW_TRACKING_URI --body "$MLFLOW_TRACKING_URI"
gh secret set DAGSHUB_USER        --body "$DAGSHUB_USER"
gh secret set DAGSHUB_TOKEN       --body "$DAGSHUB_TOKEN"
gh secret list
```

### Step 7 · `scripts/gate.py`: better than the champion, same test set

```python
#!/usr/bin/env python3
"""The metric gate: fail the job unless the model this run just trained beats the baseline.

    python scripts/gate.py --min-gain 0.002            # after `dvc repro`

new       eval/metrics.json written by this run
baseline  eval/metrics.json committed on the ref the change is measured against:
          origin/main for a pull request, the previous main commit after a merge,
          the current commit for a scheduled retrain (see baseline_ref)

Both runs score the same fixed test split (prepare.seed), so the numbers compare.
Two thresholds: the headline metric must gain at least --min-gain, and the metric
the support team feels (recall on negative reviews) must not drop by more than
--max-recall-drop. Exit 1 fails the job, which fails the status check, which
blocks the merge. If the run produced the very same model as the baseline
(identical models/model.pkl in dvc.lock), there is nothing to decide: exit 0
and tell the workflow not to register it (model_changed=false).
"""

import argparse
import json
import os
import subprocess
import sys

import yaml


def baseline_ref() -> str:
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    if event == "push":  # the merge just landed on main: compare with main as it was before it
        return "HEAD~1"
    if event in ("schedule", "workflow_dispatch"):  # retrain of the current code: compare with its own record
        return "HEAD"
    return "origin/main"  # a pull request (or a laptop): compare with the champion's branch


def git_show(ref: str, path: str) -> str:
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], text=True)


def model_md5(lock_text: str) -> str:
    return yaml.safe_load(lock_text)["stages"]["train"]["outs"][0]["md5"]


def output(key: str, value: str) -> None:  # a step output for the workflow, when running in Actions
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"{key}={value}\n")


ap = argparse.ArgumentParser()
ap.add_argument("--min-gain", type=float, default=0.0, help="required f1_macro improvement")
ap.add_argument("--max-recall-drop", type=float, default=0.01, help="tolerated fall of recall_negative")
ap.add_argument("--baseline", default=baseline_ref(), help="git ref whose eval/metrics.json is the baseline")
a = ap.parse_args()
if a.baseline == "HEAD":  # a retrain of the champion's own commit: it must not regress, it need not improve
    a.min_gain = min(a.min_gain, 0.0)

if model_md5(open("dvc.lock").read()) == model_md5(git_show(a.baseline, "dvc.lock")):
    print(f"no model change: models/model.pkl is identical to {a.baseline}'s, nothing to gate or register")
    output("model_changed", "false")
    sys.exit(0)
output("model_changed", "true")

new = json.load(open("eval/metrics.json"))  # this run
base = json.loads(git_show(a.baseline, "eval/metrics.json"))  # the baseline's committed record
gain = new["f1_macro"] - base["f1_macro"]
drop = base["recall_negative"] - new["recall_negative"]
print(f"baseline {a.baseline} (min gain {a.min_gain}, max recall drop {a.max_recall_drop})")
print(f"f1_macro {base['f1_macro']:.4f} -> {new['f1_macro']:.4f} ({gain:+.4f})")
print(f"recall_negative {base['recall_negative']:.4f} -> {new['recall_negative']:.4f} (drop {drop:+.4f})")
if gain < a.min_gain:
    sys.exit(f"GATE FAILED: gain {gain:+.4f} < {a.min_gain}")
if drop > a.max_recall_drop:
    sys.exit(f"GATE FAILED: recall_negative fell {drop:.4f} > {a.max_recall_drop}")
print("GATE PASSED")
```

- **The baseline is `main`.** On `main`, `eval/metrics.json` and `dvc.lock` describe the
  run the champion came from, so "better than main" means "better than the champion". Both
  numbers come from the same fixed test split (`prepare.seed`), so they are comparable.
  After a merge the workflow runs on `main` itself, and the baseline is then `main` as it
  was before the merge (`HEAD~1`); a scheduled retrain compares with its own record.
- **Two thresholds.** The headline metric must gain at least `--min-gain`: 0.002 here,
  because a model that is 0.0005 better is noise, not worth a deployment. And the metric
  the business feels, recall on negative reviews, must not drop. Lena's month-3 model (case
  problem 3) had +0.004 F1 and −0.12 recall: the second threshold would have stopped it.
- **Exit 1** fails the job, which fails the status check, which blocks the merge. Both
  numbers are printed so the log explains the decision.
- **No change, no decision.** If the run produced the very same model (`models/model.pkl`
  has the same hash as the baseline's `dvc.lock` says), the gate passes with
  `model_changed=false`, and the workflow will not register a duplicate version. A comment
  edit in `src/` retrains and produces the identical model; it must not fail, and it must
  not become version 5.

Try it on your laptop against your own `main` (nothing changed, so it says so):

```bash
python scripts/gate.py --min-gain 0.002
```

```
no model change: models/model.pkl is identical to origin/main's, nothing to gate or register
```

### Step 8 · `.github/workflows/train.yml`

```yaml
# .github/workflows/train.yml — retrain and validate when the pipeline's inputs change, and every Monday
name: Train
on:
  pull_request: { paths: ["src/**", "params.yaml", "dvc.yaml", "data/*.dvc"] }   # not on a README edit
  push:         { branches: [main], paths: ["src/**", "params.yaml", "dvc.yaml", "data/*.dvc"] }
  schedule:     [{ cron: "0 6 * * 1" }]                                          # Mondays 06:00 UTC: continuous training
  workflow_dispatch:                                                             # and a button
permissions:
  contents: read
jobs:
  train:
    runs-on: ubuntu-latest
    env:
      # Settings → Secrets and variables → Actions. The DVC remote and the MLflow server are the
      # DagsHub repository linked to this one; one token opens both (see LAB.md, step 1).
      MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}     # https://dagshub.com/<user>/<repo>.mlflow
      MLFLOW_TRACKING_USERNAME: ${{ secrets.DAGSHUB_USER }}       # your DagsHub user name
      MLFLOW_TRACKING_PASSWORD: ${{ secrets.DAGSHUB_TOKEN }}      # the same token as below
      AWS_ACCESS_KEY_ID: ${{ secrets.DAGSHUB_TOKEN }}             # DagsHub storage is S3-compatible:
      AWS_SECRET_ACCESS_KEY: ${{ secrets.DAGSHUB_TOKEN }}         # the same token is both keys
    steps:
      - uses: actions/checkout@v7
        with: { fetch-depth: 0 }                                   # the gate compares against main
      - uses: actions/setup-python@v6
        with: { python-version: "3.12", cache: pip }
      - run: pip install -r requirements.txt
      - run: dvc pull data/reviews.csv                             # Unit 5, step 4 — now a machine does it
      - run: dvc repro                                             # prepare → featurize → train → evaluate, on the runner
      - run: python scripts/validate_data.py                       # schema and class balance of data/prepared
      - run: python scripts/gate.py --min-gain 0.002               # exit 1 unless better than the baseline
        id: gate
      - run: python src/register_model.py --alias challenger && dvc push   # registry version N, outputs to the remote
        if: github.ref == 'refs/heads/main' && steps.gate.outputs.model_changed == 'true'   # PRs validate; main registers
```

- **Triggers.** Pull requests and pushes that touch the pipeline's inputs, a `paths` filter
  so that a README edit does not retrain; a cron on Monday mornings, which is continuous
  training; and a manual button. (A pull request that touches none of those paths does not
  start this workflow at all; see Step 10.)
- **Secrets become environment variables.** `train.py` and `register_model.py` read
  `MLFLOW_TRACKING_URI` exactly as on your laptop; DVC reads the two AWS variables, which
  DagsHub accepts as its S3-compatible keys. Nothing in the file names a host or a token.
- **`fetch-depth: 0`**: the whole history, because the gate reads `origin/main`'s files.
- **`dvc pull data/reviews.csv`** pulls the dataset and nothing else, so `dvc repro` has
  to run every stage: the runner trains the model itself, it does not reuse yours. That is
  the point: the number in the log was produced by a machine nobody edited by hand.
- **Validate, then gate.** `scripts/validate_data.py` (in the template) checks the prepared
  splits with a pandera schema: labels in {0, 1}, no empty document, known categories,
  class balance in range. Schema drift fails there with the column's name; a worse model
  fails at the gate.
- **Register only on `main`, only when the model changed.** A pull request validates;
  it never touches the registry. After the merge, the same workflow runs on `main`,
  registers the new version as `@challenger` and pushes the outputs to the remote.

```bash
git add scripts/gate.py .github/workflows/train.yml
git commit -m "Train workflow: dvc pull, dvc repro, data validation, the metric gate, register on main"
git push
```

This push does not start `Train`: it changed no file in the `paths` list. The next step does.

### Step 9 · Pull request 1: the winning Unit 5 experiment

Propose the trigram model as a change, the way a teammate would:

```bash
git switch -c trigrams
sed -i 's/ngrams: 2/ngrams: 3/; s/C: 1.0/C: 2/' params.yaml     # or edit params.yaml by hand
dvc repro
cat eval/metrics.json
dvc push
git commit -am "Trigrams, C=2: the winning Unit 5 experiment (F1 0.9298)"
git push -u origin trigrams
gh pr create --fill
```

```
evaluate: {"f1_macro": 0.9298, "accuracy": 0.9586, "precision_negative": 0.9201, "recall_negative": 0.852, "n_test": 29699}
https://github.com/<you>/reviewsense-deploy/pull/1
```

The commit carries `params.yaml`, `dvc.lock` and `eval/metrics.json`: the pull request
*proposes a model*, with its numbers. Open the PR on GitHub. Two workflows started: `CI`
(two jobs, green in a minute) and `Train`. Open the `Train` run and read the gate's step:

```
Run python scripts/gate.py --min-gain 0.002
baseline origin/main (min gain 0.002, max recall drop 0.01)
f1_macro 0.9251 -> 0.9299 (+0.0048)
recall_negative 0.8336 -> 0.8522 (drop -0.0186)
GATE PASSED
```

The runner trained the model itself and got 0.9299 where your laptop got 0.9298: logistic
regression on a different CPU and BLAS library can differ in the fourth decimal, which is
exactly why the gate compares against a minimum gain and not against equality. Scroll up:
`dvc pull` fetched one file in a few seconds, `dvc repro` ran all four stages, `train.py`
logged a run to your DagsHub MLflow (open **Experiments**: it is there, tagged with the
PR's commit). Nothing was registered: this is a pull request.

### Step 10 · The branch rule: no green, no merge

The gate's verdict only matters if a red X blocks the merge button. Settings → **Rules** →
**Rulesets** → **New ruleset** → **New branch ruleset**:

- Name: `main: tests and the gate before a merge`. Enforcement: **Active**.
  Target branches: **Add target** → **Include default branch**.
- Bypass list: **Add bypass** → **Repository admin** (that is you; see below for why).
- Rules: tick **Restrict deletions**, **Block force pushes**, **Require a pull request
  before merging** (0 required approvals: you cannot approve your own PR), and
  **Require status checks to pass**. Under it, **Add checks** and type the three names
  exactly as the Actions tab shows them: `test (3.11)`, `test (3.12)`, `train`.
- **Create**.

Why the bypass: a pull request that touches none of the paths in `train.yml` never starts
`Train`, so its `train` check stays "Expected" forever and the merge button stays grey.
As the repository's admin you can then merge with "Merge without waiting for requirements".
For the pull requests of this lab the check always runs.

From the terminal, the same ruleset in one command (optional, instead of the clicks):

```bash
gh api -X POST repos/<you>/reviewsense-deploy/rulesets --input - <<'EOF'
{"name":"main: tests and the gate before a merge","target":"branch","enforcement":"active",
 "conditions":{"ref_name":{"include":["~DEFAULT_BRANCH"],"exclude":[]}},
 "bypass_actors":[{"actor_id":5,"actor_type":"RepositoryRole","bypass_mode":"always"}],
 "rules":[{"type":"deletion"},{"type":"non_fast_forward"},
          {"type":"pull_request","parameters":{"required_approving_review_count":0,"dismiss_stale_reviews_on_push":false,"require_code_owner_review":false,"require_last_push_approval":false,"required_review_thread_resolution":false}},
          {"type":"required_status_checks","parameters":{"strict_required_status_checks_policy":false,"required_status_checks":[{"context":"test (3.11)"},{"context":"test (3.12)"},{"context":"train"}]}}]}
EOF
```

### Step 11 · Merge, and watch `main` register the model

Back on the pull request: three green checks, the merge button is enabled. **Merge pull
request** → **Confirm merge** (or `gh pr merge 1 --merge`). The merge is a push to `main`
that changed `params.yaml`, so `Train` runs again, this time on `main`:

```
baseline HEAD~1 (min gain 0.002, max recall drop 0.01)
f1_macro 0.9251 -> 0.9299 (+0.0048)
recall_negative 0.8336 -> 0.8522 (drop -0.0186)
GATE PASSED
Run python src/register_model.py --alias challenger && dvc push
registered run 1fce9f08 as reviewsense version 4 (tags: git_commit, data_md5)
alias @challenger -> reviewsense version 4; load it with models:/reviewsense@challenger
5 files pushed
```

```bash
git switch main && git pull
python src/list_models.py
```

```
version 1  -                       run 09e88658  commit 5de31ad  data 1c0848
version 2  -                       run 8f26de1a  commit 5de31ad  data 6bbc13
version 3  @champion               run 6e481a51  commit 5de31ad  data 6bbc13
version 4  @challenger             run 1fce9f08  commit 394e536  data 6bbc13
```

Version 4 exists, tagged with the merge commit; `@champion` has not moved. Promotion is
a separate decision (Part E). Restart your API now and `/health` still says 3.

### Step 12 · Pull request 2: the gate says no

A change that looks reasonable and is not. Prepare it from `main`:

```bash
git switch -c c01
sed -i 's/max_features: 10000/max_features: 5000/; s/ngrams: 3/ngrams: 1/; s/C: 2/C: 0.1/' params.yaml
dvc repro && cat eval/metrics.json
dvc push
git commit -am "C=0.1, unigrams, 5k features: a weaker model (F1 0.8814)"
git push -u origin c01
gh pr create --fill
```

`CI` is green: the code is fine. `Train` is red:

```
baseline origin/main (min gain 0.002, max recall drop 0.01)
f1_macro 0.9298 -> 0.8814 (-0.0484)
recall_negative 0.8520 -> 0.7097 (drop +0.1423)
GATE FAILED: gain -0.0484 < 0.002
Error: Process completed with exit code 1.
```

The baseline is now the trigram model, because `main` moved: the gate always measures against
the champion's current record (0.9298 is the value your laptop committed in PR 1; the runner
may print 0.8811 or 0.8814 for the new model). On the pull request page the `train` check has
a red X and the merge button is disabled:
"Required statuses must pass before merging". Nothing reached the registry. Leave this
pull request open; it is part of what you hand in. Case problem 3, answered: a model
reaches the registry only with a passing gate, and the next step adds a named approver.

## Part E · Promote, and roll back (25 min)

### Step 13 · `.github/workflows/release.yml` and the `production` environment

```yaml
# .github/workflows/release.yml — promote a registry version to @champion (and roll back the same way)
name: Promote
on:
  workflow_dispatch:
    inputs:
      version: { description: "registry version → @champion", required: true }
permissions:
  contents: read
jobs:
  promote:
    runs-on: ubuntu-latest
    environment: production             # Settings → Environments: a required reviewer approves, the secrets live here
    env:
      MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
      MLFLOW_TRACKING_USERNAME: ${{ secrets.DAGSHUB_USER }}
      MLFLOW_TRACKING_PASSWORD: ${{ secrets.DAGSHUB_TOKEN }}
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with: { python-version: "3.12", cache: pip }
      - run: pip install mlflow
      - run: python src/register_model.py --alias champion --version ${{ inputs.version }}
      # Then the API has to notice. Today: `systemctl --user restart reviewsense` on the serving host,
      # so the gunicorn workers reload @champion. Unit 7: a rolling restart of containers.
```

```bash
git switch main
git add .github/workflows/release.yml
git commit -m "Promote workflow: move @champion by button, behind the production environment"
git push
```

Then Settings → **Environments** → **New environment** → `production` → **Configure
environment** → tick **Required reviewers**, add yourself, **Save protection rules**.
(On a private repository GitHub refuses this rule unless the plan is Enterprise; the
environment still exists and the workflow runs without the pause.)

- **`workflow_dispatch` with one input**: a button on the Actions tab that asks for the
  registry version. Continuous delivery is a button someone presses, with a log of who
  pressed it and when.
- **`environment: production`**: the run pauses until a required reviewer approves, and
  only then receives the environment's secrets. Who approved, when and which version are
  on the run page forever.
- **The step is the Unit 5 script.** Promotion moves a pointer; nothing is copied or
  retrained. The API never changes, because it loads `@champion`; it only needs a restart.

### Step 14 · Press the button

Actions → **Promote** → **Run workflow** → version `4` → **Run workflow**. The run appears
with a yellow "Waiting": **Review deployments** → tick `production` → **Approve and deploy**.

```
Run python src/register_model.py --alias champion --version 4
alias @champion -> reviewsense version 4; load it with models:/reviewsense@champion
```

```bash
python src/list_models.py
gunicorn -w 4 -b 127.0.0.1:8000 "app:create_app()" &
curl -s localhost:8000/health
```

```
version 3  -                       ...
version 4  @challenger @champion   ...
{"model":"reviewsense","status":"ok","version":"4"}
```

The API changed with no code change: it loaded whatever `@champion` pointed at when it
started. Rollback is the same button with the previous number: run **Promote** with `3`,
approve, restart. Seconds, not a rebuild. Do it once, then promote 4 again, so your
registry ends with version 4 as champion. Stop gunicorn (`kill %1`).

## Hand in

Add a section `## Lab 02 results` to your `README.md` with: the URL of the pull request the
gate accepted and of the one it refused, the F1 values the runner printed for both, the
version your registry has as `@champion`, and the path of a screenshot of the approved
Promote run (commit it under `docs/`). Push and submit the repository URL. The instructor
checks:

- [ ] `app.py` with `/health` and `/v1/predict`, loading by alias, and a passing `pytest -q tests/`
- [ ] `CI` green on `main` on both Python versions; `ruff check .` clean
- [ ] three secrets set; `Train` ran on both pull requests; one `GATE PASSED`, one `GATE FAILED`
- [ ] the ruleset: the refused pull request shows a disabled merge button
- [ ] the registry has version 4 tagged with the merge commit, and `@champion` on it
- [ ] a `Promote` run approved by you (the screenshot), and the `## Lab 02 results` section

## Troubleshooting

- `ModuleNotFoundError: No module named 'app'` from pytest in CI but not on your laptop: `pyproject.toml` in the template has `pythonpath = ["."]`; keep it, or run `python -m pytest`.
- `API request ... failed with error code 404` or `401` from DagsHub MLflow: the client is not signed in. Set `MLFLOW_TRACKING_USERNAME` to your DagsHub *user name* and `MLFLOW_TRACKING_PASSWORD` to the token (the token as user name does not work).
- `dvc pull` says `Forbidden` or `403`: the token in `.dvc/config.local` (laptop) or in the `DAGSHUB_TOKEN` secret (runner) is wrong, or the DagsHub repository name in `endpointurl` is not yours.
- `train` check stays "Expected" on a pull request: the PR touched none of `src/**`, `params.yaml`, `dvc.yaml`, `data/*.dvc`, so `Train` did not start. Merge with the admin bypass, or add the change that needs retraining.
- `GATE FAILED: gain +0.0000` on a push that only touched a script: that script is a dependency of a stage, so `dvc repro` retrained and got the same model. `scripts/gate.py` already treats an identical `models/model.pkl` as "no change"; if your copy predates that, update it.
- "Failed to create the environment protection rule ... billing plan": required reviewers need a public repository (or an Enterprise plan). Make the repository public (Settings → General → Danger Zone → Change visibility) or do Part E without the pause.
- `Address already in use` on port 8000: another server is running; `pkill -f gunicorn` / stop the dev server.
- `flask --app app run` exits at once with an MLflow error: the shell has no `MLFLOW_TRACKING_URI`, or the registry has no `@champion` yet (run `scripts/bootstrap.sh`).
- The dev server prints `WARNING: This is a development server`: correct, and the reason Step 3 exists.
