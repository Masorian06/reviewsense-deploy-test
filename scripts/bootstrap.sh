#!/usr/bin/env bash
# Rebuild the state Unit 5 left behind, so Unit 6 can start from it in minutes:
#
#   data/reviews.csv        both months (160,000 rows, md5 6bbc136b...), built from the public source
#   the pipeline outputs    dvc repro at the committed params (bigrams, 10k features): F1 0.9251
#   the MLflow run table    the four runs of the Unit 5 lab (m1-base, m2-base, m2-c01, m2-bigrams)
#   the Model Registry      reviewsense versions 1, 2, 3 with @champion on version 3
#
# Usage (from the repository root, with the environment activated):
#   export MLFLOW_TRACKING_URI=http://127.0.0.1:5000        # a local `mlflow server`, or your DagsHub URI
#   bash scripts/bootstrap.sh
#
# Safe to rerun: downloads are cached, the data step is idempotent, and the registry
# step is skipped when the model already exists. About 5 minutes the first time (the
# raw files are 311 MB + 75 MB); the four training runs take about two minutes.
set -euo pipefail

: "${MLFLOW_TRACKING_URI:?set MLFLOW_TRACKING_URI first (see README.md)}"
export DVC_NO_ANALYTICS=1
TRACKED="params.yaml dvc.lock eval/metrics.json data/reviews.csv.dvc"
EXPECTED_MD5=$(git show HEAD:data/reviews.csv.dvc | python -c "import sys, yaml; print(yaml.safe_load(sys.stdin)['outs'][0]['md5'])")
restore() {  # whatever happens, leave the four tracked files and the data at the committed state
    git checkout -q -- $TRACKED && dvc checkout -q 2>/dev/null || true
}
trap restore EXIT
restore                                        # also heals a workspace left behind by an interrupted run

run_id() {  # run_id <dvc experiment name> -> MLflow run id
    python - "$1" <<'PY'
import sys, mlflow
runs = mlflow.search_runs(experiment_names=["reviewsense"], filter_string=f"tags.dvc_exp = '{sys.argv[1]}'")
sys.exit(f"no run tagged dvc_exp={sys.argv[1]}") if runs.empty else print(runs.iloc[0].run_id)
PY
}

echo "== 1/4  data: month 1 (120k rows), then month 2 appended (160k rows)"
python src/get_data.py --month 1 2
python src/merge_sources.py
actual=$(md5sum data/reviews.csv | cut -d' ' -f1)
if [ "$actual" != "$EXPECTED_MD5" ]; then
    echo "data/reviews.csv has md5 $actual, expected $EXPECTED_MD5 (the committed pointer)"; exit 1
fi
dvc commit -f -q data/reviews.csv.dvc          # same hash: the bytes go into the DVC cache, the pointer is unchanged
echo "   data/reviews.csv matches the committed pointer ($EXPECTED_MD5)"

echo "== 2/4  pipeline: dvc repro at the committed params (expect f1_macro 0.9251)"
dvc repro -q
cat eval/metrics.json

if python src/list_models.py >/dev/null 2>&1; then
    echo "== 3/4  registry: model 'reviewsense' already exists on $MLFLOW_TRACKING_URI, skipping the four runs"
    python src/list_models.py
    exit 0
fi

echo "== 3/4  the four runs of the Unit 5 lab (each 20-30 s; -f so the run cache never skips the training)"
dvc exp remove m1-base m2-base m2-c01 m2-bigrams >/dev/null 2>&1 || true   # names from an earlier, interrupted run
python src/get_data.py --force                 # month 1 only: data version 1 for the first run
dvc commit -f -q data/reviews.csv.dvc          # pointer -> v1 (1c08488b...), train.py tags the run with it
dvc exp run -f -q -n m1-base    -S featurize.max_features=5000  -S featurize.ngrams=1 -S train.C=1.0
python src/merge_sources.py                    # back to version 2
dvc commit -f -q data/reviews.csv.dvc
dvc exp run -f -q -n m2-base    -S featurize.max_features=5000  -S featurize.ngrams=1 -S train.C=1.0
dvc exp run -f -q -n m2-c01     -S featurize.max_features=5000  -S featurize.ngrams=1 -S train.C=0.1
dvc exp run -f -q -n m2-bigrams -S featurize.max_features=10000 -S featurize.ngrams=2 -S train.C=1.0

echo "== 4/4  registry: versions 1-3, then the promotion of version 3"
python src/register_model.py                   --run-id "$(run_id m1-base)"       # version 1
python src/register_model.py --alias champion   --run-id "$(run_id m2-c01)"        # version 2
python src/register_model.py --alias challenger --run-id "$(run_id m2-bigrams)"    # version 3
python src/register_model.py --alias champion   --version 3                        # the promotion
python src/list_models.py
echo "done: version 3 is @champion on $MLFLOW_TRACKING_URI; the workspace is back at the committed state (git status)"
