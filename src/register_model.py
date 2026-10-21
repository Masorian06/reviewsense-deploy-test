#!/usr/bin/env python3
"""Block 03: put a logged model in the Model Registry and move an alias to it.

    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    python src/register_model.py                                     # register the latest run, no alias
    python src/register_model.py --alias challenger                 # latest run of the experiment
    python src/register_model.py --alias champion --run-id <run id>  # a specific run
    python src/register_model.py --alias champion --version 2        # move the alias, register nothing

Aliases replaced the deprecated stages (Staging / Production / Archived):
@champion is what the API loads, @challenger is the candidate. Promotion is
moving the alias; no file is copied and no deployment code changes.
"""
import argparse
import sys

import mlflow
from mlflow import MlflowClient

NAME = "reviewsense"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alias", help="e.g. champion or challenger (optional)")
    ap.add_argument("--run-id", help="run whose model to register (default: the latest run)")
    ap.add_argument("--version", type=int, help="existing registered version to point the alias at")
    a = ap.parse_args()
    client = MlflowClient()
    version = a.version
    if version is None:
        run_id = a.run_id
        if run_id is None:
            runs = mlflow.search_runs(experiment_names=[NAME], order_by=["start_time DESC"], max_results=1)
            if runs.empty:
                sys.exit(f"no runs in experiment {NAME!r}; train with MLFLOW_TRACKING_URI set first")
            run_id = runs.iloc[0].run_id
        run = client.get_run(run_id)
        logged = client.search_logged_models(experiment_ids=[run.info.experiment_id],
                                             filter_string=f"source_run_id = '{run_id}'")
        if not logged:
            sys.exit(f"run {run_id[:8]} logged no model")
        mv = mlflow.register_model(f"models:/{logged[0].model_id}", NAME)
        for key in ("git_commit", "data_md5"):
            if key in run.data.tags:
                client.set_model_version_tag(NAME, mv.version, key, run.data.tags[key])
        version = int(mv.version)
        print(f"registered run {run_id[:8]} as {NAME} version {version} "
              f"(tags: {', '.join(k for k in ('git_commit', 'data_md5') if k in run.data.tags)})")
    if a.alias:
        client.set_registered_model_alias(NAME, a.alias, version)
        print(f"alias @{a.alias} -> {NAME} version {version}; load it with models:/{NAME}@{a.alias}")


if __name__ == "__main__":
    main()
