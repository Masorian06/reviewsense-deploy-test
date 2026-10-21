#!/usr/bin/env python3
"""Block 03: show the registered model's versions, aliases and lineage tags.

    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    python src/list_models.py
"""
from mlflow import MlflowClient

NAME = "reviewsense"


def main() -> None:
    client = MlflowClient()
    aliases = client.get_registered_model(NAME).aliases          # alias -> version
    by_version: dict[str, list[str]] = {}
    for alias, version in aliases.items():
        by_version.setdefault(str(version), []).append(alias)
    for v in sorted(client.search_model_versions(f"name='{NAME}'"), key=lambda v: int(v.version)):
        names = " ".join("@" + a for a in sorted(by_version.get(str(v.version), []))) or "-"
        print(f"version {v.version}  {names:22s}  run {v.run_id[:8]}  "
              f"commit {v.tags.get('git_commit', '?')}  data {v.tags.get('data_md5', '?')[:6]}")


if __name__ == "__main__":
    main()
