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
