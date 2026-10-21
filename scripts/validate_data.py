#!/usr/bin/env python3
"""Data validation, the check that runs before the metric gate.

    python scripts/validate_data.py          # after `dvc repro`, before scripts/gate.py

A pandera schema on data/prepared/: every row has a label in {0, 1}, a non-empty
document and a known category, and the class balance stays inside the range the
model was built for. Schema drift (a column renamed, ratings in a new scale,
empty documents after a preprocessing bug) fails here, with a message that names
the column, instead of silently training a worse model.
"""

import sys

import pandas as pd
import pandera.pandas as pa

CATEGORIES = {"All_Beauty", "Digital_Music"}
POSITIVE_SHARE = (0.70, 0.90)  # the share seen in months 1 and 2 was 0.81

schema = pa.DataFrameSchema(
    {
        "review_id": pa.Column(str, pa.Check.str_matches(r"^[BM]\d{6}$"), unique=True),
        "category": pa.Column(str, pa.Check.isin(CATEGORIES)),
        "label": pa.Column(int, pa.Check.isin([0, 1])),
        "document": pa.Column(str, pa.Check.str_length(min_value=5)),
    },
    checks=pa.Check(lambda df: POSITIVE_SHARE[0] <= df.label.mean() <= POSITIVE_SHARE[1], name="positive_share"),
    strict=True,
)

ok = True
for split in ("train", "test"):
    df = pd.read_csv(f"data/prepared/{split}.csv", keep_default_na=False)
    try:
        schema.validate(df, lazy=True)
        print(f"{split}: {len(df):,} rows, positive share {df.label.mean():.3f}, schema OK")
    except pa.errors.SchemaErrors as e:
        ok = False
        print(f"{split}: VALIDATION FAILED")
        print(e.failure_cases.head(10).to_string())
sys.exit(0 if ok else "data validation failed")
