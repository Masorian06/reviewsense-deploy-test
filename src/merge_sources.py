#!/usr/bin/env python3
"""Month 2: append the incoming Digital_Music reviews to data/reviews.csv.

    python src/get_data.py --month 2      # writes data/incoming/music_reviews.csv
    python src/merge_sources.py           # appends it to data/reviews.csv

Idempotent: rows whose review_id is already present are skipped, so running it
twice changes nothing. This script is committed to Git because the merge is part
of the data's lineage: the new dataset version has a known origin.
"""
import csv
import sys
from pathlib import Path

SRC = Path("data/incoming/music_reviews.csv")
DST = Path("data/reviews.csv")


def main() -> None:
    if not SRC.exists():
        sys.exit(f"{SRC} not found: run `python src/get_data.py --month 2` first")
    if not DST.exists():
        sys.exit(f"{DST} not found: run `python src/get_data.py` first")
    with open(DST, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        existing = {row[0] for row in reader}
    with open(SRC, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        if next(reader) != header:
            sys.exit("column mismatch between the two files")
        new_rows = [row for row in reader if row[0] not in existing]
    with open(DST, "a", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(new_rows)
    print(f"appended {len(new_rows):,} rows; {DST} now has {len(existing) + len(new_rows):,} rows")


if __name__ == "__main__":
    main()
