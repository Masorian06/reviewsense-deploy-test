#!/usr/bin/env python3
"""Build the ReviewSense datasets from Amazon Reviews 2023 (McAuley-Lab, UCSD).

Month 1  ->  data/reviews.csv                 120,000 reviews from All_Beauty
Month 2  ->  data/incoming/music_reviews.csv   40,000 reviews from Digital_Music
             (append them to data/reviews.csv with src/merge_sources.py)

Usage
    python src/get_data.py                # month 1
    python src/get_data.py --month 2      # month 2, the incoming file
    python src/get_data.py --month 1 2    # both

Source files (one JSON object per line, one file per category):
    default   Hugging Face Hub, dataset McAuley-Lab/Amazon-Reviews-2023,
              raw/review_categories/<Category>.jsonl  (311 MB + 75 MB; uses HF_TOKEN if set)
    fallback  UCSD mirror, <Category>.jsonl.gz          (90 MB + 25 MB; --source ucsd)
Downloads are cached in data/downloads/ (gitignored).

The sample is a seeded reservoir sample over the file's line order, so everyone
who runs this gets the same rows, the same bytes and therefore the same hash.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

HF_REPO = "McAuley-Lab/Amazon-Reviews-2023"
UCSD = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/"
DOWNLOADS = Path("data/downloads")
SEED = 2023
MONTHS = {
    1: {"category": "All_Beauty", "rows": 120_000, "out": Path("data/reviews.csv")},
    2: {"category": "Digital_Music", "rows": 40_000, "out": Path("data/incoming/music_reviews.csv")},
}
PREFIX = {"All_Beauty": "B", "Digital_Music": "M"}
FIELDS = ["review_id", "category", "rating", "title", "text", "verified_purchase",
          "helpful_vote", "date", "parent_asin", "user_id"]


def fetch_hub(category: str) -> Path:
    from huggingface_hub import hf_hub_download  # optional dependency, imported here

    print(f"  downloading raw/review_categories/{category}.jsonl from the Hugging Face Hub")
    path = hf_hub_download(repo_id=HF_REPO, repo_type="dataset",
                           filename=f"raw/review_categories/{category}.jsonl",
                           local_dir=DOWNLOADS)
    return Path(path)


def fetch_ucsd(category: str) -> Path:
    url = f"{UCSD}{category}.jsonl.gz"
    dest = DOWNLOADS / f"{category}.jsonl.gz"
    if dest.exists():
        return dest
    print(f"  downloading {url}")
    part = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
    print()
    part.rename(dest)
    return dest


def fetch(category: str, source: str) -> Path:
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    if source in ("auto", "hub"):
        try:
            return fetch_hub(category)
        except Exception as e:  # no huggingface_hub installed, offline, Hub down ...
            if source == "hub":
                raise
            print(f"  Hub download failed ({type(e).__name__}); falling back to the UCSD mirror")
    return fetch_ucsd(category)


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def reservoir_sample(path: Path, k: int, seed: int) -> list[dict]:
    """k lines chosen uniformly from the file, returned in file order."""
    rng = random.Random(seed)
    kept: list[tuple[int, str]] = []
    n = 0
    with open_text(path) as f:
        for n, line in enumerate(f, 1):
            if n <= k:
                kept.append((n, line))
            else:
                j = rng.randrange(n)
                if j < k:
                    kept[j] = (n, line)
    kept.sort()
    print(f"  {n:,} reviews in the file, {len(kept):,} sampled")
    return [json.loads(line) for _, line in kept]


def to_row(prefix: str, n: int, r: dict, category: str) -> list:
    ts = r.get("timestamp") or 0
    date = datetime.fromtimestamp(ts / 1000, tz=UTC).date().isoformat() if ts else ""
    # one review per line: newlines inside a review become spaces, so wc -l, head and grep behave
    clean = lambda s: " ".join((s or "").replace("\x00", "").split())
    return [f"{prefix}{n:06d}", category, int(float(r.get("rating") or 0)),
            clean(r.get("title")), clean(r.get("text")),
            int(bool(r.get("verified_purchase"))), int(r.get("helpful_vote") or 0),
            date, r.get("parent_asin") or "", r.get("user_id") or ""]


def write_csv(rows: list[list], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(FIELDS)
        w.writerows(rows)
    md5 = hashlib.md5(out.read_bytes()).hexdigest()
    print(f"  wrote {out}: {len(rows):,} rows, {out.stat().st_size / 1e6:,.1f} MB, md5 {md5}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--month", nargs="+", type=int, default=[1], choices=[1, 2])
    ap.add_argument("--source", choices=["auto", "hub", "ucsd"], default="auto",
                    help="auto = Hub, then the UCSD mirror if the Hub fails (default)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true", help="overwrite an existing output file")
    a = ap.parse_args()
    for m in a.month:
        spec = MONTHS[m]
        print(f"Month {m}: {spec['rows']:,} reviews from {spec['category']}")
        if spec["out"].exists() and not a.force:
            print(f"  {spec['out']} already exists; use --force to rebuild it")
            continue
        src = fetch(spec["category"], a.source)
        records = reservoir_sample(src, spec["rows"], a.seed)
        rows = [to_row(PREFIX[spec["category"]], i, r, spec["category"]) for i, r in enumerate(records, 1)]
        write_csv(rows, spec["out"])


if __name__ == "__main__":
    main()
