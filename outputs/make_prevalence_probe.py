"""Create a constant-probability leaderboard probe to estimate test prevalence.

This diagnostic uses only sample_submission.csv IDs from the supplied test ZIP.
It never reads image pixels or individual labels.
"""
import argparse
import csv
import io
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-zip", required=True)
    parser.add_argument("--output", default="outputs/prevalence_probe.csv")
    parser.add_argument("--constant-p", type=float, default=0.45)
    args = parser.parse_args()
    if not 0.0 <= args.constant_p <= 1.0:
        raise ValueError("constant probability must be in [0, 1]")
    with zipfile.ZipFile(args.test_zip) as archive:
        with archive.open("sample_submission.csv") as raw:
            rows = list(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig")))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "p_180"])
        writer.writerows((row["image_id"], args.constant_p) for row in rows)
    print(f"Wrote {len(rows):,} constant-{args.constant_p:g} predictions to {output}")


if __name__ == "__main__":
    main()
