"""Recompute all Table VIII SLA means from the archived episode records."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results/noms_horizon_check"
SMALL = ARCHIVE / "smaller_fleets_cuda_20261120T075503Z"
LARGE = ARCHIVE / "mobility_cuda_20261120T050646Z"
UPDATE = ARCHIVE / "all_fleets_ippo_qmix_10k_20261121T200939002634Z"
METRICS = ("overall", "critical", "standard", "batch")
KEYS = ("protocol", "horizon_s", "method", "window")


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def main():
    grouped = defaultdict(list)
    for fleet in (4, 8, 16, 32, 64, 128):
        original = LARGE if fleet == 128 else SMALL / f"fleet{fleet}"
        sources = (
            (original / "full/episodes.csv", {"fieldsoar", "critical_local_first"}),
            (UPDATE / f"fleet{fleet}/episodes.csv", {"ippo", "qmix"}),
        )
        for path, policies in sources:
            for row in rows(path):
                if row["method"] in policies:
                    grouped[str(fleet), *(row[k] for k in KEYS)].append(row)
    published = ROOT / "results/published/table_VIII_summary.csv"
    published_rows = list(rows(published))
    if len(grouped) != len(published_rows) or len(published_rows) != 480:
        raise ValueError("Expected 480 unique mission-duration results")
    for row in published_rows:
        key = row["fleet_uavs"], *(row[k] for k in KEYS)
        episodes = grouped.pop(key)
        if len(episodes) != int(row["episodes"]):
            raise ValueError(f"Episode count mismatch: {key}")
        for metric in METRICS:
            scores = [float(ep[f"{metric}_sla"]) for ep in episodes if ep[f"{metric}_sla"] != ""]
            reported = row[f"{metric}_sla_pct"]
            actual = round(100 * mean(scores), 4) if scores else None
            expected = float(reported) if reported else None
            if actual != expected:
                raise ValueError(f"Mission-duration mismatch for {key}, {metric}: {actual}, expected {expected}")
    print("Verified all 480 Table VIII rows from the underlying episode records.")


if __name__ == "__main__":
    main()
