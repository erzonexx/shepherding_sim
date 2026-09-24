"""Create paired controller summaries from two experiment-runner CSV files."""

import argparse
from datetime import datetime
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


PAIR_KEYS = ["Case", "Seed"]
METRICS = [
    "Frames",
    "FinalDist",
    "MinDist",
    "MaxDisp",
    "DangerFrames",
    "PeakInsideGoal",
    "FinalInsideGoal",
    "GoalExits",
]


def _as_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False})


def exact_signed_mean_pvalue(differences):
    values = np.asarray(differences, dtype=float)
    if len(values) == 0 or np.allclose(values, 0.0):
        return 1.0
    if len(values) > 20:
        return None
    observed = abs(float(np.mean(values)))
    extreme = 0
    total = 2 ** len(values)
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted = abs(float(np.mean(values * np.asarray(signs))))
        extreme += permuted >= observed - 1e-12
    return extreme / total


def compare_frames(baseline, candidate):
    for label, frame in (("baseline", baseline), ("candidate", candidate)):
        missing = set(PAIR_KEYS + ["Success"] + METRICS) - set(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing columns: {sorted(missing)}")
        if frame.duplicated(PAIR_KEYS).any():
            raise ValueError(f"{label} contains duplicate Case/Seed rows")

    paired = baseline[PAIR_KEYS + ["Success"] + METRICS].merge(
        candidate[PAIR_KEYS + ["Success"] + METRICS],
        on=PAIR_KEYS,
        how="inner",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    if len(paired) != len(baseline) or len(paired) != len(candidate):
        raise ValueError("baseline and candidate Case/Seed sets do not match")

    paired["Success_baseline"] = _as_bool(paired["Success_baseline"])
    paired["Success_candidate"] = _as_bool(paired["Success_candidate"])
    if paired[["Success_baseline", "Success_candidate"]].isna().any().any():
        raise ValueError("Success columns must contain true/false values")

    for metric in METRICS:
        paired[f"Delta_{metric}"] = (
            paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"]
        )

    improved = (~paired["Success_baseline"]) & paired["Success_candidate"]
    regressed = paired["Success_baseline"] & (~paired["Success_candidate"])
    summary = {
        "pairs": len(paired),
        "baseline_successes": int(paired["Success_baseline"].sum()),
        "candidate_successes": int(paired["Success_candidate"].sum()),
        "failure_to_success": int(improved.sum()),
        "success_to_failure": int(regressed.sum()),
        "metrics": {},
    }
    for metric in METRICS:
        differences = paired[f"Delta_{metric}"].to_numpy(dtype=float)
        summary["metrics"][metric] = {
            "baseline_mean": float(paired[f"{metric}_baseline"].mean()),
            "candidate_mean": float(paired[f"{metric}_candidate"].mean()),
            "mean_delta": float(np.mean(differences)),
            "median_delta": float(np.median(differences)),
            "exploratory_signed_mean_p": exact_signed_mean_pvalue(differences),
        }
    return paired, summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    baseline = pd.read_csv(args.baseline)
    candidate = pd.read_csv(args.candidate)
    paired, summary = compare_frames(baseline, candidate)
    summary.update(
        {
            "created_at": datetime.now().isoformat(),
            "baseline": str(args.baseline),
            "candidate": str(args.candidate),
            "note": "P-values are exploratory exact sign-randomization tests of paired mean differences.",
        }
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    paired.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Paired rows saved to {csv_path}")
    print(f"Summary saved to {json_path}")


if __name__ == "__main__":
    main()
