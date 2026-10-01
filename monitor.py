from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.modeling import FEATURES, population_stability_index


def drift_report(reference_path: Path, current_path: Path, threshold: float = 0.2) -> dict:
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)
    scores = {
        feature: round(population_stability_index(reference[feature].to_numpy(), current[feature].to_numpy()), 4)
        for feature in FEATURES
    }
    flagged = sorted(feature for feature, score in scores.items() if score >= threshold)
    return {"threshold": threshold, "flagged_features": flagged, "scores": scores}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("current", type=Path, help="CSV with the model feature columns")
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()
    root = Path(__file__).parent
    report = drift_report(root / "artifacts" / "reference_sample.csv", args.current, args.threshold)
    output = root / "artifacts" / "drift_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

