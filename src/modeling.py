from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data import SENSORS, add_target

FEATURES = ["cycle", "setting_1", "setting_2", "setting_3", *SENSORS]


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    exp = np.histogram(expected, bins=edges)[0] / max(len(expected), 1)
    act = np.histogram(actual, bins=edges)[0] / max(len(actual), 1)
    exp, act = np.clip(exp, 1e-6, None), np.clip(act, 1e-6, None)
    return float(np.sum((act - exp) * np.log(act / exp)))


def train(csv_path: Path, artifact_dir: Path) -> dict[str, float | int]:
    df = add_target(pd.read_csv(csv_path))
    engine_ids = np.array(sorted(df["engine_id"].unique()))
    cutoff = int(len(engine_ids) * .8)
    train_ids, test_ids = engine_ids[:cutoff], engine_ids[cutoff:]
    train_df = df[df.engine_id.isin(train_ids)]
    test_df = df[df.engine_id.isin(test_ids)]
    model = HistGradientBoostingRegressor(max_iter=220, learning_rate=.06, max_leaf_nodes=24, l2_regularization=.3, random_state=42)
    model.fit(train_df[FEATURES], train_df["rul"])
    predictions = np.clip(model.predict(test_df[FEATURES]), 0, 125)
    critical = test_df["rul"].to_numpy() <= 30
    predicted_critical = predictions <= 30
    critical_recall = float((predicted_critical & critical).sum() / max(critical.sum(), 1))
    metrics = {
        "train_engines": int(len(train_ids)), "test_engines": int(len(test_ids)),
        "mae_cycles": round(float(mean_absolute_error(test_df["rul"], predictions)), 4),
        "rmse_cycles": round(float(mean_squared_error(test_df["rul"], predictions) ** .5), 4),
        "critical_recall": round(critical_recall, 4),
    }
    importance = permutation_importance(model, test_df[FEATURES].sample(min(2500, len(test_df)), random_state=42), test_df["rul"].sample(min(2500, len(test_df)), random_state=42), n_repeats=3, random_state=42)
    importance_df = pd.DataFrame({"feature": FEATURES, "importance": importance.importances_mean}).sort_values("importance", ascending=False)
    reference = train_df[FEATURES].agg(["mean", "std", "min", "max"]).T.reset_index(names="feature")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, artifact_dir / "model.joblib")
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    importance_df.to_csv(artifact_dir / "feature_importance.csv", index=False)
    reference.to_csv(artifact_dir / "reference_stats.csv", index=False)
    return metrics

