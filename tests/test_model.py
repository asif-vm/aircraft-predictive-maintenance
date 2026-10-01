from pathlib import Path

import pandas as pd
from src.data import add_target, generate_demo
from src.modeling import train
from src.modeling import population_stability_index


def test_engine_split_and_artifacts(tmp_path: Path):
    data = generate_demo(tmp_path / "demo.csv", engines=30)
    metrics = train(data, tmp_path / "artifacts")
    assert metrics["test_engines"] == 6
    assert metrics["mae_cycles"] >= 0
    assert (tmp_path / "artifacts" / "model.joblib").exists()
    targeted = add_target(pd.read_csv(data))
    assert targeted.groupby("engine_id")["rul"].min().eq(0).all()
    values = targeted["sensor_1"].to_numpy()
    assert population_stability_index(values, values) == 0.0
