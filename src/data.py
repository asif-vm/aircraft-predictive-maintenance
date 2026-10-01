from __future__ import annotations

from pathlib import Path
from io import BytesIO
import zipfile

import numpy as np
import pandas as pd
import requests

NASA_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.%20Turbofan%20Engine%20Degradation%20Simulation%20Data%20Set.zip"
SENSORS = [f"sensor_{i}" for i in range(1, 22)]
COLS = ["engine_id", "cycle", "setting_1", "setting_2", "setting_3", *SENSORS]


def generate_demo(path: Path, engines: int = 120, seed: int = 42) -> Path:
    """Create a C-MAPSS-shaped run-to-failure dataset for an offline demo."""
    rng = np.random.default_rng(seed)
    records = []
    for engine_id in range(1, engines + 1):
        life = int(rng.integers(120, 260))
        slopes = rng.uniform(.2, 1.0, len(SENSORS)) * rng.choice([-1, 1], len(SENSORS))
        baseline = rng.normal(50, 8, len(SENSORS))
        for cycle in range(1, life + 1):
            degradation = cycle / life
            sensors = baseline + slopes * degradation * 12 + rng.normal(0, .7, len(SENSORS))
            records.append([engine_id, cycle, rng.normal(), rng.normal(), rng.normal(), *sensors])
    df = pd.DataFrame(records, columns=COLS)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def download_nasa(destination: Path) -> Path:
    archive = destination.parent / "cmapps.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(NASA_URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        with archive.open("wb") as out:
            for chunk in response.iter_content(1024 * 1024):
                out.write(chunk)
    with zipfile.ZipFile(archive) as outer:
        names = outer.namelist()
        direct = next((name for name in names if name.endswith("train_FD001.txt")), None)
        if direct:
            source_bytes = outer.read(direct)
        else:
            nested_name = next(name for name in names if name.endswith("CMAPSSData.zip"))
            with zipfile.ZipFile(BytesIO(outer.read(nested_name))) as nested:
                member = next(name for name in nested.namelist() if name.endswith("train_FD001.txt"))
                source_bytes = nested.read(member)
    df = pd.read_csv(BytesIO(source_bytes), sep=r"\s+", header=None, names=COLS)
    df.to_csv(destination, index=False)
    archive.unlink(missing_ok=True)
    return destination


def add_target(df: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    out = df.copy()
    final_cycle = out.groupby("engine_id")["cycle"].transform("max")
    out["rul"] = (final_cycle - out["cycle"]).clip(upper=cap)
    return out
