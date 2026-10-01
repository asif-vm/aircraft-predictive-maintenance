import argparse
from pathlib import Path

from src.data import download_nasa, generate_demo
from src.modeling import train


def log_mlflow(metrics: dict, artifact_dir: Path) -> None:
    try:
        import mlflow
    except ImportError as exc:
        raise SystemExit("Install `requirements-mlops.txt` to enable MLflow logging") from exc
    tracking_db = (artifact_dir.parent / "mlflow.db").resolve().as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
    mlflow.set_experiment("aircraft-rul")
    with mlflow.start_run(run_name="hist-gradient-boosting-fd001"):
        mlflow.log_params({"split": "engine-level-80-20", "rul_cap": 125, "critical_window": 30})
        mlflow.log_metrics(metrics)
        for artifact in artifact_dir.iterdir():
            mlflow.log_artifact(str(artifact), artifact_path="model_outputs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--engines", type=int, default=120)
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent
    data_path = root / "data" / ("train_FD001.csv" if args.real else "demo_cmapps.csv")
    if not data_path.exists():
        download_nasa(data_path) if args.real else generate_demo(data_path, args.engines)
    artifact_dir = root / "artifacts"
    metrics = train(data_path, artifact_dir)
    if args.mlflow:
        log_mlflow(metrics, artifact_dir)
    print(metrics)
