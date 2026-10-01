import argparse
from pathlib import Path

from src.data import download_nasa, generate_demo
from src.modeling import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--engines", type=int, default=120)
    args = parser.parse_args()
    root = Path(__file__).parent
    data_path = root / "data" / ("train_FD001.csv" if args.real else "demo_cmapps.csv")
    if not data_path.exists():
        download_nasa(data_path) if args.real else generate_demo(data_path, args.engines)
    print(train(data_path, root / "artifacts"))

