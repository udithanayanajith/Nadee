from __future__ import annotations

import argparse
from pathlib import Path
import sys

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project))
from train_gnn import run_gnn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the temporal GNN and create its plots")
    parser.add_argument("--data", type=Path, default=project / "dataset" / "Component4_ML_Training_Dataset_Synthetic_1000.csv")
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()
    run_gnn(args.data, epochs=args.epochs)
