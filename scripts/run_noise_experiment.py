#!/usr/bin/env python3
"""Запустить только эксперимент с синтетическим шумом псевдоглубины.

    python scripts/run_noise_experiment.py --config configs/prototype.yaml

Скрипт НЕ затрагивает основные результаты LOSO-CV в results/: все прогоны с
шумом выполняются с save=False и пишут только noise_robustness_metrics.csv
и noise_delta_acer.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.noise_experiment import run_noise_experiment
from src.utils import load_config, set_seed, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Эксперимент с шумом псевдоглубины")
    parser.add_argument("--config", default="configs/prototype.yaml")
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config(args.config)
    set_seed(config.seed)

    features_csv = config.path("results_dir") / "video_features.csv"
    if not features_csv.exists():
        logger.error(
            "Нет %s. Сначала запустите scripts/run_pipeline.py", features_csv
        )
        return 2

    noise_df = run_noise_experiment(config, pd.read_csv(features_csv))
    if noise_df.empty:
        logger.error("Эксперимент не дал результатов (недостаточно данных).")
        return 3
    print(noise_df.groupby(["model", "sigma"])["acer"].mean().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
