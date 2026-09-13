#!/usr/bin/env python3
"""Пересчитать LOSO-CV, метрики и графики из готового results/video_features.csv.

Полезно, когда landmark'и уже извлечены и нужно только изменить порог,
направление правила или набор моделей:

    python scripts/run_evaluation.py --config configs/prototype.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.evaluation import run_loso
from src.plotting import generate_all_plots
from src.utils import get_logger, load_config, set_seed, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Оценка LOSO-CV по готовым признакам")
    parser.add_argument("--config", default="configs/prototype.yaml")
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config(args.config)
    set_seed(config.seed)

    features_csv = config.path("results_dir") / "video_features.csv"
    if not features_csv.exists():
        logger.error(
            "Файл признаков не найден: %s\n"
            "Сначала запустите: python scripts/run_pipeline.py --config %s",
            features_csv,
            args.config,
        )
        return 2

    features = pd.read_csv(features_csv)
    if features.empty or features["subject_id"].nunique() < 2:
        logger.error("Недостаточно данных для LOSO-CV (нужно >= 2 субъектов).")
        return 3

    predictions, fold_metrics, pooled_metrics, _ = run_loso(features, config)
    generate_all_plots(config, predictions, pooled_metrics)

    print("\nPooled LOSO-CV метрики:")
    cols = [c for c in ["model", "n", "apcer", "bpcer", "acer", "accuracy", "roc_auc"]
            if c in pooled_metrics.columns]
    print(pooled_metrics[cols].to_string(index=False))
    print("\nПометрично по фолдам: results/loso_fold_metrics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
