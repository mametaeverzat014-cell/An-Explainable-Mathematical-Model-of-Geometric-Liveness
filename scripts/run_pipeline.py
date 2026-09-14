#!/usr/bin/env python3
"""Одна команда, запускающая весь пайплайн MVP.

    python scripts/run_pipeline.py --config configs/prototype.yaml

Этапы:
    1. чтение и валидация metadata.csv;
    2. извлечение landmark'ов MediaPipe (results/landmarks/*.csv);
    3. контроль качества (results/landmark_quality_report.csv);
    4. геометрические признаки IOD, r1..r3, G1..G3 (results/video_features.csv);
    5. subject-disjoint LOSO-CV для B0 / B1 / M1;
    6. сохранение предсказаний и метрик;
    7. построение графиков;
    8. (опционально) абляция и эксперимент с шумом.

Если данных нет, скрипт НЕ выдумывает метрики, а печатает инструкцию,
куда положить видео.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Позволяет запускать скрипт напрямую из корня репозитория.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.build_features import build_video_features
from src.evaluation import run_loso
from src.extract_landmarks import LandmarkBackendError, run_extraction
from src.plotting import generate_all_plots, plot_ablation, plot_landmark_example
from src.quality_control import run_quality_control
from src.utils import (
    MetadataError,
    describe_dataset,
    ensure_dir,
    get_logger,
    load_config,
    load_metadata,
    no_data_message,
    set_seed,
    setup_logging,
)


def run_ablation(config, features: pd.DataFrame) -> pd.DataFrame:
    """Опционально: PLS без G1 / без G2 / без G3 (pooled LOSO-CV)."""
    from src.pls_model import PLSModel

    logger = get_logger()
    variants = {
        "PLS_full_G1G2G3": ["G1", "G2", "G3"],
        "PLS_without_G1": ["G2", "G3"],
        "PLS_without_G2": ["G1", "G3"],
        "PLS_without_G3": ["G1", "G2"],
    }
    rows = []
    for variant, g_feats in variants.items():
        preds, _, pooled, _ = run_loso(
            features,
            config,
            model_builders={variant: (lambda gf=g_feats: PLSModel(config, g_features=gf))},
            save=False,
            bootstrap=False,
        )
        if pooled.empty:
            continue
        row = pooled.iloc[0].to_dict()
        row["variant"] = variant
        row["g_features"] = "+".join(g_feats)
        rows.append(row)
    ablation = pd.DataFrame(rows)
    if not ablation.empty:
        out = config.path("results_dir") / "ablation_metrics.csv"
        ablation.to_csv(out, index=False)
        logger.info("Абляция сохранена: %s", out)
    return ablation


def main() -> int:
    parser = argparse.ArgumentParser(description="Полный пайплайн PAD-прототипа")
    parser.add_argument("--config", default="configs/prototype.yaml", help="путь к YAML-конфигу")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="не пересчитывать landmark'и (использовать готовые CSV)")
    parser.add_argument("--skip-noise", action="store_true",
                        help="пропустить опциональный эксперимент с шумом")
    parser.add_argument("--verbose", action="store_true", help="подробный лог")
    args = parser.parse_args()

    logger = setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    try:
        config = load_config(args.config)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    set_seed(config.seed)
    ensure_dir(config.path("results_dir"))
    ensure_dir(config.path("figures_dir"))
    raw_dir, metadata_csv = config.path("raw_dir"), config.path("metadata_csv")

    # --- 1. метаданные --------------------------------------------------
    try:
        metadata = load_metadata(metadata_csv, raw_dir)
    except MetadataError as exc:
        logger.error("Ошибка metadata.csv: %s", exc)
        return 2

    if metadata.empty:
        print(no_data_message(raw_dir, metadata_csv))
        return 0
    logger.info("Набор данных: %s", describe_dataset(metadata))

    # --- 2. landmark'и ---------------------------------------------------
    try:
        if args.skip_extraction:
            landmarks_dir = config.path("landmarks_dir")
            extraction = pd.DataFrame(
                [
                    {
                        "video_id": vid,
                        "landmark_csv": str(landmarks_dir / f"{vid}.csv"),
                        "n_frames": 0,
                        "n_valid_frames": 0,
                        "extraction_error": "",
                    }
                    for vid in metadata["video_id"]
                ]
            )
        else:
            extraction = run_extraction(config, metadata)
    except LandmarkBackendError as exc:
        logger.error("MediaPipe недоступен:\n%s", exc)
        return 3

    # --- 3. контроль качества --------------------------------------------
    qc_report = run_quality_control(config, metadata, extraction)
    if not qc_report["passed"].any():
        logger.error(
            "Ни одно видео не прошло контроль качества. "
            "Причины см. в results/landmark_quality_report.csv — метрики не вычисляются."
        )
        return 4

    # --- 4. признаки ------------------------------------------------------
    features = build_video_features(config, metadata, qc_report, extraction)
    if features.empty:
        logger.error("Таблица признаков пуста — оценка невозможна.")
        return 4

    # График с семантическими точками строится СРАЗУ: он не зависит от LOSO-CV и
    # нужен для пробной съёмки одним участником, чтобы проверить разметку точек.
    first_video = metadata.iloc[0]
    example = (
        config.path("landmarks_dir") / f"{first_video['video_id']}.csv",
        Path(first_video["abs_path"]),
    )
    plot_landmark_example(config, *example)

    n_subjects = features["subject_id"].nunique()
    if n_subjects < 2:
        logger.error(
            "Для LOSO-CV нужно минимум 2 субъекта, найдено: %d. "
            "Добавьте видео других участников в data/raw/.",
            n_subjects,
        )
        print(
            "\n" + "=" * 72 + "\n"
            "ПРОБНЫЙ ЗАПУСК ВЫПОЛНЕН ЧАСТИЧНО: метрики требуют >= 2 участников.\n"
            + "=" * 72 + "\n"
            "Но проверить качество разметки уже можно:\n"
            f"  1) {config.path('results_dir') / 'landmark_quality_report.csv'}\n"
            "     -> колонка passed должна быть True, иначе смотрите колонку reason\n"
            f"  2) {config.path('figures_dir') / 'landmark_example.png'}\n"
            "     -> точки должны лежать на носу, глазах, щеках и подбородке\n"
            f"  3) {config.path('results_dir') / 'video_features.csv'}\n"
            "     -> значения G1, G2, G3 для каждого видео\n\n"
            "Когда разметка выглядит правильно, добавьте видео других участников.\n"
            + "=" * 72
        )
        return 5
    if features["y"].nunique() < 2:
        logger.warning(
            "В данных присутствует только один класс — метрики APCER/BPCER будут неполными."
        )

    # --- 5-6. LOSO-CV и метрики -------------------------------------------
    predictions, fold_metrics, pooled_metrics, thresholds = run_loso(features, config)

    # --- 7. графики --------------------------------------------------------
    generate_all_plots(config, predictions, pooled_metrics, landmark_example=example)

    # --- 8. опциональные эксперименты --------------------------------------
    eval_cfg = config.get("evaluation", {})
    if bool(eval_cfg.get("ablation", False)):
        ablation = run_ablation(config, features)
        if not ablation.empty:
            plot_ablation(config, ablation)

    if bool(config.get("noise_experiment", {}).get("enabled", False)) and not args.skip_noise:
        from src.noise_experiment import run_noise_experiment

        run_noise_experiment(config, features)

    # --- итоговая сводка ----------------------------------------------------
    print("\n" + "=" * 72)
    print("ИТОГОВЫЕ МЕТРИКИ (pooled LOSO-CV, subject-disjoint)")
    print("=" * 72)
    cols = [c for c in ["model", "n", "apcer", "bpcer", "acer", "accuracy", "roc_auc"]
            if c in pooled_metrics.columns]
    print(pooled_metrics[cols].to_string(index=False))
    print("=" * 72)
    print(f"Субъектов (фолдов): {n_subjects} | видео в оценке: {len(features)}")
    print(f"Все результаты: {config.path('results_dir')}")
    print(f"Все графики:    {config.path('figures_dir')}")
    print("Напоминание: z MediaPipe — оценочная относительная псевдоглубина,")
    print("а не физически измеренная глубина. Выводы применимы только к")
    print("протестированным людям, устройствам и условиям съёмки.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
