#!/usr/bin/env python3
"""Демонстрация модели на одной записи — для стенда.

    python scripts/demo.py data/raw/live/s001_live_01.mp4

Скрипт извлекает точки лица, считает G1, G2, G3, применяет обученную модель и
показывает не только вердикт, но и **вклад каждого признака**. Именно
покомпонентный разбор отличает эту модель от нейросетевого «чёрного ящика»:
видно, какая именно геометрическая величина привела к решению.

Если landmark'и уже извлечены, повторное извлечение можно пропустить:

    python scripts/demo.py --landmarks results/landmarks/s001_live_01.csv

ЧЕСТНОСТЬ ДЕМОНСТРАЦИИ. Модель обучается на всех собранных записях из
results/video_features.csv. Если показать ей запись, которая входила в
обучение, вердикт не является проверкой: скрипт это обнаруживает и печатает
предупреждение. Для настоящей проверки нужна запись нового человека.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.build_features import compute_g_features, compute_ratios
from src.pls_model import PLSModel
from src.quality_control import valid_frame_mask
from src.utils import Config, get_logger, load_config, set_seed, setup_logging

WIDTH = 68

#: Человеческие названия признаков — для стенда, а не для отчёта.
FEATURE_MEANING = {
    "G1": "псевдоглубина носа относительно щёк",
    "G2": "лево-правая асимметрия (основной признак)",
    "G3": "вертикальная пропорция лица",
}


def rule(char: str = "=") -> str:
    return char * WIDTH


def extract_landmarks_from_video(config: Config, video: Path) -> pd.DataFrame:
    """Извлечь точки лица из видео. Требует mediapipe и opencv."""
    from src.extract_landmarks import create_backend, extract_video_landmarks

    ex_cfg = dict(config.get("extraction", {}))
    backend = create_backend(ex_cfg)
    return extract_video_landmarks(
        video_path=video,
        video_id=video.stem,
        backend=backend,
        max_frames=int(ex_cfg.get("max_frames", 120)),
        stride=int(ex_cfg.get("frame_stride", 1)),
    )


def describe_quality(landmarks: pd.DataFrame, valid: pd.DataFrame) -> None:
    total = len(landmarks)
    found = int((landmarks["face_found"] == 1).sum()) if total else 0
    print(f"  кадров обработано:      {total}")
    print(f"  лицо найдено:           {found}")
    print(f"  пригодно для расчёта:   {len(valid)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Показать вердикт и вклад признаков для одной записи"
    )
    parser.add_argument("video", nargs="?", help="путь к видеофайлу")
    parser.add_argument("--landmarks", help="готовый landmark-CSV вместо видео")
    parser.add_argument("--config", default="configs/prototype.yaml")
    args = parser.parse_args()

    if not args.video and not args.landmarks:
        parser.error("укажите видеофайл или --landmarks")

    logger = setup_logging()
    config = load_config(args.config)
    set_seed(config.seed)

    # ---------------------------------------------------- обучающая выборка
    features_csv = config.path("results_dir") / "video_features.csv"
    if not features_csv.exists():
        print(
            f"\nНет файла с признаками: {features_csv}\n"
            "Демонстрация требует обученной модели. Сначала выполните:\n"
            f"  python scripts/run_pipeline.py --config {args.config}"
        )
        return 2

    train = pd.read_csv(features_csv)
    if train.empty or train["y"].nunique() < 2:
        print("\nВ обучающей выборке нет обоих классов — модель обучить нельзя.")
        return 2

    # ---------------------------------------------------- признаки записи
    if args.landmarks:
        source = Path(args.landmarks)
        if not source.exists():
            print(f"\nФайл не найден: {source}")
            return 2
        landmarks = pd.read_csv(source)
        video_id = source.stem
    else:
        source = Path(args.video)
        if not source.exists():
            print(f"\nФайл не найден: {source}")
            return 2
        video_id = source.stem
        logger.info("Извлечение точек лица: %s", source.name)
        landmarks = extract_landmarks_from_video(config, source)

    max_abs = float(config.get("quality_control", {}).get("max_abs_coord", 5.0))
    valid = landmarks[valid_frame_mask(landmarks, max_abs)].reset_index(drop=True)

    print("\n" + rule())
    print(f"  ЗАПИСЬ: {video_id}")
    print(rule())
    print("\nКОНТРОЛЬ КАЧЕСТВА")
    describe_quality(landmarks, valid)

    min_frames = int(config.get("quality_control", {}).get("min_valid_frames", 20))
    if len(valid) < min_frames:
        print(
            f"\n  ОТКАЗ: пригодных кадров {len(valid)}, требуется минимум {min_frames}.\n"
            "  Модель не выдаёт вердикт на записи такого качества — это\n"
            "  предусмотрено протоколом, а не сбой."
        )
        return 1

    ratios = compute_ratios(valid, float(config.get("features", {}).get("epsilon", 1e-8)))
    g_values = compute_g_features(ratios)

    # ---------------------------------------------------- модель
    model = PLSModel(config).fit(train)
    row = pd.DataFrame([{**g_values, "video_id": video_id}])
    explained = model.explain(row).iloc[0]
    score = float(explained["score_oriented"])
    tau = float(model.tau_)
    is_live = score >= tau

    print("\nГЕОМЕТРИЧЕСКИЕ ПРИЗНАКИ (дисперсия отношения по времени)")
    for name in ("G1", "G2", "G3"):
        print(f"  {name} = {g_values[name]:.3e}   {FEATURE_MEANING[name]}")

    # Вклад в решение: PLS есть среднее нормированных признаков, поэтому
    # каждый вносит ровно Gk_norm / 3. Знак зависит от направления правила.
    sign = 1.0 if model.direction_ == "live_high" else -1.0
    contributions = [
        (name, sign * float(explained[f"{name}_norm"]) / 3.0)
        for name in ("G1", "G2", "G3")
    ]
    # Полосы масштабируются по самому сильному вкладу этой записи, иначе при
    # малых значениях они вырождаются в один-два символа и ничего не показывают.
    scale = max(abs(value) for _, value in contributions) or 1.0
    print("\nВКЛАД В РЕШЕНИЕ (положительный — в пользу «живое»)")
    for name, value in contributions:
        bar = ("#" if value >= 0 else "-") * max(1, round(abs(value) / scale * 26))
        print(f"  {name}: {value:+7.3f}  {bar}")

    leader = max(contributions, key=lambda item: abs(item[1]))
    print(f"\n  наибольший вклад: {leader[0]} — {FEATURE_MEANING[leader[0]]}")

    print("\nИТОГ")
    print(f"  оценка живости PLS:     {score:+.3f}")
    print(f"  порог tau:              {tau:+.3f}   (правило: {model.threshold_rule_})")
    print(f"\n  ВЕРДИКТ: {'ЖИВОЕ ЛИЦО' if is_live else 'ПЛОСКАЯ АТАКА'}")

    # ---------------------------------------------------- честные оговорки
    print("\n" + rule("-"))
    if video_id in set(train["video_id"].astype(str)):
        print("  ВНИМАНИЕ: эта запись ВХОДИЛА В ОБУЧЕНИЕ модели.")
        print("  Вердикт показывает работу разбора признаков, но проверкой")
        print("  точности не является. Для проверки нужна запись нового человека.")
    else:
        print(f"  Запись не входила в обучающую выборку ({len(train)} записей,")
        print(f"  {train['subject_id'].nunique()} участников).")
    print(
        "\n  Модель обучена на атаках ОДНОГО типа — статическое фото с экрана.\n"
        "  Против атаки воспроизведением (видео на экране) она не работает:\n"
        "  это доказано (утверждение 7) и проверено численно.\n"
        "  Прототип исследовательский; для контроля доступа не предназначен."
    )
    print(rule("-"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
