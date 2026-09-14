#!/usr/bin/env python3
"""Численное исследование модели на синтетической 3D-сцене — БЕЗ видео.

    python scripts/run_synthetic_study.py --config configs/prototype.yaml

Эксперименты:
    E1 — зависимость признаков G1, G2, G3 от рельефа лица;
    E2 — зависимость разделения классов от расстояния до камеры;
    E3 — какое движение головы включает какой признак;
    E4 — классификация полным пайплайном LOSO и устойчивость к шуму;
    E5 — сравнение правил выбора порога;
    E6 — устоит ли модель против атаки воспроизведением (replay).

Результаты: results/synthetic/*.csv и figures/synthetic/*.png

ВНИМАНИЕ: это симуляция идеализированной сцены. Полученные числа описывают
механизм модели и её принципиальные ограничения, но НЕ являются оценкой
точности на реальных предъявлениях и не заменяют эксперимент с участниками.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.plotting import generate_synthetic_plots
from src.synthetic_study import run_synthetic_study
from src.utils import load_config, set_seed, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Синтетическое исследование модели")
    parser.add_argument("--config", default="configs/prototype.yaml")
    parser.add_argument("--subjects", type=int, default=8,
                        help="число синтетических участников для эксперимента E4")
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config(args.config)
    set_seed(config.seed)

    results = run_synthetic_study(config, n_subjects=args.subjects)
    generate_synthetic_plots(config, results)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 72)
    print("СИНТЕТИЧЕСКОЕ ИССЛЕДОВАНИЕ — КРАТКИЕ ИТОГИ")
    print("=" * 72)

    depth = results["depth"]
    flat = depth[depth["nose_depth"] == 0.0].iloc[0]
    relief = depth[depth["nose_depth"] >= 0.12].iloc[0]
    print("\nE1. Рельеф лица (плоскость -> нос вперёд на %.2f):" % relief["nose_depth"])
    for g in ("G1", "G2", "G3"):
        ratio = relief[g] / flat[g] if flat[g] else float("inf")
        print(f"    {g}: {flat[g]:.2e} -> {relief[g]:.2e}  (в {ratio:.1f} раз(а) больше)")

    dist = results["distance"]
    print("\nE2. Расстояние до камеры (разделение по G1):")
    for _, r in dist.iterrows():
        print(f"    d = {r['camera_distance']:6.1f}:  идеально {r['G1_separation_ideal']:9.1f}x"
              f"   |  с шумом {r['G1_separation_noisy']:5.2f}x")

    motion = results["motion"]
    at25 = motion[motion["amplitude_deg"] == 25.0]
    print("\nE3. Режим движения головы (амплитуда 25°):")
    for _, r in at25.iterrows():
        print(f"    {r['regime']:18s}  G1: {r['G1_live']:.2e} / {r['G1_planar']:.2e}"
              f"   G2: {r['G2_live']:.2e} / {r['G2_planar']:.2e}")
    print("    (формат: живое лицо / плоская атака)")

    print("\nE4. Классификация на синтетическом наборе (LOSO):")
    cols = [c for c in ["model", "n", "apcer", "bpcer", "acer", "accuracy", "roc_auc"]
            if c in results["pooled"].columns]
    print(results["pooled"][cols].to_string(index=False))

    replay = results["replay_predictions"]
    share = replay.groupby("attack_type")["predicted_live"].mean()
    names = {"none": "живое лицо", "screen": "фото на экране", "replay": "ВИДЕО на экране"}
    print("\nE6. Атака воспроизведением (обучение только на live + screen):")
    for key in ("none", "screen", "replay"):
        if key in share.index:
            print(f"    {names[key]:18s} принято за живое: {share[key] * 100:5.1f}%")
    if float(share.get("replay", 0.0)) > 0.5:
        print("    ВЫВОД: модель не распознаёт атаку воспроизведением.")
        print("    Это предсказание механизма, а не сбой кода (см. research_log.md).")

    print("\n" + "=" * 72)
    print("Таблицы: results/synthetic/    Графики: figures/synthetic/")
    print("НАПОМИНАНИЕ: это симуляция идеализированной сцены, а не эксперимент")
    print("на людях. Числа описывают механизм модели, а не её точность на")
    print("реальных предъявлениях.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
