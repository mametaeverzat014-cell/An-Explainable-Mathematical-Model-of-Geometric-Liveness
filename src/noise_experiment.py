"""Опционально: контролируемый эксперимент с синтетическим шумом псевдоглубины.

Модель наблюдения:

    z_obs,i(t) = z_true,i(t) + eta_i(t),   eta_i(t) ~ N(0, sigma^2)

Шум добавляется к координате z landmark'ов, после чего ЗАНОВО пересчитываются
r1(t), G1..G3 и выполняется полный LOSO-CV. Базлайн B1 не использует z, поэтому
служит контрольной моделью (его метрики меняются только из-за случайного seed).

Определяется также:

    Delta_ACER(sigma) = ACER_baseline(sigma) - ACER_PLS(sigma)

ОГРАНИЧЕНИЕ: никакого универсального порога шума здесь не утверждается.
Если в отчёте приводится sigma_crit, она должна называться
«оценка порога, специфичная для данного протокола и синтетической модели
возмущений».
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.build_features import G_NAMES, compute_g_features, compute_ratios
from src.evaluation import run_loso
from src.quality_control import valid_frame_mask
from src.utils import Config, ensure_dir, get_logger


def perturb_z(landmark_df: pd.DataFrame, sigma: float, rng: np.random.Generator) -> pd.DataFrame:
    """Вернуть копию таблицы landmark'ов с зашумлённой псевдоглубиной z."""
    out = landmark_df.copy()
    z_cols = [c for c in out.columns if c.endswith("_z")]
    if sigma > 0:
        noise = rng.normal(0.0, sigma, size=(len(out), len(z_cols)))
        out[z_cols] = out[z_cols].to_numpy(dtype=np.float64) + noise
    return out


def recompute_features_with_noise(
    features: pd.DataFrame,
    landmarks_dir: Path,
    sigma: float,
    rng: np.random.Generator,
    epsilon: float,
    max_abs_coord: float,
) -> pd.DataFrame:
    """Пересчитать G1..G3 при заданном уровне шума (остальные колонки не меняются)."""
    noisy = features.copy()
    for i, video_id in enumerate(features["video_id"]):
        csv_path = landmarks_dir / f"{video_id}.csv"
        if not csv_path.exists():
            continue
        landmark_df = pd.read_csv(csv_path)
        valid = landmark_df[valid_frame_mask(landmark_df, max_abs_coord)].reset_index(drop=True)
        if valid.empty:
            continue
        ratios = compute_ratios(perturb_z(valid, sigma, rng), epsilon)
        for name, value in compute_g_features(ratios).items():
            noisy.loc[i, name] = value
    return noisy


def run_noise_experiment(config: Config, features: pd.DataFrame) -> pd.DataFrame:
    """Прогнать сетку sigma и сохранить метрики устойчивости.

    Returns:
        DataFrame: sigma, repeat, model, apcer, bpcer, acer, accuracy.
    """
    from src.plotting import plot_noise_robustness

    logger = get_logger()
    cfg = config.get("noise_experiment", {})
    sigma_grid = [float(s) for s in cfg.get("sigma_grid", [0.0])]
    repeats = int(cfg.get("repeats", 10))
    epsilon = float(config.get("features", {}).get("epsilon", 1e-8))
    max_abs_coord = float(config.get("quality_control", {}).get("max_abs_coord", 5.0))
    landmarks_dir = config.path("landmarks_dir")
    results_dir = ensure_dir(config.path("results_dir"))

    if features.empty or features["subject_id"].nunique() < 2:
        logger.warning("Эксперимент с шумом пропущен: недостаточно данных.")
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for sigma in sigma_grid:
        n_rep = 1 if sigma == 0.0 else repeats  # при sigma=0 повторы идентичны
        for rep in range(n_rep):
            rng = np.random.default_rng(config.seed + rep * 1000 + int(sigma * 1e6))
            noisy_features = recompute_features_with_noise(
                features, landmarks_dir, sigma, rng, epsilon, max_abs_coord
            )
            _, _, pooled, _ = run_loso(
                noisy_features, config, save=False, bootstrap=False
            )
            for _, row in pooled.iterrows():
                rows.append(
                    {
                        "sigma": sigma,
                        "repeat": rep,
                        "model": row["model"],
                        "apcer": row["apcer"],
                        "bpcer": row["bpcer"],
                        "acer": row["acer"],
                        "accuracy": row["accuracy"],
                    }
                )
        logger.info("Шум sigma=%.4f: готово (%d повтор(ов))", sigma, n_rep)

    noise_df = pd.DataFrame(rows)
    if noise_df.empty:
        return noise_df

    # Delta_ACER(sigma) = ACER_baseline - ACER_PLS (положительное => PLS лучше)
    pivot = noise_df.groupby(["sigma", "model"])["acer"].mean().unstack("model")
    if "B1_static2d" in pivot.columns and "M1_pls" in pivot.columns:
        delta = (pivot["B1_static2d"] - pivot["M1_pls"]).rename("delta_acer").reset_index()
        delta.to_csv(results_dir / "noise_delta_acer.csv", index=False)

    out = results_dir / "noise_robustness_metrics.csv"
    noise_df.to_csv(out, index=False)
    logger.info("Метрики устойчивости к шуму: %s", out)
    plot_noise_robustness(config, noise_df)

    return noise_df
