"""Численное исследование модели на синтетической 3D-сцене (без видео).

Эксперименты отвечают на вопросы, которые нельзя проверить на реальных
данных, не имея их в большом количестве:

* **E1** — как временные признаки G1, G2, G3 зависят от рельефа лица;
* **E2** — как сигнал слабеет с удалением лица от камеры;
* **E3** — какое движение головы нужно, чтобы признаки заработали;
* **E4** — разделяются ли классы полным пайплайном (PLS против базлайна)
  и как разделение разрушается шумом псевдоглубины.

ГРАНИЦЫ ПРИМЕНИМОСТИ. Это симуляция идеализированной сцены: плоская атака
здесь идеально плоская, а точки определяются без ошибок. Настоящая
распечатка имеет текстуру, экран — блики и муар, а MediaPipe вносит
собственный шум. Результаты описывают **механизм** модели и её принципиальные
ограничения, но НЕ являются оценкой точности на реальных предъявлениях и не
заменяют эксперимент с участниками.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.build_features import (
    BASELINE_FEATURE_NAMES,
    compute_baseline_features,
    compute_g_features,
    compute_ratios,
)
from src.constants import LABEL_TO_INT
from src.synthetic_face import (
    CAMERA_DISTANCE,
    LIVE_NOSE_DEPTH,
    PLANAR_NOSE_DEPTH,
    head_motion,
    render_sequence,
)
from src.utils import Config, ensure_dir, get_logger

#: Число кадров в синтетической «записи» (~1.3 с при 30 fps).
N_FRAMES = 40


# --------------------------------------------------------------------------
# Вспомогательные расчёты
# --------------------------------------------------------------------------
def g_features_for(
    nose_depth: float,
    yaw: np.ndarray,
    pitch: np.ndarray,
    camera_distance: float = CAMERA_DISTANCE,
    sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Посчитать G1, G2, G3 для одной синтетической записи."""
    frames = render_sequence(nose_depth, yaw, pitch, camera_distance)
    if sigma > 0:
        if rng is None:
            rng = np.random.default_rng(0)
        z_cols = [c for c in frames.columns if c.endswith("_z")]
        frames[z_cols] = frames[z_cols].to_numpy(dtype=float) + rng.normal(
            0.0, sigma, size=(len(frames), len(z_cols))
        )
    return compute_g_features(compute_ratios(frames))


def separation(planar: float, relief: float) -> float:
    """Во сколько раз признак у рельефного лица больше, чем у плоского."""
    if planar <= 0:
        return float("inf") if relief > 0 else float("nan")
    return float(relief / planar)


# --------------------------------------------------------------------------
# E1. Зависимость признаков от рельефа лица
# --------------------------------------------------------------------------
def experiment_depth_sweep(depths: Sequence[float] | None = None) -> pd.DataFrame:
    """Как G1, G2, G3 растут с выносом носа вперёд."""
    if depths is None:
        depths = np.round(np.linspace(0.0, 0.25, 11), 3).tolist()
    yaw, pitch = head_motion(N_FRAMES)
    rows = []
    for depth in depths:
        g = g_features_for(depth, yaw, pitch)
        rows.append({"nose_depth": float(depth), **g})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# E2. Зависимость от расстояния до камеры
# --------------------------------------------------------------------------
def experiment_distance_sweep(
    distances: Sequence[float] | None = None,
    noise_sigma: float = 0.002,
    repeats: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """Как меняется разделение классов при удалении лица от камеры.

    Считаются два варианта: идеальный (без шума) и с шумом псевдоглубины.
    Это принципиально: в идеальной сцене отношение G1(живое)/G1(плоское)
    с расстоянием даже РАСТЁТ, потому что вклад перспективы у плоского
    объекта убывает быстрее. Но абсолютные значения G1 при этом падают, и
    при любом реальном шуме определения точек сигнал тонет. Требование
    протокола «лицо крупно в кадре» обосновывается именно вторым столбцом.

    Args:
        distances: расстояния до камеры.
        noise_sigma: шум псевдоглубины для «реалистичного» варианта.
        repeats: число повторов усреднения для зашумлённого варианта.
        seed: seed генератора.
    """
    if distances is None:
        distances = [1.5, 2.0, 3.0, 5.0, 8.0, 15.0, 40.0, 200.0]
    yaw, pitch = head_motion(N_FRAMES)
    rng = np.random.default_rng(seed)
    rows = []
    for dist in distances:
        g_live = g_features_for(LIVE_NOSE_DEPTH, yaw, pitch, camera_distance=dist)
        g_flat = g_features_for(PLANAR_NOSE_DEPTH, yaw, pitch, camera_distance=dist)

        noisy_live, noisy_flat = [], []
        for _ in range(repeats):
            noisy_live.append(
                g_features_for(LIVE_NOSE_DEPTH, yaw, pitch, dist, noise_sigma, rng)["G1"]
            )
            noisy_flat.append(
                g_features_for(PLANAR_NOSE_DEPTH, yaw, pitch, dist, noise_sigma, rng)["G1"]
            )
        mean_live, mean_flat = float(np.mean(noisy_live)), float(np.mean(noisy_flat))

        rows.append(
            {
                "camera_distance": float(dist),
                "G1_live": g_live["G1"],
                "G1_planar": g_flat["G1"],
                "G1_separation_ideal": separation(g_flat["G1"], g_live["G1"]),
                "G1_live_noisy": mean_live,
                "G1_planar_noisy": mean_flat,
                "G1_separation_noisy": separation(mean_flat, mean_live),
                "G2_live": g_live["G2"],
                "G2_planar": g_flat["G2"],
                "G2_separation_ideal": separation(g_flat["G2"], g_live["G2"]),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# E3. Зависимость от типа и амплитуды движения головы
# --------------------------------------------------------------------------
def experiment_motion_types(amplitudes: Sequence[float] | None = None) -> pd.DataFrame:
    """Какое движение головы нужно, чтобы признаки разделяли классы.

    Сравниваются три режима: только поворот, только наклон, поворот с наклоном.
    """
    if amplitudes is None:
        amplitudes = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    zero = np.zeros(N_FRAMES)
    rows = []
    for amp in amplitudes:
        swing = np.linspace(-amp, amp, N_FRAMES)
        regimes = {
            "только поворот": (swing, zero),
            "только наклон": (zero, swing),
            "поворот + наклон": (swing, swing * 0.6),
        }
        for regime, (yaw, pitch) in regimes.items():
            g_live = g_features_for(LIVE_NOSE_DEPTH, yaw, pitch)
            g_flat = g_features_for(PLANAR_NOSE_DEPTH, yaw, pitch)
            rows.append(
                {
                    "amplitude_deg": float(amp),
                    "regime": regime,
                    "G1_live": g_live["G1"],
                    "G1_planar": g_flat["G1"],
                    "G1_separation": separation(g_flat["G1"], g_live["G1"]),
                    "G2_live": g_live["G2"],
                    "G2_planar": g_flat["G2"],
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# E4. Полный пайплайн на синтетическом наборе
# --------------------------------------------------------------------------
def build_synthetic_dataset(
    n_subjects: int = 8,
    n_videos_per_class: int = 2,
    sigma: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Собрать синтетический набор данных в формате ``video_features.csv``.

    Каждый «участник» получает индивидуальный рельеф лица, своё расстояние до
    камеры и свою манеру движения — чтобы субъект-дизъюнктная проверка была
    содержательной, а не тривиальной.

    Args:
        n_subjects: число синтетических участников (= число фолдов LOSO).
        n_videos_per_class: записей каждого класса на участника.
        sigma: шум псевдоглубины (0 — без шума).
        seed: seed генератора.

    Returns:
        Таблица признаков уровня видео, пригодная для :func:`src.evaluation.run_loso`.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for subject_idx in range(n_subjects):
        subject_id = f"synthetic_{subject_idx + 1:03d}"
        # индивидуальные особенности участника
        subject_depth = float(np.clip(rng.normal(LIVE_NOSE_DEPTH, 0.025), 0.05, 0.22))
        subject_distance = float(np.clip(rng.normal(CAMERA_DISTANCE, 0.35), 1.2, 3.5))
        subject_amp = float(np.clip(rng.normal(25.0, 6.0), 10.0, 38.0))

        specs = [("live", "none", subject_depth)]
        specs += [("attack", "print", PLANAR_NOSE_DEPTH)]
        specs += [("attack", "screen", PLANAR_NOSE_DEPTH)]

        for label, attack_type, depth in specs:
            for take in range(n_videos_per_class):
                # дубль к дублю движение слегка отличается
                amp = subject_amp * float(rng.uniform(0.85, 1.15))
                yaw, pitch = head_motion(N_FRAMES, amp, amp * 0.6)
                yaw = yaw + rng.normal(0, 0.6, N_FRAMES)
                pitch = pitch + rng.normal(0, 0.4, N_FRAMES)
                distance = subject_distance * float(rng.uniform(0.95, 1.05))

                frames = render_sequence(depth, yaw, pitch, distance)
                if sigma > 0:
                    z_cols = [c for c in frames.columns if c.endswith("_z")]
                    frames[z_cols] = frames[z_cols].to_numpy(dtype=float) + rng.normal(
                        0.0, sigma, size=(len(frames), len(z_cols))
                    )

                ratios = compute_ratios(frames)
                g_feats = compute_g_features(ratios)
                base = compute_baseline_features(frames.iloc[len(frames) // 2])
                base["b_sharpness"] = float("nan")  # видео нет, резкость не определена

                suffix = "live" if label == "live" else attack_type
                rows.append(
                    {
                        "video_id": f"{subject_id}_{suffix}_{take + 1}",
                        "subject_id": subject_id,
                        "label": label,
                        "y": LABEL_TO_INT[label],
                        "attack_type": attack_type,
                        "device": "synthetic",
                        "lighting": "synthetic",
                        "n_valid_frames": int(len(frames)),
                        "median_iod": float(np.median(ratios["iod"])),
                        **g_feats,
                        **base,
                    }
                )

    columns = [
        "video_id", "subject_id", "label", "y", "attack_type", "device", "lighting",
        "n_valid_frames", "median_iod", "G1", "G2", "G3", *BASELINE_FEATURE_NAMES,
    ]
    return pd.DataFrame(rows, columns=columns)


def experiment_classification(
    config: Config, n_subjects: int = 8, sigma_grid: Sequence[float] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Прогнать полный LOSO-пайплайн на синтетическом наборе.

    Returns:
        (метрики без шума, метрики по сетке sigma).
    """
    from src.evaluation import run_loso

    if sigma_grid is None:
        sigma_grid = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

    clean = build_synthetic_dataset(n_subjects=n_subjects, seed=config.seed)
    predictions, _, pooled, _ = run_loso(clean, config, save=False, bootstrap=False)
    pooled = pooled.copy()
    pooled.insert(0, "sigma", 0.0)

    noise_rows = []
    for sigma in sigma_grid:
        for repeat in range(3):
            data = build_synthetic_dataset(
                n_subjects=n_subjects, sigma=float(sigma), seed=config.seed + repeat * 97
            )
            _, _, pooled_s, _ = run_loso(data, config, save=False, bootstrap=False)
            for _, row in pooled_s.iterrows():
                noise_rows.append(
                    {
                        "sigma": float(sigma),
                        "repeat": repeat,
                        "model": row["model"],
                        "apcer": row["apcer"],
                        "bpcer": row["bpcer"],
                        "acer": row["acer"],
                        "accuracy": row["accuracy"],
                    }
                )
    return pooled, pd.DataFrame(noise_rows), predictions


def run_synthetic_study(config: Config, n_subjects: int = 8) -> dict[str, pd.DataFrame]:
    """Выполнить все эксперименты и сохранить таблицы в ``results/synthetic/``."""
    logger = get_logger()
    out_dir = ensure_dir(config.path("results_dir") / "synthetic")

    logger.info("E1: зависимость признаков от рельефа лица")
    depth = experiment_depth_sweep()
    depth.to_csv(out_dir / "synthetic_depth_sweep.csv", index=False)

    logger.info("E2: зависимость от расстояния до камеры")
    distance = experiment_distance_sweep()
    distance.to_csv(out_dir / "synthetic_distance_sweep.csv", index=False)

    logger.info("E3: зависимость от типа движения головы")
    motion = experiment_motion_types()
    motion.to_csv(out_dir / "synthetic_motion_types.csv", index=False)

    logger.info("E4: классификация полным пайплайном (%d участников)", n_subjects)
    pooled, noise, predictions = experiment_classification(config, n_subjects=n_subjects)
    pooled.to_csv(out_dir / "synthetic_loso_metrics.csv", index=False)
    noise.to_csv(out_dir / "synthetic_noise_sweep.csv", index=False)
    predictions.to_csv(out_dir / "synthetic_predictions.csv", index=False)

    logger.info("Таблицы сохранены в %s", out_dir)
    return {
        "depth": depth,
        "distance": distance,
        "motion": motion,
        "pooled": pooled,
        "noise": noise,
        "predictions": predictions,
    }
