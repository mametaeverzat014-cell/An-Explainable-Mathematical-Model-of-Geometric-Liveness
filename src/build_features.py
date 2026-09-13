"""Расчёт геометрических признаков: IOD, r1/r2/r3, G1/G2/G3 и признаков базлайна.

Математическая модель (см. protocol.md):

    d2D(a, b) = sqrt((x_a - x_b)^2 + (y_a - y_b)^2)
    IOD(t)    = d2D(eye_left(t), eye_right(t))

    r1(t) = [z_nose(t) - (z_cheek_left(t) + z_cheek_right(t))/2] / [IOD(t) + eps]
    r2(t) = d2D(nose(t), eye_left(t)) / [d2D(nose(t), eye_right(t)) + eps]
    r3(t) = d2D(nose(t), chin(t)) / [d2D(cheek_left(t), cheek_right(t)) + eps]

    G1 = Var_t[r1(t)],  G2 = Var_t[r2(t)],  G3 = Var_t[r3(t)]

Координата z — ОЦЕНОЧНАЯ ОТНОСИТЕЛЬНАЯ ПСЕВДОГЛУБИНА MediaPipe,
а не физически измеренная глубина.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.constants import DEFAULT_EPSILON, LABEL_TO_INT
from src.quality_control import valid_frame_mask
from src.utils import Config, ensure_dir, get_logger

#: Имена покадровых геометрических отношений.
RATIO_NAMES = ("r1", "r2", "r3")
#: Имена временных признаков (дисперсий отношений).
G_NAMES = ("G1", "G2", "G3")
#: Признаки статического однокадрового 2D-базлайна B1.
BASELINE_FEATURE_NAMES = (
    "b_nose_eye_left",      # d2D(nose, eye_left) / IOD
    "b_nose_eye_right",     # d2D(nose, eye_right) / IOD
    "b_nose_chin_ratio",    # d2D(nose, chin) / cheek_width
    "b_face_wh_ratio",      # cheek_width / face_height
    "b_sharpness",          # дисперсия лапласиана (опционально, иначе NaN)
)


def d2d(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Евклидово расстояние в плоскости изображения между наборами точек (N, 2)."""
    diff = p - q
    return np.sqrt((diff * diff).sum(axis=-1))


def _xy(df: pd.DataFrame, name: str) -> np.ndarray:
    return np.column_stack(
        [df[f"{name}_x"].to_numpy(dtype=np.float64), df[f"{name}_y"].to_numpy(dtype=np.float64)]
    )


def compute_ratios(df: pd.DataFrame, epsilon: float = DEFAULT_EPSILON) -> pd.DataFrame:
    """Вычислить покадровые r1(t), r2(t), r3(t) и IOD(t).

    Args:
        df: таблица landmark'ов (только валидные кадры).
        epsilon: защита от деления на ноль.

    Returns:
        DataFrame с колонками frame_idx, iod, r1, r2, r3.
    """
    nose, eye_l, eye_r = _xy(df, "nose"), _xy(df, "eye_left"), _xy(df, "eye_right")
    cheek_l, cheek_r, chin = _xy(df, "cheek_left"), _xy(df, "cheek_right"), _xy(df, "chin")

    iod = d2d(eye_l, eye_r)
    z_nose = df["nose_z"].to_numpy(dtype=np.float64)
    z_cheek_mean = 0.5 * (
        df["cheek_left_z"].to_numpy(dtype=np.float64)
        + df["cheek_right_z"].to_numpy(dtype=np.float64)
    )

    r1 = (z_nose - z_cheek_mean) / (iod + epsilon)
    r2 = d2d(nose, eye_l) / (d2d(nose, eye_r) + epsilon)
    r3 = d2d(nose, chin) / (d2d(cheek_l, cheek_r) + epsilon)

    return pd.DataFrame(
        {
            "frame_idx": df["frame_idx"].to_numpy() if "frame_idx" in df else np.arange(len(df)),
            "iod": iod,
            "r1": r1,
            "r2": r2,
            "r3": r3,
        }
    )


def compute_g_features(ratios: pd.DataFrame) -> dict[str, float]:
    """Временные дисперсии G1, G2, G3 по всем валидным кадрам видео."""
    return {
        "G1": float(np.var(ratios["r1"].to_numpy(dtype=np.float64))),
        "G2": float(np.var(ratios["r2"].to_numpy(dtype=np.float64))),
        "G3": float(np.var(ratios["r3"].to_numpy(dtype=np.float64))),
    }


def compute_baseline_features(
    frame: pd.Series, epsilon: float = DEFAULT_EPSILON
) -> dict[str, float]:
    """Интерпретируемые 2D-признаки ОДНОГО кадра (для статического базлайна B1)."""
    def pt(name: str) -> np.ndarray:
        return np.array([float(frame[f"{name}_x"]), float(frame[f"{name}_y"])])

    nose, eye_l, eye_r = pt("nose"), pt("eye_left"), pt("eye_right")
    cheek_l, cheek_r, chin = pt("cheek_left"), pt("cheek_right"), pt("chin")

    iod = float(np.linalg.norm(eye_l - eye_r))
    cheek_width = float(np.linalg.norm(cheek_l - cheek_r))
    eye_mid = 0.5 * (eye_l + eye_r)
    face_height = float(np.linalg.norm(eye_mid - chin))

    return {
        "b_nose_eye_left": float(np.linalg.norm(nose - eye_l)) / (iod + epsilon),
        "b_nose_eye_right": float(np.linalg.norm(nose - eye_r)) / (iod + epsilon),
        "b_nose_chin_ratio": float(np.linalg.norm(nose - chin)) / (cheek_width + epsilon),
        "b_face_wh_ratio": cheek_width / (face_height + epsilon),
    }


def laplacian_variance(video_path: Path, frame_index: int) -> float:
    """Дисперсия лапласиана выбранного кадра — простой прокси резкости.

    Возвращает NaN, если кадр прочитать не удалось (признак не блокирует пайплайн).
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover
        return float("nan")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return float("nan")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            return float("nan")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:  # pragma: no cover - устойчивость к экзотическим кодекам
        return float("nan")
    finally:
        cap.release()


def build_video_features(
    config: Config,
    metadata: pd.DataFrame,
    qc_report: pd.DataFrame,
    extraction_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Собрать таблицу признаков уровня видео (одна строка = одно видео).

    Обрабатываются только видео, прошедшие контроль качества.

    Returns:
        DataFrame: video_id, subject_id, label, y (1=live), attack_type,
        n_valid_frames, G1..G3 и признаки базлайна b_*.
    """
    logger = get_logger()
    epsilon = float(config.get("features", {}).get("epsilon", DEFAULT_EPSILON))
    max_abs_coord = float(config.get("quality_control", {}).get("max_abs_coord", 5.0))
    use_sharpness = bool(config.get("baseline", {}).get("use_laplacian_sharpness", True))
    results_dir = ensure_dir(config.path("results_dir"))

    passed = set(qc_report.loc[qc_report["passed"], "video_id"]) if len(qc_report) else set()
    csv_paths = (
        extraction_summary.set_index("video_id")["landmark_csv"].to_dict()
        if not extraction_summary.empty
        else {}
    )

    rows: list[dict[str, Any]] = []
    for _, meta_row in metadata.iterrows():
        video_id = meta_row["video_id"]
        if video_id not in passed:
            continue
        landmark_df = pd.read_csv(csv_paths[video_id])
        mask = valid_frame_mask(landmark_df, max_abs_coord)
        valid = landmark_df[mask].reset_index(drop=True)

        ratios = compute_ratios(valid, epsilon)
        g_feats = compute_g_features(ratios)

        # Статический базлайн: ОДИН средний валидный кадр.
        mid = len(valid) // 2
        base_feats = compute_baseline_features(valid.iloc[mid], epsilon)
        base_feats["b_sharpness"] = (
            laplacian_variance(Path(meta_row["abs_path"]), int(valid.iloc[mid]["frame_idx"]))
            if use_sharpness
            else float("nan")
        )

        rows.append(
            {
                "video_id": video_id,
                "subject_id": meta_row["subject_id"],
                "label": meta_row["label"],
                "y": LABEL_TO_INT[meta_row["label"]],
                "attack_type": meta_row["attack_type"],
                "device": meta_row["device"],
                "lighting": meta_row["lighting"],
                "n_valid_frames": int(len(valid)),
                "median_iod": float(np.median(ratios["iod"].to_numpy())),
                **g_feats,
                **base_feats,
            }
        )

    columns = [
        "video_id", "subject_id", "label", "y", "attack_type", "device", "lighting",
        "n_valid_frames", "median_iod", *G_NAMES, *BASELINE_FEATURE_NAMES,
    ]
    features = pd.DataFrame(rows, columns=columns)
    out_path = results_dir / "video_features.csv"
    features.to_csv(out_path, index=False)
    logger.info("Признаки уровня видео: %d строк -> %s", len(features), out_path)
    return features


def save_ratio_timeseries(
    ratios: pd.DataFrame, landmarks_dir: Path, video_id: str
) -> Path:
    """Сохранить покадровые r1/r2/r3 (полезно для отладки и графиков)."""
    out = ensure_dir(landmarks_dir / "ratios") / f"{video_id}_ratios.csv"
    ratios.to_csv(out, index=False)
    return out
