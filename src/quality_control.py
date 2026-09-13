"""Контроль качества извлечённых landmark'ов.

Отбраковываются видео, для которых геометрические признаки нельзя посчитать
надёжно. Каждое решение сопровождается человекочитаемой причиной и
сохраняется в ``results/landmark_quality_report.csv``.

Проверки:
    1. no_landmark_file  — CSV с landmark'ами отсутствует;
    2. extraction_error  — ошибка чтения видео/MediaPipe;
    3. no_face_detected  — лицо не найдено ни на одном кадре;
    4. too_few_valid_frames / low_valid_ratio — мало валидных кадров;
    5. invalid_landmark_values — NaN/inf или аномальные координаты;
    6. degenerate_iod    — межзрачковое расстояние близко к нулю.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.constants import LANDMARK_NAMES
from src.utils import Config, ensure_dir, get_logger

#: Колонки отчёта контроля качества.
QC_COLUMNS = (
    "video_id",
    "subject_id",
    "label",
    "attack_type",
    "n_frames",
    "n_valid_frames",
    "valid_ratio",
    "median_iod",
    "passed",
    "reason",
)


def _coord_columns() -> list[str]:
    return [f"{n}_{a}" for n in LANDMARK_NAMES for a in ("x", "y", "z")]


def valid_frame_mask(df: pd.DataFrame, max_abs_coord: float) -> np.ndarray:
    """Маска кадров с найденным лицом и конечными координатами в пределах нормы."""
    if df.empty:
        return np.zeros(0, dtype=bool)
    cols = _coord_columns()
    values = df[cols].to_numpy(dtype=np.float64)
    finite = np.isfinite(values).all(axis=1)
    in_range = (np.abs(values) <= max_abs_coord).all(axis=1, where=np.isfinite(values))
    found = df["face_found"].to_numpy() == 1
    return found & finite & in_range


def interocular_distance(df: pd.DataFrame) -> np.ndarray:
    """IOD(t) = d2D(eye_left, eye_right) — масштаб лица в каждом кадре."""
    dx = df["eye_left_x"].to_numpy(dtype=np.float64) - df["eye_right_x"].to_numpy(dtype=np.float64)
    dy = df["eye_left_y"].to_numpy(dtype=np.float64) - df["eye_right_y"].to_numpy(dtype=np.float64)
    return np.sqrt(dx * dx + dy * dy)


def assess_video(
    landmark_df: pd.DataFrame,
    qc_cfg: dict[str, Any],
    extraction_error: str = "",
) -> dict[str, Any]:
    """Оценить качество одного видео.

    Returns:
        Словарь с полями отчёта (без метаданных субъекта).
    """
    min_valid_frames = int(qc_cfg.get("min_valid_frames", 10))
    min_valid_ratio = float(qc_cfg.get("min_valid_ratio", 0.5))
    min_iod = float(qc_cfg.get("min_iod_pixels_norm", 0.02))
    max_abs_coord = float(qc_cfg.get("max_abs_coord", 5.0))

    out: dict[str, Any] = {
        "n_frames": int(len(landmark_df)),
        "n_valid_frames": 0,
        "valid_ratio": 0.0,
        "median_iod": np.nan,
        "passed": False,
        "reason": "",
    }

    if extraction_error:
        out["reason"] = f"extraction_error: {extraction_error}"
        return out
    if landmark_df.empty:
        out["reason"] = "no_landmark_file: нет ни одного прочитанного кадра"
        return out

    face_found = int((landmark_df["face_found"] == 1).sum())
    if face_found == 0:
        out["reason"] = (
            "no_face_detected: MediaPipe не нашёл лицо ни на одном кадре "
            "(проверьте освещение, расстояние до камеры, кадрирование)"
        )
        return out

    mask = valid_frame_mask(landmark_df, max_abs_coord)
    n_valid = int(mask.sum())
    out["n_valid_frames"] = n_valid
    out["valid_ratio"] = float(n_valid / len(landmark_df))

    if n_valid < face_found:
        # часть кадров отброшена из-за NaN/inf/аномалий — это не фатально,
        # но фиксируется в причине, если видео в итоге не проходит.
        pass

    if n_valid == 0:
        out["reason"] = "invalid_landmark_values: все кадры содержат NaN/inf/аномальные координаты"
        return out

    iod = interocular_distance(landmark_df[mask])
    out["median_iod"] = float(np.median(iod))

    if n_valid < min_valid_frames:
        out["reason"] = (
            f"too_few_valid_frames: {n_valid} < min_valid_frames={min_valid_frames} "
            "(снимите видео длиннее или улучшите условия съёмки)"
        )
        return out
    if out["valid_ratio"] < min_valid_ratio:
        out["reason"] = (
            f"low_valid_ratio: {out['valid_ratio']:.2f} < min_valid_ratio={min_valid_ratio} "
            "(лицо теряется на значительной части кадров)"
        )
        return out
    if out["median_iod"] < min_iod:
        out["reason"] = (
            f"degenerate_iod: median_iod={out['median_iod']:.4f} < {min_iod} "
            "(лицо слишком мелкое в кадре — подойдите ближе к камере)"
        )
        return out

    out["passed"] = True
    out["reason"] = "ok"
    return out


def run_quality_control(
    config: Config, metadata: pd.DataFrame, extraction_summary: pd.DataFrame
) -> pd.DataFrame:
    """Построить отчёт контроля качества и сохранить его в results/.

    Returns:
        DataFrame со схемой :data:`QC_COLUMNS`.
    """
    logger = get_logger()
    qc_cfg = config.get("quality_control", {})
    results_dir = ensure_dir(config.path("results_dir"))
    out_path = results_dir / "landmark_quality_report.csv"

    errors = (
        extraction_summary.set_index("video_id")["extraction_error"].to_dict()
        if not extraction_summary.empty
        else {}
    )
    csv_paths = (
        extraction_summary.set_index("video_id")["landmark_csv"].to_dict()
        if not extraction_summary.empty
        else {}
    )

    rows: list[dict[str, Any]] = []
    for _, meta_row in metadata.iterrows():
        video_id = meta_row["video_id"]
        csv_path = csv_paths.get(video_id)
        if csv_path and Path(csv_path).exists():
            landmark_df = pd.read_csv(csv_path)
        else:
            landmark_df = pd.DataFrame()
        assessment = assess_video(landmark_df, qc_cfg, errors.get(video_id, ""))
        rows.append(
            {
                "video_id": video_id,
                "subject_id": meta_row["subject_id"],
                "label": meta_row["label"],
                "attack_type": meta_row["attack_type"],
                **assessment,
            }
        )

    report = pd.DataFrame(rows, columns=list(QC_COLUMNS))
    report.to_csv(out_path, index=False)
    n_pass = int(report["passed"].sum()) if len(report) else 0
    logger.info(
        "Контроль качества: принято %d из %d видео -> %s", n_pass, len(report), out_path
    )
    if len(report) and n_pass < len(report):
        for _, r in report[~report["passed"]].iterrows():
            logger.warning("  отбраковано [%s]: %s", r["video_id"], r["reason"])
    return report
