#!/usr/bin/env python3
"""Извлечь из live-видео кадр, пригодный как фотография для атаки.

    python scripts/extract_attack_photo.py --config configs/prototype.yaml

Для каждой записи класса `live` выбирается лучший кадр и сохраняется в
`results/attack_photos/<subject_id>.png`. Эту картинку затем показывают на
экране телефона (атака `screen`) или печатают (атака `print`).

Почему это корректно. Злоумышленник в реальности располагает готовой
фотографией жертвы, и кадр из её же видео — честная модель такой фотографии.
Более того, так совпадают освещение, камера и сам человек, поэтому между
классами остаётся ровно одно различие — плоскость вместо рельефа. Это именно
то, что должна измерять модель.

Кадр выбирается по двум критериям: лицо максимально фронтально (отношение
расстояний нос-глаз близко к единице) и кадр максимально резкий (дисперсия
лапласиана). Фронтальность важна, потому что с повёрнутого фото атака
выглядит неестественно и добавляет в эксперимент лишнюю переменную.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.build_features import compute_ratios
from src.extract_landmarks import LandmarkBackendError, create_backend, iter_video_frames
from src.quality_control import valid_frame_mask
from src.utils import ensure_dir, get_logger, load_config, load_metadata, setup_logging
from src.constants import LANDMARK_CSV_COLUMNS, LANDMARK_INDICES, LANDMARK_NAMES

import pandas as pd


def score_frames(video_path: Path, backend, max_frames: int) -> tuple[int, np.ndarray] | None:
    """Найти самый фронтальный и резкий кадр с лицом.

    Returns:
        (индекс кадра, кадр в формате RGB) либо None, если лицо не найдено.
    """
    import cv2

    rows: list[dict[str, float]] = []
    frames: dict[int, np.ndarray] = {}
    for frame_idx, rgb in iter_video_frames(video_path, max_frames, 1):
        coords = backend.process(rgb, int(1000 * frame_idx / 30.0))
        if coords is None or coords.shape[0] <= max(LANDMARK_INDICES.values()):
            continue
        row: dict[str, float] = {"video_id": "photo", "frame_idx": frame_idx, "face_found": 1}
        for name, idx in LANDMARK_INDICES.items():
            row[f"{name}_x"] = float(coords[idx, 0])
            row[f"{name}_y"] = float(coords[idx, 1])
            row[f"{name}_z"] = float(coords[idx, 2])
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        row["sharpness"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        rows.append(row)
        frames[frame_idx] = rgb

    if not rows:
        return None

    df = pd.DataFrame(rows)
    valid = df[valid_frame_mask(df, 5.0)].reset_index(drop=True)
    if valid.empty:
        return None

    ratios = compute_ratios(valid)
    frontality = np.abs(ratios["r2"].to_numpy() - 1.0)     # 0 = идеально фронтально
    sharpness = valid["sharpness"].to_numpy()

    # нормируем обе величины и складываем: резкость максимизируем, отклонение
    # от фронтальности минимизируем
    sharp_norm = (sharpness - sharpness.min()) / (np.ptp(sharpness) + 1e-9)
    front_norm = frontality / (frontality.max() + 1e-9)
    best = int(np.argmax(sharp_norm - 1.5 * front_norm))

    frame_idx = int(valid.loc[best, "frame_idx"])
    return frame_idx, frames[frame_idx]


def main() -> int:
    parser = argparse.ArgumentParser(description="Кадр из live-видео для атаки")
    parser.add_argument("--config", default="configs/prototype.yaml")
    parser.add_argument("--max-frames", type=int, default=120)
    args = parser.parse_args()

    import cv2

    logger = setup_logging()
    config = load_config(args.config)
    metadata = load_metadata(config.path("metadata_csv"), config.path("raw_dir"))

    live = metadata[metadata["label"] == "live"]
    if live.empty:
        logger.error("В metadata.csv нет ни одной записи класса live.")
        return 1

    out_dir = ensure_dir(config.path("results_dir") / "attack_photos")
    backend = None
    made = 0
    try:
        for subject_id, group in live.groupby("subject_id"):
            row = group.iloc[0]
            if backend is None:
                backend = create_backend(config.get("extraction", {}))
                logger.info("Бэкенд MediaPipe: %s", backend.name)
            result = score_frames(Path(row["abs_path"]), backend, args.max_frames)
            if result is None:
                logger.warning("[%s] лицо не найдено, фото не извлечено", subject_id)
                continue
            frame_idx, rgb = result
            out_path = out_dir / f"{subject_id}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            logger.info("[%s] кадр %d -> %s", subject_id, frame_idx, out_path)
            made += 1
    except LandmarkBackendError as exc:
        logger.error("MediaPipe недоступен:\n%s", exc)
        return 3
    finally:
        if backend is not None:
            backend.close()

    print(f"\nГотово: {made} фото в {out_dir}")
    print("Дальше: откройте фото на телефоне во весь экран и снимите атаку screen,")
    print("повторяя то же движение, что участник делал головой.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
