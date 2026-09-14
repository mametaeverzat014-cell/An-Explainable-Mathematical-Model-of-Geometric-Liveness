"""Извлечение семантических landmark'ов лица из видео (MediaPipe FaceMesh).

Для каждого видео создаётся CSV `results/landmarks/<video_id>.csv` со строкой
на кадр: координаты x, y (нормированные к размеру кадра) и z.

ВАЖНО О КООРДИНАТЕ z:
    MediaPipe возвращает *оценочную относительную псевдоглубину*, а не
    физически измеренную глубину. Значение z выражено примерно в тех же
    единицах, что и x (ширина кадра), и отсчитывается относительно плоскости
    головы. Мы НИКОГДА не называем его "реальной глубиной".

Модуль можно запускать самостоятельно:
    python -m src.extract_landmarks --config configs/prototype.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from src.constants import LANDMARK_CSV_COLUMNS, LANDMARK_INDICES, LANDMARK_NAMES
from src.utils import Config, ensure_dir, get_logger, load_config, load_metadata

#: Путь к файлу модели Tasks API (нужен только для mediapipe>=1.0).
TASK_MODEL_ENV = "FACE_LANDMARKER_TASK"


class LandmarkBackendError(RuntimeError):
    """MediaPipe недоступен или сконфигурирован неверно."""


# --------------------------------------------------------------------------
# Бэкенды MediaPipe
# --------------------------------------------------------------------------
class _LegacyFaceMeshBackend:
    """Бэкенд `mediapipe.solutions.face_mesh` (mediapipe 0.10.x).

    Предпочтительный вариант: модель встроена в пакет, сеть не нужна.
    """

    name = "solutions.face_mesh"

    def __init__(self, cfg: dict[str, Any]) -> None:
        import mediapipe as mp  # локальный импорт: тяжёлая зависимость

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=bool(cfg.get("static_image_mode", False)),
            max_num_faces=int(cfg.get("max_num_faces", 1)),
            refine_landmarks=bool(cfg.get("refine_landmarks", True)),
            min_detection_confidence=float(cfg.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(cfg.get("min_tracking_confidence", 0.5)),
        )

    def process(self, rgb_frame: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        """Вернуть массив (N, 3) нормированных координат или None."""
        result = self._mesh.process(rgb_frame)
        if not result.multi_face_landmarks:
            return None
        face = result.multi_face_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in face.landmark], dtype=np.float64)

    def close(self) -> None:
        self._mesh.close()


class _TasksFaceLandmarkerBackend:
    """Бэкенд `mediapipe.tasks` (mediapipe >= 1.0).

    Требует ЛОКАЛЬНЫЙ файл модели `face_landmarker.task`; путь задаётся
    переменной окружения ``FACE_LANDMARKER_TASK``. Проект никогда не
    скачивает модель сам.
    """

    name = "tasks.FaceLandmarker"

    def __init__(self, cfg: dict[str, Any]) -> None:
        import os

        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        model_path = os.environ.get(TASK_MODEL_ENV, "")
        if not model_path or not Path(model_path).exists():
            raise LandmarkBackendError(
                "Установлена версия mediapipe>=1.0, в которой удалён "
                "solutions.face_mesh, а Tasks API требует локальный файл модели.\n"
                "Решение 1 (рекомендуется): pip install 'mediapipe>=0.10,<1.0'\n"
                f"Решение 2: скачать face_landmarker.task вручную и указать путь в "
                f"переменной окружения {TASK_MODEL_ENV}."
            )
        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=int(cfg.get("max_num_faces", 1)),
            min_face_detection_confidence=float(cfg.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(cfg.get("min_tracking_confidence", 0.5)),
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def process(self, rgb_frame: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return None
        face = result.face_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in face], dtype=np.float64)

    def close(self) -> None:
        self._landmarker.close()


def create_backend(cfg: dict[str, Any]):
    """Выбрать доступный бэкенд MediaPipe с понятным сообщением об ошибке."""
    try:
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        import sys as _sys

        version_hint = ""
        if _sys.version_info >= (3, 13):
            version_hint = (
                f"\nВЕРОЯТНАЯ ПРИЧИНА: у вас Python "
                f"{_sys.version_info.major}.{_sys.version_info.minor}, "
                "а mediapipe собран только для Python 3.9-3.12.\n"
                "Создайте окружение на Python 3.12, например:\n"
                "    brew install python@3.12\n"
                "    /opt/homebrew/bin/python3.12 -m venv .venv\n"
                "    source .venv/bin/activate && pip install -r requirements.txt"
            )
        raise LandmarkBackendError(
            "Пакет mediapipe не установлен.\n"
            "Установите зависимости: pip install -r requirements.txt\n"
            "(нужна ветка 'mediapipe>=0.10,<1.0' — в ней модель встроена)"
            + version_hint
        ) from exc

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        return _LegacyFaceMeshBackend(cfg)
    return _TasksFaceLandmarkerBackend(cfg)


# --------------------------------------------------------------------------
# Чтение видео
# --------------------------------------------------------------------------
def iter_video_frames(
    video_path: Path, max_frames: int, stride: int
) -> Iterator[tuple[int, np.ndarray]]:
    """Итератор по кадрам видео в формате RGB.

    Yields:
        (индекс кадра, RGB-кадр).

    Raises:
        FileNotFoundError: файл отсутствует.
        RuntimeError: OpenCV не смог открыть файл (кодек/повреждение).
    """
    import cv2

    if not video_path.exists():
        raise FileNotFoundError(f"Видеофайл не найден: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV не смог открыть видео: {video_path}. "
            "Проверьте кодек/целостность файла (рекомендуется MP4/H.264)."
        )
    try:
        emitted, idx = 0, 0
        while emitted < max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if idx % max(1, stride) == 0:
                yield idx, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                emitted += 1
            idx += 1
    finally:
        cap.release()


def extract_video_landmarks(
    video_path: Path, video_id: str, backend, max_frames: int, stride: int, fps_hint: float = 30.0
) -> pd.DataFrame:
    """Извлечь семантические landmark'и из одного видео.

    Returns:
        DataFrame со схемой :data:`LANDMARK_CSV_COLUMNS`; кадры без лица
        сохраняются со ``face_found=0`` и NaN-координатами.
    """
    rows: list[dict[str, Any]] = []
    for frame_idx, rgb in iter_video_frames(video_path, max_frames, stride):
        timestamp_ms = int(1000.0 * frame_idx / max(fps_hint, 1.0))
        coords = backend.process(rgb, timestamp_ms)
        row: dict[str, Any] = {
            "video_id": video_id,
            "frame_idx": frame_idx,
            "face_found": 0,
        }
        for name in LANDMARK_NAMES:
            row[f"{name}_x"] = np.nan
            row[f"{name}_y"] = np.nan
            row[f"{name}_z"] = np.nan
        if coords is not None:
            n_points = coords.shape[0]
            needed = max(LANDMARK_INDICES.values())
            if n_points > needed:
                row["face_found"] = 1
                for name, idx in LANDMARK_INDICES.items():
                    row[f"{name}_x"] = float(coords[idx, 0])
                    row[f"{name}_y"] = float(coords[idx, 1])
                    row[f"{name}_z"] = float(coords[idx, 2])
        rows.append(row)
    return pd.DataFrame(rows, columns=list(LANDMARK_CSV_COLUMNS))


def landmark_csv_path(landmarks_dir: Path, video_id: str) -> Path:
    """Путь к per-frame CSV для видео."""
    return landmarks_dir / f"{video_id}.csv"


def run_extraction(config: Config, metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    """Извлечь landmark'и для всех видео из metadata.csv.

    Returns:
        DataFrame с колонками video_id, landmark_csv, n_frames, n_valid_frames,
        extraction_error.
    """
    logger = get_logger()
    ex_cfg = config.get("extraction", {})
    landmarks_dir = ensure_dir(config.path("landmarks_dir"))

    if metadata is None:
        metadata = load_metadata(config.path("metadata_csv"), config.path("raw_dir"))
    if metadata.empty:
        logger.warning("Нет видео для извлечения landmark'ов.")
        return pd.DataFrame(
            columns=["video_id", "landmark_csv", "n_frames", "n_valid_frames", "extraction_error"]
        )

    overwrite = bool(ex_cfg.get("overwrite", False))
    max_frames = int(ex_cfg.get("max_frames", 120))
    stride = int(ex_cfg.get("frame_stride", 1))

    # Бэкенд создаётся лениво: если все CSV уже посчитаны, MediaPipe не нужен.
    backend = None
    records: list[dict[str, Any]] = []
    try:
        for _, row in metadata.iterrows():
            video_id = row["video_id"]
            out_csv = landmark_csv_path(landmarks_dir, video_id)
            error = ""
            if out_csv.exists() and not overwrite:
                logger.info("[%s] landmark-CSV уже существует, пропуск", video_id)
                df = pd.read_csv(out_csv)
            else:
                try:
                    if backend is None:
                        backend = create_backend(ex_cfg)
                        logger.info("Бэкенд MediaPipe: %s", backend.name)
                    df = extract_video_landmarks(
                        Path(row["abs_path"]), video_id, backend, max_frames, stride
                    )
                    df.to_csv(out_csv, index=False)
                    logger.info(
                        "[%s] кадров: %d, с лицом: %d",
                        video_id,
                        len(df),
                        int(df["face_found"].sum()),
                    )
                except (FileNotFoundError, RuntimeError, LandmarkBackendError) as exc:
                    if isinstance(exc, LandmarkBackendError):
                        raise
                    error = f"{type(exc).__name__}: {exc}"
                    logger.error("[%s] ошибка извлечения: %s", video_id, error)
                    df = pd.DataFrame(columns=list(LANDMARK_CSV_COLUMNS))
            records.append(
                {
                    "video_id": video_id,
                    "landmark_csv": str(out_csv),
                    "n_frames": int(len(df)),
                    "n_valid_frames": int(df["face_found"].sum()) if len(df) else 0,
                    "extraction_error": error,
                }
            )
    finally:
        if backend is not None:
            backend.close()
    return pd.DataFrame(records)


def main() -> None:
    """CLI-точка входа модуля."""
    parser = argparse.ArgumentParser(description="Извлечение landmark'ов из видео")
    parser.add_argument("--config", default="configs/prototype.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    summary = run_extraction(config)
    get_logger().info("Обработано видео: %d", len(summary))


if __name__ == "__main__":
    main()
