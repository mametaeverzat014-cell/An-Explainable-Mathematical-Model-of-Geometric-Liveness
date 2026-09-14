"""Синтетическая 3D-модель лица и её проекция в координаты landmark'ов.

Модуль позволяет проверять математику модели **без единого видео**: лицо
задаётся как набор точек в 3D, поворачивается перед виртуальной камерой и
проецируется в те же колонки, которые выдаёт MediaPipe.

Два свойства модели критичны и легко теряются при упрощении:

1. **Проекция перспективная.** Точки ближе к камере проецируются крупнее —
   именно это порождает параллакс. При ортографической проекции множитель
   ``cos(yaw)`` сокращается между числителем ``r1`` и нормировкой на ``IOD``,
   и сигнал исчезает полностью.
2. **Координата z выражена в масштабе изображения**, как у MediaPipe: чем
   дальше лицо, тем меньше ``|z|``. Если задать z в мировых единицах,
   дисперсия ``G1`` будет расти с расстоянием вместо того, чтобы убывать.

Это модель-идеализация: настоящая распечатка имеет текстуру и блики, а
MediaPipe вносит собственный шум. Выводы, полученные здесь, описывают
**механизм** модели, а не её точность на реальных данных.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import LANDMARK_NAMES

#: Расстояние от камеры до центра головы (условные единицы).
#: Задаёт силу перспективного эффекта: чем больше, тем слабее параллакс.
CAMERA_DISTANCE = 2.0

#: Фокусное расстояние виртуальной камеры.
FOCAL = 1.0

#: Рельеф «живого» лица по умолчанию — вынос кончика носа вперёд.
LIVE_NOSE_DEPTH = 0.12

#: Рельеф плоской атаки: все точки в одной плоскости.
PLANAR_NOSE_DEPTH = 0.0


def canonical_face(nose_depth: float) -> dict[str, np.ndarray]:
    """Каноническая 3D-модель лица (X, Y, Z) в условных единицах.

    Args:
        nose_depth: вынос кончика носа вперёд по оси Z.
            ``0.0`` соответствует идеально плоскому объекту — модели
            распечатки или экрана.

    Returns:
        Отображение "имя точки -> координаты (X, Y, Z)".
    """
    return {
        "nose": np.array([0.00, 0.00, nose_depth]),
        "eye_left": np.array([-0.15, -0.10, 0.0]),
        "eye_right": np.array([0.15, -0.10, 0.0]),
        "cheek_left": np.array([-0.25, 0.00, 0.0]),
        "cheek_right": np.array([0.25, 0.00, 0.0]),
        "chin": np.array([0.00, 0.30, 0.0]),
    }


def head_motion(
    n_frames: int = 40, yaw_amplitude: float = 25.0, pitch_amplitude: float = 15.0
) -> tuple[np.ndarray, np.ndarray]:
    """Движение головы, предписанное протоколом съёмки: поворот плюс кивок.

    Args:
        n_frames: число кадров.
        yaw_amplitude: амплитуда поворота в градусах (в каждую сторону).
        pitch_amplitude: амплитуда наклона в градусах.

    Returns:
        Пара массивов углов (поворот, наклон) по кадрам.
    """
    yaw = np.linspace(-yaw_amplitude, yaw_amplitude, n_frames)
    pitch = np.linspace(-pitch_amplitude, pitch_amplitude, n_frames)
    return yaw, pitch


def render_sequence(
    nose_depth: float,
    yaw_degrees: np.ndarray,
    pitch_degrees: np.ndarray | None = None,
    camera_distance: float = CAMERA_DISTANCE,
    video_id: str = "synthetic",
) -> pd.DataFrame:
    """Спроецировать движущееся лицо в таблицу landmark'ов.

    Args:
        nose_depth: рельеф лица (0.0 — плоская поверхность).
        yaw_degrees: углы поворота вокруг вертикальной оси по кадрам.
        pitch_degrees: углы наклона (кивка); по умолчанию нули.
        camera_distance: расстояние до камеры.
        video_id: идентификатор для колонки ``video_id``.

    Returns:
        DataFrame со схемой landmark-CSV: ``video_id``, ``frame_idx``,
        ``face_found`` и координаты ``*_x``, ``*_y``, ``*_z``.
        Колонки ``*_z`` содержат оценочную относительную псевдоглубину
        в масштабе изображения, а не физическую глубину.
    """
    if pitch_degrees is None:
        pitch_degrees = np.zeros_like(yaw_degrees)
    face = canonical_face(nose_depth)

    rows: list[dict[str, float]] = []
    for frame_idx, (yaw, pitch) in enumerate(zip(yaw_degrees, pitch_degrees)):
        theta, phi = np.deg2rad(yaw), np.deg2rad(pitch)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cos_p, sin_p = np.cos(phi), np.sin(phi)

        row: dict[str, float] = {
            "video_id": video_id,
            "frame_idx": frame_idx,
            "face_found": 1,
        }
        for name in LANDMARK_NAMES:
            x, y, z = face[name]
            # Поворот вокруг вертикальной оси Y (yaw).
            x_rot, z_yaw = cos_t * x + sin_t * z, -sin_t * x + cos_t * z
            # Наклон вокруг горизонтальной оси X (pitch).
            y_rot, z_rot = cos_p * y + sin_p * z_yaw, -sin_p * y + cos_p * z_yaw
            # Перспективная проекция: положительное z_rot = ближе к камере.
            scale = FOCAL / (camera_distance - z_rot)
            row[f"{name}_x"] = 0.5 + x_rot * scale
            row[f"{name}_y"] = 0.5 + y_rot * scale
            row[f"{name}_z"] = z_rot * scale
        rows.append(row)
    return pd.DataFrame(rows)
