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
        row: dict[str, float] = {
            "video_id": video_id,
            "frame_idx": frame_idx,
            "face_found": 1,
        }
        for name in LANDMARK_NAMES:
            x, y, z = face[name]
            u, v, w = project_point(x, y, z, theta, phi, camera_distance)
            row[f"{name}_x"] = u
            row[f"{name}_y"] = v
            row[f"{name}_z"] = w
        rows.append(row)
    return pd.DataFrame(rows)


def project_point(
    x: float,
    y: float,
    z: float,
    theta: float,
    phi: float,
    camera_distance: float,
) -> tuple[float, float, float]:
    """Повернуть точку и спроецировать её перспективно в координаты кадра.

    Единственное место, где записана геометрия сцены: и живое лицо, и плоский
    носитель атаки проецируются этой же функцией.

    Args:
        x, y, z: координаты точки в системе объекта.
        theta: угол поворота вокруг вертикальной оси (радианы).
        phi: угол наклона вокруг горизонтальной оси (радианы).
        camera_distance: расстояние от камеры до центра объекта.

    Returns:
        Тройка ``(x_img, y_img, z_img)``: координаты в кадре и оценочная
        относительная псевдоглубина в масштабе изображения.
    """
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    # Поворот вокруг вертикальной оси Y (yaw).
    x_rot, z_yaw = cos_t * x + sin_t * z, -sin_t * x + cos_t * z
    # Наклон вокруг горизонтальной оси X (pitch).
    y_rot, z_rot = cos_p * y + sin_p * z_yaw, -sin_p * y + cos_p * z_yaw
    # Перспективная проекция: положительное z_rot = ближе к камере.
    scale = FOCAL / (camera_distance - z_rot)
    return 0.5 + x_rot * scale, 0.5 + y_rot * scale, z_rot * scale


def render_replay_sequence(
    yaw_displayed: np.ndarray,
    pitch_displayed: np.ndarray,
    yaw_screen: np.ndarray,
    pitch_screen: np.ndarray,
    nose_depth: float = LIVE_NOSE_DEPTH,
    screen_scale: float = 2.0,
    record_distance: float = CAMERA_DISTANCE,
    camera_distance: float = CAMERA_DISTANCE,
    video_id: str = "synthetic_replay",
) -> pd.DataFrame:
    """Смоделировать атаку воспроизведением: видео живого лица на плоском экране.

    Сцена собирается в два шага, как и в действительности:

    1. **Запись.** Настоящая трёхмерная голова с рельефом ``nose_depth``
       поворачивается перед камерой и проецируется в кадр. Получается плоская
       картинка — записанное видео.
    2. **Предъявление.** Эта картинка выводится на экран, экран держат перед
       камерой и **двигают**, как при обычной плоской атаке.

    Ключевое отличие от атаки статической фотографией: изображённое лицо
    **поворачивается само**, поэтому лево-правая асимметрия, которую измеряет
    ``G2``, возникает без какого-либо рельефа у носителя. Псевдоглубина при
    этом остаётся плоской: все точки лежат в плоскости экрана.

    Args:
        yaw_displayed: поворот головы в записанном видео, градусы по кадрам.
        pitch_displayed: наклон головы в записанном видео, градусы по кадрам.
        yaw_screen: поворот самого экрана в руке атакующего, градусы по кадрам.
        pitch_screen: наклон самого экрана, градусы по кадрам.
        nose_depth: рельеф лица, которое было записано на видео.
        screen_scale: во сколько раз изображение на экране крупнее исходной
            проекции; компенсирует уменьшение при записи.
        record_distance: расстояние до камеры на этапе записи видео.
        camera_distance: расстояние от камеры до экрана при предъявлении.
        video_id: идентификатор для колонки ``video_id``.

    Returns:
        DataFrame со схемой landmark-CSV, как у :func:`render_sequence`.
    """
    face = canonical_face(nose_depth)

    rows: list[dict[str, float]] = []
    frames = zip(yaw_displayed, pitch_displayed, yaw_screen, pitch_screen)
    for frame_idx, (yaw_d, pitch_d, yaw_s, pitch_s) in enumerate(frames):
        theta_d, phi_d = np.deg2rad(yaw_d), np.deg2rad(pitch_d)
        theta_s, phi_s = np.deg2rad(yaw_s), np.deg2rad(pitch_s)

        row: dict[str, float] = {
            "video_id": video_id,
            "frame_idx": frame_idx,
            "face_found": 1,
        }
        for name in LANDMARK_NAMES:
            x, y, z = face[name]
            # Шаг 1: как точка выглядела в записанном видео.
            u, v, _ = project_point(x, y, z, theta_d, phi_d, record_distance)
            # Шаг 2: эта картинка лежит В ПЛОСКОСТИ экрана, рельефа у неё нет.
            x_screen = (u - 0.5) * screen_scale
            y_screen = (v - 0.5) * screen_scale
            u2, v2, w2 = project_point(
                x_screen, y_screen, 0.0, theta_s, phi_s, camera_distance
            )
            row[f"{name}_x"] = u2
            row[f"{name}_y"] = v2
            row[f"{name}_z"] = w2
        rows.append(row)
    return pd.DataFrame(rows)
