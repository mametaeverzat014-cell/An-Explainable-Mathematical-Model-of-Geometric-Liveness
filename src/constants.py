"""Центральный файл констант: индексы landmark'ов MediaPipe и схемы таблиц.

ЕДИНСТВЕННОЕ место, где заданы номера точек MediaPipe FaceMesh (468/478 точек).
Чтобы использовать другие точки — меняйте значения только здесь.

Нумерация соответствует канонической модели MediaPipe Face Mesh.
Проверить/изменить точки можно по визуализации `figures/landmark_example.png`,
которую строит пайплайн.
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------
# Семантические точки лица (индексы MediaPipe FaceMesh)
# --------------------------------------------------------------------------
# 1   — кончик носа (nose tip)
# 33  — внешний угол ПРАВОГО глаза наблюдателя = левый глаз человека на изображении.
#       В коде используется система "как на картинке": EYE_LEFT — глаз слева на кадре.
# 263 — внешний угол противоположного глаза (справа на кадре).
# 234 — крайняя точка контура щеки слева на кадре.
# 454 — крайняя точка контура щеки справа на кадре.
# 152 — нижняя точка подбородка (chin).
NOSE: Final[int] = 1
EYE_LEFT: Final[int] = 33
EYE_RIGHT: Final[int] = 263
CHEEK_LEFT: Final[int] = 234
CHEEK_RIGHT: Final[int] = 454
CHIN: Final[int] = 152

#: Отображение "семантическое имя -> индекс MediaPipe".
LANDMARK_INDICES: Final[dict[str, int]] = {
    "nose": NOSE,
    "eye_left": EYE_LEFT,
    "eye_right": EYE_RIGHT,
    "cheek_left": CHEEK_LEFT,
    "cheek_right": CHEEK_RIGHT,
    "chin": CHIN,
}

#: Порядок семантических точек во всех таблицах (фиксирован!).
LANDMARK_NAMES: Final[tuple[str, ...]] = tuple(LANDMARK_INDICES.keys())

# --------------------------------------------------------------------------
# Схемы таблиц
# --------------------------------------------------------------------------
#: Обязательные колонки metadata.csv.
METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "video_id",
    "relative_path",
    "subject_id",
    "label",
    "attack_type",
    "device",
    "lighting",
    "notes",
)

#: Допустимые метки класса (бинарная задача PAD).
VALID_LABELS: Final[tuple[str, ...]] = ("live", "attack")

#: Допустимые типы атак.
VALID_ATTACK_TYPES: Final[tuple[str, ...]] = (
    "none",
    "print",
    "screen",
    "replay",
    "curved_print",
)

#: Колонки per-frame CSV с landmark'ами.
LANDMARK_CSV_COLUMNS: Final[tuple[str, ...]] = (
    ("video_id", "frame_idx", "face_found")
    + tuple(f"{name}_{axis}" for name in LANDMARK_NAMES for axis in ("x", "y", "z"))
)

#: Числовое кодирование меток: 1 = live (bona fide), 0 = attack.
LABEL_TO_INT: Final[dict[str, int]] = {"live": 1, "attack": 0}
INT_TO_LABEL: Final[dict[int, str]] = {1: "live", 0: "attack"}

#: Расширения видеофайлов, которые ищет пайплайн в data/raw/.
VIDEO_EXTENSIONS: Final[tuple[str, ...]] = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")

#: Epsilon из математической модели (см. protocol.md).
DEFAULT_EPSILON: Final[float] = 1e-8
