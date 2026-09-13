"""Общие утилиты: конфигурация, логирование, seed, работа с metadata.csv."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.constants import (
    METADATA_COLUMNS,
    VALID_ATTACK_TYPES,
    VALID_LABELS,
    VIDEO_EXTENSIONS,
)

LOGGER_NAME = "facepad"


# --------------------------------------------------------------------------
# Логирование
# --------------------------------------------------------------------------
def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Настроить единый логгер проекта (идемпотентно)."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)-7s %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    """Вернуть логгер проекта."""
    return setup_logging()


# --------------------------------------------------------------------------
# Конфигурация
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    """Обёртка над YAML-конфигурацией с разрешением путей от корня проекта."""

    raw: dict[str, Any]
    project_root: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def path(self, key: str) -> Path:
        """Абсолютный путь для записи `paths.<key>` конфигурации."""
        paths = self.raw.get("paths", {})
        if key not in paths:
            raise KeyError(
                f"В конфигурации нет paths.{key}. Доступные ключи: {sorted(paths)}"
            )
        return (self.project_root / paths[key]).resolve()

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))


def project_root() -> Path:
    """Корень репозитория (родитель каталога `src/`)."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path) -> Config:
    """Загрузить YAML-конфигурацию.

    Raises:
        FileNotFoundError: если файла конфигурации нет — с подсказкой.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = (project_root() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Файл конфигурации не найден: {path}\n"
            "Ожидался, например, configs/prototype.yaml. "
            "Запускайте из корня проекта: "
            "python scripts/run_pipeline.py --config configs/prototype.yaml"
        )
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config(raw=raw, project_root=project_root())


def set_seed(seed: int) -> None:
    """Зафиксировать единственный seed для numpy и random."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> Path:
    """Создать каталог (вместе с родителями) и вернуть его."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------
# metadata.csv
# --------------------------------------------------------------------------
class MetadataError(ValueError):
    """Ошибка структуры или содержимого metadata.csv."""


def load_metadata(metadata_csv: Path, raw_dir: Path) -> pd.DataFrame:
    """Прочитать и провалидировать metadata.csv.

    Возвращает DataFrame только с существующими на диске видео и
    дополнительной колонкой ``abs_path``.

    Raises:
        MetadataError: при отсутствии обязательных колонок, недопустимых
            метках/типах атак или дублирующихся ``video_id``.
    """
    logger = get_logger()
    if not metadata_csv.exists():
        raise MetadataError(
            f"Файл метаданных не найден: {metadata_csv}\n"
            "Создайте его по шаблону data/metadata.csv со столбцами: "
            + ",".join(METADATA_COLUMNS)
        )

    # comment='#' позволяет держать закомментированные примеры строк в шаблоне.
    df = pd.read_csv(metadata_csv, comment="#", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in METADATA_COLUMNS if c not in df.columns]
    if missing:
        raise MetadataError(
            f"В {metadata_csv} отсутствуют обязательные колонки: {missing}\n"
            f"Требуется заголовок: {','.join(METADATA_COLUMNS)}"
        )

    df = df[list(METADATA_COLUMNS)].copy()
    for col in METADATA_COLUMNS:
        df[col] = df[col].astype(str).str.strip()

    # Пустые строки шаблона отбрасываем.
    df = df[df["video_id"] != ""].reset_index(drop=True)
    if df.empty:
        logger.warning(
            "metadata.csv не содержит ни одной строки с данными "
            "(это нормально, пока вы не добавили видео)."
        )
        df["abs_path"] = pd.Series(dtype=str)
        return df

    dup = df["video_id"][df["video_id"].duplicated()].unique().tolist()
    if dup:
        raise MetadataError(f"Дублирующиеся video_id в metadata.csv: {dup}")

    bad_labels = sorted(set(df["label"]) - set(VALID_LABELS))
    if bad_labels:
        raise MetadataError(
            f"Недопустимые значения label: {bad_labels}. Разрешено: {list(VALID_LABELS)}"
        )

    bad_attacks = sorted(set(df["attack_type"]) - set(VALID_ATTACK_TYPES))
    if bad_attacks:
        raise MetadataError(
            f"Недопустимые значения attack_type: {bad_attacks}. "
            f"Разрешено: {list(VALID_ATTACK_TYPES)}"
        )

    inconsistent = df[(df["label"] == "live") & (df["attack_type"] != "none")]
    if len(inconsistent):
        raise MetadataError(
            "Для label=live требуется attack_type=none. Проблемные video_id: "
            f"{inconsistent['video_id'].tolist()}"
        )
    inconsistent = df[(df["label"] == "attack") & (df["attack_type"] == "none")]
    if len(inconsistent):
        raise MetadataError(
            "Для label=attack требуется конкретный attack_type "
            f"({[a for a in VALID_ATTACK_TYPES if a != 'none']}). Проблемные video_id: "
            f"{inconsistent['video_id'].tolist()}"
        )

    df["abs_path"] = df["relative_path"].apply(lambda p: str((raw_dir / p).resolve()))
    exists = df["abs_path"].apply(lambda p: Path(p).exists())
    if (~exists).any():
        missing_files = df.loc[~exists, "relative_path"].tolist()
        logger.warning(
            "Пропущено %d строк: файлы не найдены в %s -> %s",
            len(missing_files),
            raw_dir,
            missing_files[:10],
        )
    df = df[exists].reset_index(drop=True)

    bad_ext = df[~df["relative_path"].str.lower().str.endswith(VIDEO_EXTENSIONS)]
    if len(bad_ext):
        logger.warning(
            "Файлы с неожиданным расширением (будет попытка чтения OpenCV): %s",
            bad_ext["relative_path"].tolist()[:10],
        )
    return df


def describe_dataset(df: pd.DataFrame) -> str:
    """Краткая текстовая сводка по набору данных."""
    if df.empty:
        return "Набор данных пуст."
    parts = [
        f"видео: {len(df)}",
        f"субъектов: {df['subject_id'].nunique()}",
        f"live: {(df['label'] == 'live').sum()}",
        f"attack: {(df['label'] == 'attack').sum()}",
    ]
    by_attack = df[df["label"] == "attack"]["attack_type"].value_counts().to_dict()
    if by_attack:
        parts.append(f"типы атак: {by_attack}")
    return ", ".join(parts)


def no_data_message(raw_dir: Path, metadata_csv: Path) -> str:
    """Понятное сообщение о том, куда класть данные (метрики НЕ выдумываются)."""
    return (
        "\n" + "=" * 72 + "\n"
        "НЕТ ДАННЫХ ДЛЯ АНАЛИЗА — метрики не вычисляются и не выдумываются.\n"
        + "=" * 72 + "\n"
        f"1) Положите видео (2-3 с) в: {raw_dir}\n"
        "   Структура: data/raw/subject_001/live/, .../print/, .../screen/\n"
        f"2) Опишите каждое видео строкой в: {metadata_csv}\n"
        "   Колонки: " + ",".join(METADATA_COLUMNS) + "\n"
        "   label: live | attack;  attack_type: none | print | screen | replay | curved_print\n"
        "3) Повторите запуск:\n"
        "   python scripts/run_pipeline.py --config configs/prototype.yaml\n"
        + "=" * 72
    )
