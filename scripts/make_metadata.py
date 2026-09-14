#!/usr/bin/env python3
"""Собрать data/metadata.csv автоматически из структуры каталогов data/raw/.

    python scripts/make_metadata.py --config configs/prototype.yaml

Скрипт обходит ``data/raw/<участник>/<тип>/`` и составляет таблицу: тип записи
берётся из имени подпапки (live, print, screen, replay, curved_print), участник —
из имени папки верхнего уровня.

Уже заполненные вручную поля (``device``, ``lighting``, ``notes``) СОХРАНЯЮТСЯ:
строки сопоставляются по ``relative_path``, поэтому повторный запуск после
досъёмки не затирает комментарии к прежним записям.

Сначала посмотрите, что получится, ничего не записывая:

    python scripts/make_metadata.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.constants import METADATA_COLUMNS
from src.utils import get_logger, load_config, scan_raw_videos, setup_logging


def load_existing(metadata_csv: Path) -> dict[str, dict[str, str]]:
    """Прочитать заполненные вручную поля прошлой версии таблицы."""
    if not metadata_csv.exists():
        return {}
    try:
        df = pd.read_csv(metadata_csv, comment="#", dtype=str, keep_default_na=False)
    except Exception:
        return {}
    if "relative_path" not in df.columns:
        return {}
    return {
        str(row["relative_path"]): {
            field: str(row.get(field, "")) for field in ("device", "lighting", "notes")
        }
        for _, row in df.iterrows()
        if str(row.get("relative_path", "")).strip()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Автосборка metadata.csv из data/raw/")
    parser.add_argument("--config", default="configs/prototype.yaml")
    parser.add_argument("--device", default="", help="значение по умолчанию для колонки device")
    parser.add_argument("--lighting", default="", help="значение по умолчанию для колонки lighting")
    parser.add_argument("--dry-run", action="store_true",
                        help="показать таблицу, ничего не записывая")
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config(args.config)
    raw_dir, metadata_csv = config.path("raw_dir"), config.path("metadata_csv")

    records, warnings = scan_raw_videos(raw_dir)
    for warning in warnings:
        logger.warning("%s", warning)

    if not records:
        print(
            f"\nВидео не найдены в {raw_dir}\n"
            "Ожидаемая структура каталогов:\n"
            "    data/raw/subject_001/live/live_01.mp4\n"
            "    data/raw/subject_001/print/print_01.mp4\n"
            "    data/raw/subject_001/screen/screen_01.mp4\n"
        )
        return 1

    existing = load_existing(metadata_csv)
    rows = []
    kept = 0
    for record in records:
        previous = existing.get(record["relative_path"], {})
        if previous.get("device") or previous.get("lighting") or previous.get("notes"):
            kept += 1
        rows.append(
            {
                **record,
                "device": previous.get("device") or args.device,
                "lighting": previous.get("lighting") or args.lighting,
                "notes": previous.get("notes", ""),
            }
        )

    df = pd.DataFrame(rows, columns=list(METADATA_COLUMNS))

    print(f"\nНайдено видео: {len(df)} | участников: {df['subject_id'].nunique()}")
    print(df.groupby(["label", "attack_type"]).size().rename("записей").to_string())
    if kept:
        print(f"\nСохранены заполненные вручную поля для {kept} записей.")
    print("\n" + df[["video_id", "relative_path", "subject_id", "label", "attack_type"]]
          .to_string(index=False))

    if df["subject_id"].nunique() < 2:
        logger.warning(
            "Участник только один — LOSO-CV требует минимум двух. "
            "Пайплайн построит график с точками и признаки, но не метрики."
        )

    if args.dry_run:
        print(f"\n--dry-run: файл {metadata_csv} НЕ изменён.")
        return 0

    df.to_csv(metadata_csv, index=False)
    print(f"\nЗаписано: {metadata_csv}")
    print("Заполните колонки device, lighting и notes — и запускайте:")
    print("    python scripts/run_pipeline.py --config configs/prototype.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
