#!/usr/bin/env python3
"""Разложить видео из одной папки по структуре data/raw/.

    python scripts/import_videos.py --source ~/Desktop/видео --dry-run
    python scripts/import_videos.py --source ~/Desktop/видео

Имя файла должно содержать участника и тип записи, разделённые подчёркиванием:

    s001_live_01.mov      -> data/raw/s001/live/s001_live_01.mov
    s001_screen_02.mp4    -> data/raw/s001/screen/s001_screen_02.mp4
    asiya_print_1.mov     -> data/raw/asiya/print/asiya_print_1.mov

Номер дубля необязателен. Регистр и язык раскладки значения не имеют, но тип
записи должен быть одним из: live, print, screen, replay, curved_print.

Если все файлы в папке принадлежат ОДНОМУ участнику, его можно задать флагом
``--subject``, и тогда в имени достаточно указать только тип записи:

    python scripts/import_videos.py --source ~/Desktop/даулет --subject s001

По умолчанию файлы КОПИРУЮТСЯ, а не переносятся: записи участников
невосстановимы, и оригиналы должны остаться на месте до тех пор, пока вы не
убедитесь, что всё разложено верно. Для переноса используйте ``--move``.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import VIDEO_EXTENSIONS
from src.utils import FOLDER_TO_LABEL, ensure_dir, get_logger, load_config, setup_logging

#: Понятные синонимы типов записи, которые встречаются в именах файлов.
CLASS_ALIASES = {
    "live": "live", "лайв": "live", "жив": "live", "real": "live",
    "print": "print", "принт": "print", "печать": "print", "paper": "print",
    "screen": "screen", "скрин": "screen", "экран": "screen", "phone": "screen",
    "replay": "replay", "реплей": "replay",
    "curved": "curved_print", "curved_print": "curved_print",
}


def detect_class(name: str) -> str | None:
    """Определить тип записи по имени файла."""
    lowered = name.lower()
    for alias, folder in CLASS_ALIASES.items():
        if re.search(rf"(^|[^a-zа-я]){re.escape(alias)}([^a-zа-я]|$)", lowered):
            return folder
    return None


def detect_subject(name: str, forced: str | None) -> str | None:
    """Определить участника: явно заданного или первую часть имени файла."""
    if forced:
        return forced
    head = re.split(r"[_\-\s]", Path(name).stem, maxsplit=1)[0].strip()
    return head or None


def main() -> int:
    parser = argparse.ArgumentParser(description="Раскладка видео по data/raw/")
    parser.add_argument("--source", required=True, help="папка, куда вы сложили все записи")
    parser.add_argument("--config", default="configs/prototype.yaml")
    parser.add_argument("--subject", default=None,
                        help="участник для ВСЕЙ папки, если в именах его нет")
    parser.add_argument("--move", action="store_true",
                        help="перенести файлы вместо копирования (по умолчанию копия)")
    parser.add_argument("--dry-run", action="store_true",
                        help="показать план, ничего не трогая")
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config(args.config)
    raw_dir = config.path("raw_dir")

    source = Path(args.source).expanduser()
    if not source.exists():
        logger.error("Папка не найдена: %s", source)
        return 2
    if not source.is_dir():
        logger.error("Это не папка: %s", source)
        return 2

    videos = sorted(
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS and not p.name.startswith(".")
    )
    if not videos:
        logger.error("В %s нет видеофайлов (%s)", source, ", ".join(VIDEO_EXTENSIONS))
        return 1

    plan: list[tuple[Path, Path]] = []
    problems: list[str] = []
    for video in videos:
        video_class = detect_class(video.name)
        subject = detect_subject(video.name, args.subject)
        if video_class is None:
            problems.append(
                f"{video.name}: не понятен тип записи. Допишите в имя одно из: "
                f"{', '.join(sorted(FOLDER_TO_LABEL))}"
            )
            continue
        if subject is None:
            problems.append(f"{video.name}: не понятен участник. Задайте --subject")
            continue
        plan.append((video, raw_dir / subject / video_class / video.name))

    print(f"\nНайдено видеофайлов: {len(videos)}   распознано: {len(plan)}\n")
    for src, dst in plan:
        print(f"  {src.name}  ->  {dst.relative_to(raw_dir.parent.parent)}")

    if problems:
        print(f"\nНе распознано ({len(problems)}):")
        for problem in problems:
            print(f"  {problem}")

    if not plan:
        print("\nРаскладывать нечего. Переименуйте файлы и запустите снова.")
        return 1

    subjects = sorted({dst.parent.parent.name for _, dst in plan})
    classes = sorted({dst.parent.name for _, dst in plan})
    print(f"\nУчастников: {len(subjects)} ({', '.join(subjects)})")
    print(f"Типы записей: {', '.join(classes)}")
    if len(subjects) < 2:
        print("\nВНИМАНИЕ: участник только один — LOSO-CV требует минимум двух.")
    if "live" not in classes:
        print("\nВНИМАНИЕ: нет записей класса live.")
    if not ({"print", "screen", "replay", "curved_print"} & set(classes)):
        print("\nВНИМАНИЕ: нет ни одной записи атаки — метрики PAD посчитать нельзя.")

    if args.dry_run:
        print("\n--dry-run: ничего не скопировано.")
        return 0

    existing = [dst for _, dst in plan if dst.exists()]
    if existing:
        print(f"\nОШИБКА: {len(existing)} файлов уже существуют, например {existing[0].name}.")
        print("Удалите их или переименуйте исходные — перезаписывать записи опасно.")
        return 3

    for src, dst in plan:
        ensure_dir(dst.parent)
        if args.move:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(src, dst)
    action = "перенесено" if args.move else "скопировано"
    print(f"\nГотово: {action} файлов — {len(plan)}")
    print("\nДальше:")
    print("  python scripts/make_metadata.py --device \"<камера>\" --lighting \"<свет>\"")
    print("  python scripts/run_pipeline.py --config configs/prototype.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
