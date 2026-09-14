"""Тесты автосборки metadata.csv из структуры каталогов data/raw/."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import scan_raw_videos  # noqa: E402


def _make_video(root: Path, relative: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_scan_maps_folders_to_labels(tmp_path: Path) -> None:
    """Тип записи определяется именем подпапки."""
    _make_video(tmp_path, "subject_001/live/a.mp4")
    _make_video(tmp_path, "subject_001/print/b.mp4")
    _make_video(tmp_path, "subject_001/screen/c.mp4")

    records, warnings = scan_raw_videos(tmp_path)
    by_id = {r["video_id"]: r for r in records}

    assert len(records) == 3
    assert not warnings
    assert by_id["subject_001_live_01"]["label"] == "live"
    assert by_id["subject_001_live_01"]["attack_type"] == "none"
    assert by_id["subject_001_print_01"]["label"] == "attack"
    assert by_id["subject_001_print_01"]["attack_type"] == "print"
    assert by_id["subject_001_screen_01"]["attack_type"] == "screen"


def test_scan_numbers_takes_per_class(tmp_path: Path) -> None:
    """Дубли внутри одного класса нумеруются по порядку."""
    _make_video(tmp_path, "subject_002/live/first.mp4")
    _make_video(tmp_path, "subject_002/live/second.mov")

    records, _ = scan_raw_videos(tmp_path)
    assert [r["video_id"] for r in records] == [
        "subject_002_live_01",
        "subject_002_live_02",
    ]
    assert records[1]["relative_path"] == "subject_002/live/second.mov"


def test_scan_warns_about_unknown_folder(tmp_path: Path) -> None:
    """Папка с неизвестным типом записи пропускается с предупреждением."""
    _make_video(tmp_path, "subject_003/mistake/x.mp4")
    records, warnings = scan_raw_videos(tmp_path)
    assert records == []
    assert any("неизвестный тип записи" in w for w in warnings)


def test_scan_skips_non_video_files(tmp_path: Path) -> None:
    """Посторонние файлы не попадают в таблицу."""
    _make_video(tmp_path, "subject_004/live/ok.mp4")
    _make_video(tmp_path, "subject_004/live/notes.txt")
    records, warnings = scan_raw_videos(tmp_path)
    assert len(records) == 1
    assert any("не видеофайл" in w for w in warnings)


def test_scan_reports_empty_directory(tmp_path: Path) -> None:
    """Пустой каталог — это предупреждение, а не молчаливый пустой результат."""
    records, warnings = scan_raw_videos(tmp_path)
    assert records == []
    assert warnings


def test_scan_handles_missing_directory(tmp_path: Path) -> None:
    """Отсутствующий каталог не приводит к исключению."""
    records, warnings = scan_raw_videos(tmp_path / "нет-такого")
    assert records == []
    assert any("не найден" in w for w in warnings)


# --------------------------------------------------------------------------
# Защита от аварийного завершения mediapipe 1.x под macOS
# --------------------------------------------------------------------------
import unittest.mock as _mock  # noqa: E402

import pytest  # noqa: E402

from src.extract_landmarks import (  # noqa: E402
    ALLOW_MACOS_TASKS_ENV,
    LandmarkBackendError,
    check_macos_tasks_bug,
)


def test_macos_with_mediapipe_1x_is_refused_before_it_aborts() -> None:
    """Сочетание должно дать понятное сообщение, а не падение процесса."""
    with _mock.patch("sys.platform", "darwin"):
        with pytest.raises(LandmarkBackendError, match="python.org"):
            check_macos_tasks_bug("1.0.1")


def test_macos_with_legacy_branch_is_allowed() -> None:
    """Ветка 0.10.x под macOS работает и запрещать её нельзя."""
    with _mock.patch("sys.platform", "darwin"):
        check_macos_tasks_bug("0.10.21")


def test_other_platforms_are_not_affected() -> None:
    """На Linux и Windows этот путь кода исправен."""
    for platform in ("linux", "win32"):
        with _mock.patch("sys.platform", platform):
            check_macos_tasks_bug("1.0.1")


def test_check_can_be_disabled_for_a_future_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Когда ошибку в mediapipe исправят, запрет должен сниматься."""
    monkeypatch.setenv(ALLOW_MACOS_TASKS_ENV, "1")
    with _mock.patch("sys.platform", "darwin"):
        check_macos_tasks_bug("1.0.1")
