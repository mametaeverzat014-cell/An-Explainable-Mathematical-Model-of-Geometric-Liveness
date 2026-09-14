"""Синтетические тесты геометрии: плоская vs непланарная траектория лица.

Идея проверки: смоделировать движение головы перед камерой и сравнить
временные признаки G1, G2, G3 для двух объектов:

* НЕПЛАНАРНЫЙ («живое» лицо): кончик носа вынесен вперёд относительно щёк,
  поэтому при движении возникает параллакс;
* ПЛАНАРНЫЙ (плоская атака: распечатка или экран): все точки лежат в одной
  плоскости, параллакса нет.

Тесты не требуют ни видео, ни MediaPipe и выполняются за доли секунды.

ВАЖНЫЙ РЕЗУЛЬТАТ, зафиксированный этими тестами (см. research_log.md):
разные компоненты модели чувствительны к РАЗНЫМ движениям головы.
    * G1 (псевдоглубина носа) реагирует на НАКЛОН головы (pitch, кивок);
      при чистом повороте (yaw) множитель cos(theta) сокращается между
      числителем r1 и нормировкой на IOD, и сигнал почти исчезает.
    * G2 (лево-правая асимметрия) реагирует на ПОВОРОТ (yaw) и тождественно
      постоянна при чистом наклоне.
Поэтому протокол съёмки требует комбинированного движения (поворот + кивок).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.build_features import compute_g_features, compute_ratios, d2d  # noqa: E402
from src.constants import LANDMARK_NAMES  # noqa: E402
from src.synthetic_face import (  # noqa: E402
    CAMERA_DISTANCE,
    LIVE_NOSE_DEPTH as _LIVE_NOSE_DEPTH,
    canonical_face as _canonical_face,
    head_motion as _head_motion,
    render_replay_sequence as _render_replay_sequence,
    render_sequence as _render_sequence,
)


# --------------------------------------------------------------------------
# Базовая геометрия
# --------------------------------------------------------------------------
def test_d2d_matches_manual_distance() -> None:
    a = np.array([[0.0, 0.0], [1.0, 1.0]])
    b = np.array([[3.0, 4.0], [1.0, 1.0]])
    assert d2d(a, b) == pytest.approx([5.0, 0.0])


def test_r2_is_symmetric_at_frontal_pose() -> None:
    """При фронтальном ракурсе отношение расстояний нос-глаз равно 1."""
    frontal = _render_sequence(0.12, np.array([0.0]))
    assert float(compute_ratios(frontal)["r2"].iloc[0]) == pytest.approx(1.0, rel=1e-6)


def test_ratios_are_scale_invariant() -> None:
    """Нормировка на IOD делает r1 и r3 инвариантными к размеру лица в кадре."""
    yaw, pitch = _head_motion(30)
    base = _render_sequence(0.12, yaw, pitch)
    scaled = base.copy()
    for name in LANDMARK_NAMES:                        # лицо вдвое крупнее в кадре
        for axis in ("x", "y", "z"):
            scaled[f"{name}_{axis}"] *= 2.0
    r_base, r_scaled = compute_ratios(base), compute_ratios(scaled)
    assert np.allclose(r_base["r1"], r_scaled["r1"], atol=1e-9)
    assert np.allclose(r_base["r3"], r_scaled["r3"], atol=1e-9)


def test_static_pose_gives_zero_variances() -> None:
    """Без движения все временные дисперсии равны нулю (параллакса нет)."""
    still = np.zeros(30)
    g = compute_g_features(compute_ratios(_render_sequence(0.12, still, still)))
    assert g["G1"] == pytest.approx(0.0, abs=1e-12)
    assert g["G2"] == pytest.approx(0.0, abs=1e-12)
    assert g["G3"] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# Плоская поверхность vs рельеф
# --------------------------------------------------------------------------
def test_planar_face_has_exactly_zero_g1_under_pitch() -> None:
    """Для плоского объекта при наклоне r1(t) тождественно равно нулю.

    Нос лежит в той же плоскости, что и щёки, поэтому z_nose(t) совпадает со
    средней псевдоглубиной щёк и G1 = 0 точно. Это «идеальный» случай,
    ради которого и вводится признак G1.
    """
    n = 40
    g = compute_g_features(
        compute_ratios(_render_sequence(0.0, np.zeros(n), np.linspace(-15, 15, n)))
    )
    assert g["G1"] == pytest.approx(0.0, abs=1e-15)


def test_planar_face_still_produces_some_g1_under_yaw() -> None:
    """ОГРАНИЧЕНИЕ: плоская атака тоже даёт ненулевой G1 при повороте.

    При повороте плоскости одна щека оказывается ближе к камере, чем другая,
    поэтому их псевдоглубины в координатах изображения масштабируются
    по-разному и средняя псевдоглубина щёк перестаёт быть нулевой.
    Следовательно, G1 — статистический признак, а НЕ строгий признак
    «плоскости»: модель не может гарантировать обнаружение любой атаки.
    """
    n = 40
    g = compute_g_features(
        compute_ratios(_render_sequence(0.0, np.linspace(-25, 25, n), np.zeros(n)))
    )
    assert g["G1"] > 0.0


def test_non_planar_face_has_larger_g1_than_planar() -> None:
    """Ключевая проверка модели: рельеф лица даёт заметную дисперсию G1."""
    yaw, pitch = _head_motion()
    g_planar = compute_g_features(compute_ratios(_render_sequence(0.0, yaw, pitch)))
    g_relief = compute_g_features(compute_ratios(_render_sequence(0.12, yaw, pitch)))
    assert g_relief["G1"] > 2.0 * g_planar["G1"]
    assert g_relief["G1"] > 1e-5


def test_non_planar_face_has_larger_g2_than_planar() -> None:
    """Лево-правая асимметрия при повороте выражена сильнее у рельефного лица."""
    yaw, pitch = _head_motion()
    g_planar = compute_g_features(compute_ratios(_render_sequence(0.0, yaw, pitch)))
    g_relief = compute_g_features(compute_ratios(_render_sequence(0.12, yaw, pitch)))
    assert g_relief["G2"] > 10 * g_planar["G2"]


def test_g1_grows_with_nose_depth() -> None:
    """Монотонность: чем сильнее рельеф, тем больше G1."""
    yaw, pitch = _head_motion()
    values = [
        compute_g_features(compute_ratios(_render_sequence(depth, yaw, pitch)))["G1"]
        for depth in (0.0, 0.05, 0.10, 0.20)
    ]
    assert all(values[i] < values[i + 1] for i in range(len(values) - 1))


# --------------------------------------------------------------------------
# Задокументированные ограничения модели
# --------------------------------------------------------------------------
def test_g1_separation_requires_pitch_not_only_yaw() -> None:
    """Ограничение: при ЧИСТОМ повороте головы G1 почти не разделяет классы.

    В r1 числитель пропорционален cos(yaw), и нормировка на IOD, тоже
    пропорциональная cos(yaw), почти полностью его сокращает. В результате
    отношение G1(рельеф)/G1(плоскость) при чистом повороте близко к 1, а при
    наклоне (кивке) разделение становится практически идеальным.
    Поэтому протокол съёмки требует КОМБИНИРОВАННОГО движения.
    """
    n = 40
    yaw, pitch, still = np.linspace(-25, 25, n), np.linspace(-15, 15, n), np.zeros(n)

    def g1(depth: float, y_: np.ndarray, p_: np.ndarray) -> float:
        return compute_g_features(compute_ratios(_render_sequence(depth, y_, p_)))["G1"]

    yaw_separation = g1(0.12, yaw, still) / g1(0.0, yaw, still)
    assert yaw_separation < 2.0                    # поворот почти не разделяет классы
    assert g1(0.0, still, pitch) == pytest.approx(0.0, abs=1e-15)
    assert g1(0.12, still, pitch) > 1e-6           # наклон разделяет идеально


def test_g2_requires_yaw_not_only_pitch() -> None:
    """Симметричное ограничение: G2 реагирует на поворот, но не на наклон."""
    n = 40
    yaw_only = compute_g_features(
        compute_ratios(_render_sequence(0.12, np.linspace(-25, 25, n), np.zeros(n)))
    )["G2"]
    pitch_only = compute_g_features(
        compute_ratios(_render_sequence(0.12, np.zeros(n), np.linspace(-15, 15, n)))
    )["G2"]
    assert pitch_only == pytest.approx(0.0, abs=1e-12)
    assert yaw_only > 1e-3


def test_orthographic_projection_destroys_parallax() -> None:
    """Ограничение: без перспективы сигнал параллакса исчезает.

    При очень большом расстоянии до камеры проекция становится практически
    ортографической и G1 стремится к нулю даже для рельефного лица. Отсюда
    требование протокола: лицо должно занимать заметную часть кадра.
    """
    yaw, pitch = _head_motion()
    g_near = compute_g_features(compute_ratios(_render_sequence(0.12, yaw, pitch, 2.0)))
    g_far = compute_g_features(compute_ratios(_render_sequence(0.12, yaw, pitch, 1000.0)))
    assert g_far["G1"] < 0.25 * g_near["G1"]


# --------------------------------------------------------------------------
# Сквозная проверка PLS
# --------------------------------------------------------------------------
def test_pls_separates_synthetic_planar_from_non_planar() -> None:
    """Сквозная проверка: PLS с train-only нормализацией разделяет два класса.

    Тестовый субъект не участвует ни в нормализации, ни в выборе порога.
    """
    from src.pls_model import PLSModel
    from src.utils import load_config

    config = load_config("configs/prototype.yaml")
    yaw, pitch = _head_motion()
    rng = np.random.default_rng(0)

    rows = []
    for subject in range(4):
        for cls, depth in (("live", 0.12), ("attack", 0.0)):
            jitter = rng.normal(0, 0.5, size=yaw.shape)   # индивидуальная вариация движения
            df = _render_sequence(depth, yaw + jitter, pitch + jitter * 0.5)
            rows.append(
                {
                    "video_id": f"s{subject}_{cls}",
                    "subject_id": f"subject_{subject}",
                    "label": cls,
                    "y": 1 if cls == "live" else 0,
                    **compute_g_features(compute_ratios(df)),
                }
            )
    features = pd.DataFrame(rows)

    train = features[features["subject_id"] != "subject_0"]
    test = features[features["subject_id"] == "subject_0"].reset_index(drop=True)
    model = PLSModel(config).fit(train)
    assert list(model.predict(test)) == list(test["y"])


# --------------------------------------------------------------------------
# Атака воспроизведением (replay): видео живого лица на плоском экране
# --------------------------------------------------------------------------
def test_replay_on_still_screen_reproduces_r2_and_r3_exactly() -> None:
    """Неподвижный экран воспроизводит r2 и r3 ТОЧНО — это тождество.

    Неподвижный экран, параллельный плоскости изображения, отображает кадр
    записи гомотетией относительно центра: все попарные 2D-расстояния
    умножаются на один и тот же коэффициент. Признаки r2 и r3 суть отношения
    расстояний, поэтому коэффициент сокращается. Следовательно, атака
    воспроизведением не приближает эти признаки к живым, а **совпадает** с
    ними, и никакой порог их не разделит.
    """
    n = 40
    yaw, pitch = _head_motion(n, 25.0, 15.0)
    still = np.zeros(n)

    live = compute_ratios(_render_sequence(_LIVE_NOSE_DEPTH, yaw, pitch))
    replay = compute_ratios(
        _render_replay_sequence(yaw, pitch, still, still, nose_depth=_LIVE_NOSE_DEPTH)
    )

    assert np.allclose(live["r2"], replay["r2"], atol=1e-12)
    assert np.allclose(live["r3"], replay["r3"], atol=1e-12)


def test_replay_on_still_screen_has_exactly_zero_g1() -> None:
    """Экран плоский, поэтому псевдоглубина тождественно равна нулю."""
    n = 40
    yaw, pitch = _head_motion(n, 25.0, 15.0)
    still = np.zeros(n)
    ratios = compute_ratios(_render_replay_sequence(yaw, pitch, still, still))
    assert np.allclose(ratios["r1"], 0.0, atol=1e-12)
    assert compute_g_features(ratios)["G1"] == pytest.approx(0.0, abs=1e-20)


def test_replay_defeats_g2_unlike_static_photo_attack() -> None:
    """G2 у replay близок к живому, а у атаки статическим фото — на порядки меньше.

    Это и есть механизм, из-за которого модель проходима: G2 отличает
    поворот настоящей головы от наклона плоскости, но НЕ отличает поворот
    настоящей головы от её же записи, показанной на экране.
    """
    n = 40
    yaw, pitch = _head_motion(n, 25.0, 15.0)

    g_live = compute_g_features(compute_ratios(_render_sequence(_LIVE_NOSE_DEPTH, yaw, pitch)))
    g_photo = compute_g_features(compute_ratios(_render_sequence(0.0, yaw, pitch)))
    g_replay = compute_g_features(compute_ratios(_render_replay_sequence(yaw, pitch, yaw, pitch)))

    # Статическое фото на экране: G2 на порядки ниже живого — модель ловит атаку.
    assert g_photo["G2"] < g_live["G2"] / 10.0
    # Видео на экране: G2 того же порядка, что у живого лица — модель слепа.
    assert 0.5 < g_replay["G2"] / g_live["G2"] < 2.0


def test_replay_attack_passes_liveness_check_in_full_pipeline() -> None:
    """Сквозная проверка: обученная на live+screen модель пропускает replay.

    Тест фиксирует ОГРАНИЧЕНИЕ работы, а не её достижение. Если он однажды
    упадёт, значит модель научилась различать replay — и это следует
    исследовать, а не «чинить» тест.
    """
    from src.synthetic_study import experiment_replay_attack
    from src.utils import load_config

    config = load_config(Path(__file__).resolve().parent.parent / "configs" / "prototype.yaml")
    _, predictions = experiment_replay_attack(config, n_subjects=6)

    share_live = predictions.groupby("attack_type")["predicted_live"].mean()
    assert share_live["none"] > 0.9, "живые записи должны проходить проверку"
    assert share_live["screen"] < 0.3, "атака статическим фото должна отсеиваться"
    assert share_live["replay"] > 0.9, "модель не распознаёт атаку воспроизведением"
