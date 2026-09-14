"""Построение графиков (matplotlib, backend Agg — работает без дисплея)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # обязательно ДО импорта pyplot: сервер без дисплея

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.constants import LANDMARK_NAMES
from src.evaluation import confusion_counts, roc_auc, roc_curve_points
from src.utils import Config, ensure_dir, get_logger

MODEL_LABELS = {
    "B0_majority": "B0: мажоритарный",
    "B1_static2d": "B1: статический 2D (логрегрессия)",
    "M1_pls": "M1: PLS (временная геометрия)",
}


def _style(config: Config) -> tuple[int, tuple[float, float]]:
    try:
        import seaborn as sns

        sns.set_theme(style=str(config.get("plotting", {}).get("style", "whitegrid")))
    except ImportError:  # pragma: no cover - seaborn необязателен для работы
        pass
    plot_cfg = config.get("plotting", {})
    dpi = int(plot_cfg.get("dpi", 200))
    figsize = tuple(plot_cfg.get("figsize_default", [7.0, 5.0]))
    return dpi, figsize  # type: ignore[return-value]


def _save(fig: plt.Figure, path: Path, dpi: int) -> Path:
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    get_logger().info("График сохранён: %s", path)
    return path


def plot_landmark_example(
    config: Config, landmark_csv: Path, video_path: Path | None = None
) -> Path | None:
    """Иллюстрация используемых семантических точек на одном кадре."""
    dpi, figsize = _style(config)
    out = config.path("figures_dir") / "landmark_example.png"
    if not landmark_csv.exists():
        return None
    df = pd.read_csv(landmark_csv)
    df = df[df["face_found"] == 1]
    if df.empty:
        return None
    row = df.iloc[len(df) // 2]

    frame_img = None
    if video_path is not None and video_path.exists():
        try:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame_idx"]))
            ok, bgr = cap.read()
            cap.release()
            if ok:
                frame_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:  # pragma: no cover
            frame_img = None

    fig, ax = plt.subplots(figsize=figsize)
    if frame_img is not None:
        h, w = frame_img.shape[:2]
        ax.imshow(frame_img)
        scale_x, scale_y = w, h
    else:
        scale_x = scale_y = 1.0
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")
    for name in LANDMARK_NAMES:
        x, y = float(row[f"{name}_x"]) * scale_x, float(row[f"{name}_y"]) * scale_y
        ax.scatter([x], [y], s=60, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_title(
        f"Семантические точки лица (MediaPipe), видео: {row['video_id']}\n"
        "z — оценочная относительная псевдоглубина, не измеренная глубина"
    )
    ax.set_xlabel("x (нормированная координата кадра)")
    ax.set_ylabel("y (нормированная координата кадра)")
    return _save(fig, out, dpi)


def plot_score_distribution(config: Config, predictions: pd.DataFrame) -> Path | None:
    """Распределение out-of-fold скоров PLS для live и attack."""
    dpi, figsize = _style(config)
    out = config.path("figures_dir") / "pls_score_distribution.png"
    pls = predictions[predictions["model"] == "M1_pls"]
    if pls.empty:
        return None
    fig, ax = plt.subplots(figsize=figsize)
    for label, grp in pls.groupby("label"):
        ax.hist(grp["score"], bins=min(20, max(5, len(grp))), alpha=0.6, label=f"{label} (n={len(grp)})")
    for tau in sorted(pls["tau"].unique()):
        ax.axvline(tau, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Ориентированный скор PLS (больше => более вероятно live)")
    ax.set_ylabel("Количество видео")
    ax.set_title("Распределение PLS по классам (out-of-fold, LOSO-CV)\nпунктир — пороги tau по фолдам")
    ax.legend()
    return _save(fig, out, dpi)


def plot_roc(config: Config, predictions: pd.DataFrame) -> Path | None:
    """ROC-кривые всех моделей по объединённым out-of-fold предсказаниям."""
    dpi, figsize = _style(config)
    out = config.path("figures_dir") / "roc_curve.png"
    if predictions.empty:
        return None
    fig, ax = plt.subplots(figsize=figsize)
    plotted = False
    for model_name, grp in predictions.groupby("model"):
        if model_name == "B0_majority":
            continue  # константный скор — ROC не определён содержательно
        fpr, tpr = roc_curve_points(grp["y_true"].to_numpy(), grp["score"].to_numpy())
        if fpr.size == 0:
            continue
        auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
        ax.plot(fpr, tpr, marker="o", markersize=3, label=f"{MODEL_LABELS.get(model_name, model_name)} (AUC={auc:.3f})")
        plotted = True
    if not plotted:
        plt.close(fig)
        get_logger().warning("ROC не построен: в данных присутствует только один класс.")
        return None
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="случайное угадывание")
    ax.set_xlabel("FPR (доля атак, принятых за live)")
    ax.set_ylabel("TPR (доля верно принятых live)")
    ax.set_title("ROC-кривые (объединённые out-of-fold предсказания LOSO-CV)")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, out, dpi)


def plot_confusion(config: Config, predictions: pd.DataFrame, model_name: str, filename: str) -> Path | None:
    """Матрица ошибок одной модели."""
    dpi, _ = _style(config)
    out = config.path("figures_dir") / filename
    grp = predictions[predictions["model"] == model_name]
    if grp.empty:
        return None
    c = confusion_counts(grp["y_true"].to_numpy(), grp["y_pred"].to_numpy())
    matrix = np.array(
        [[c["live_as_live"], c["live_as_attack"]], [c["attack_as_live"], c["attack_as_attack"]]]
    )
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["предсказано live", "предсказано attack"])
    ax.set_yticks([0, 1], ["истинно live", "истинно attack"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black", fontsize=14)
    ax.set_title(f"Матрица ошибок — {MODEL_LABELS.get(model_name, model_name)}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _save(fig, out, dpi)


def plot_metric_comparison(config: Config, pooled_metrics: pd.DataFrame) -> Path | None:
    """Сравнение APCER / BPCER / ACER / Accuracy между моделями."""
    dpi, figsize = _style(config)
    out = config.path("figures_dir") / "metric_comparison.png"
    if pooled_metrics.empty:
        return None
    metrics = ["apcer", "bpcer", "acer", "accuracy"]
    models = pooled_metrics["model"].tolist()
    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=figsize)
    for i, model_name in enumerate(models):
        row = pooled_metrics[pooled_metrics["model"] == model_name].iloc[0]
        vals = [float(row[m]) if pd.notna(row[m]) else 0.0 for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=MODEL_LABELS.get(model_name, model_name))
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set_xticks(x + width * (len(models) - 1) / 2, [m.upper() for m in metrics])
    ax.set_ylabel("Значение метрики")
    ax.set_ylim(0, 1.12)
    ax.set_title("Сравнение моделей (pooled LOSO-CV, subject-disjoint)")
    ax.legend(fontsize=8)
    return _save(fig, out, dpi)


def plot_noise_robustness(config: Config, noise_df: pd.DataFrame) -> Path | None:
    """ACER в зависимости от sigma синтетического шума псевдоглубины."""
    dpi, figsize = _style(config)
    out = config.path("figures_dir") / "noise_robustness.png"
    if noise_df.empty:
        return None
    fig, ax = plt.subplots(figsize=figsize)
    for model_name, grp in noise_df.groupby("model"):
        agg = grp.groupby("sigma")["acer"].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg["sigma"], agg["mean"], yerr=agg["std"].fillna(0.0), marker="o",
                    capsize=3, label=MODEL_LABELS.get(model_name, model_name))
    ax.set_xlabel("sigma синтетического шума псевдоглубины z")
    ax.set_ylabel("ACER (меньше — лучше)")
    ax.set_title("Устойчивость к синтетическому шуму псевдоглубины\n"
                 "(модель шума: z_obs = z_true + eta, eta ~ N(0, sigma^2))")
    ax.legend(fontsize=8)
    return _save(fig, out, dpi)


def plot_ablation(config: Config, ablation_df: pd.DataFrame) -> Path | None:
    """Сравнение полного PLS и вариантов без отдельных компонент."""
    dpi, figsize = _style(config)
    out = config.path("figures_dir") / "ablation_comparison.png"
    if ablation_df.empty:
        return None
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(ablation_df["variant"], ablation_df["acer"])
    ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_ylabel("ACER (pooled LOSO-CV)")
    ax.set_xlabel("Вариант модели PLS")
    ax.set_title("Абляция компонент PLS")
    ax.set_ylim(0, max(1.0, float(ablation_df["acer"].max()) * 1.2))
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return _save(fig, out, dpi)


def generate_all_plots(
    config: Config,
    predictions: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
    landmark_example: tuple[Path, Path | None] | None = None,
) -> dict[str, Any]:
    """Построить все основные графики MVP."""
    ensure_dir(config.path("figures_dir"))
    made: dict[str, Any] = {}
    if landmark_example is not None:
        made["landmark_example"] = plot_landmark_example(config, *landmark_example)
    made["pls_score_distribution"] = plot_score_distribution(config, predictions)
    made["roc_curve"] = plot_roc(config, predictions)
    made["confusion_matrix_pls"] = plot_confusion(
        config, predictions, "M1_pls", "confusion_matrix_pls.png"
    )
    made["confusion_matrix_baseline"] = plot_confusion(
        config, predictions, "B1_static2d", "confusion_matrix_baseline.png"
    )
    made["metric_comparison"] = plot_metric_comparison(config, pooled_metrics)
    return made


# ==========================================================================
# Графики синтетического исследования (src/synthetic_study.py)
# ==========================================================================
#: Категориальная палитра, проверенная на различимость при дальтонизме
#: (adjacent CVD dE = 10.8 protan, контраст к фону >= 3:1).
SYN_BLUE = "#4A6FD4"
SYN_ORANGE = "#C2631A"
SYN_TEAL = "#1A8F72"
SYN_GRAY = "#6F7787"
SYN_INK = "#1B1F27"
SYN_MUTED = "#5C6472"
SYN_GRID = "#D8DCE6"

#: Приписка, которая обязана стоять на каждом графике симуляции.
SYN_NOTE = "СИМУЛЯЦИЯ: идеализированная 3D-сцена, не эксперимент на людях"


def _syn_axes(ax: "plt.Axes") -> None:
    """Единое оформление осей: рецессивная сетка, спокойные подписи."""
    ax.grid(True, which="major", color=SYN_GRID, linewidth=0.8, alpha=0.9)
    ax.grid(True, which="minor", color=SYN_GRID, linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SYN_GRID)
    ax.tick_params(colors=SYN_MUTED, labelsize=9)
    ax.xaxis.label.set_color(SYN_INK)
    ax.yaxis.label.set_color(SYN_INK)


def _syn_dir(config: Config) -> Path:
    return ensure_dir(config.path("figures_dir") / "synthetic")


def plot_synthetic_depth_sweep(config: Config, df: pd.DataFrame) -> Path:
    """E1: как временные признаки растут с рельефом лица.

    Ось Y логарифмическая: значения признаков различаются на три порядка,
    поэтому вторая ось была бы ошибкой — используется один масштаб.
    """
    dpi, _ = _style(config)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    series = [("G1", SYN_BLUE, "G1 — псевдоглубина носа"),
              ("G2", SYN_ORANGE, "G2 — лево-правая асимметрия"),
              ("G3", SYN_TEAL, "G3 — вертикальная пропорция")]
    for col, color, label in series:
        ax.plot(df["nose_depth"], df[col], color=color, linewidth=2,
                marker="o", markersize=5, label=label)
        ax.annotate(col, (df["nose_depth"].iloc[-1], df[col].iloc[-1]),
                    textcoords="offset points", xytext=(8, 0), color=color,
                    fontsize=10, fontweight="bold", va="center")
    ax.axvline(0.0, color=SYN_MUTED, linewidth=1, linestyle=":")
    ax.annotate("плоская атака\n(нос в плоскости щёк)", (0.0, df["G2"].max()),
                textcoords="offset points", xytext=(8, -6), color=SYN_MUTED, fontsize=8.5)
    ax.set_yscale("log")
    ax.set_xlabel("Рельеф лица: вынос кончика носа вперёд (условные единицы)")
    ax.set_ylabel("Временная дисперсия признака (лог. шкала)")
    ax.set_title("E1. Чем выраженнее рельеф лица, тем больше все три признака\n" + SYN_NOTE,
                 fontsize=11, color=SYN_INK)
    ax.set_xlim(-0.012, df["nose_depth"].max() * 1.12)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    return _save(fig, _syn_dir(config) / "synthetic_depth_sweep.png", dpi)


def plot_synthetic_distance_sweep(config: Config, df: pd.DataFrame) -> Path:
    """E2: почему протокол требует снимать лицо крупным планом."""
    dpi, _ = _style(config)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(df["camera_distance"], df["G1_separation_ideal"], color=SYN_BLUE,
            linewidth=2, marker="o", markersize=5,
            label="идеальная сцена, без шума")
    ax.plot(df["camera_distance"], df["G1_separation_noisy"], color=SYN_ORANGE,
            linewidth=2, marker="s", markersize=5,
            label="с шумом псевдоглубины (sigma = 0.002)")
    ax.axhline(1.0, color=SYN_MUTED, linewidth=1.2, linestyle="--")
    ax.annotate("разделения нет", (df["camera_distance"].iloc[-1], 1.0),
                textcoords="offset points", xytext=(-4, 7), color=SYN_MUTED,
                fontsize=8.5, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Расстояние от камеры до лица (условные единицы)")
    ax.set_ylabel("Во сколько раз G1 живого больше, чем у плоского")
    ax.set_title("E2. Без шума разделение растёт с расстоянием — но с шумом исчезает\n" + SYN_NOTE,
                 fontsize=11, color=SYN_INK)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    return _save(fig, _syn_dir(config) / "synthetic_distance_sweep.png", dpi)


def plot_synthetic_motion_types(config: Config, df: pd.DataFrame, amplitude: float = 25.0) -> Path:
    """E3: разные признаки требуют разных движений головы.

    Нижняя граница оси зафиксирована на 1e-6: значения ниже этого порога
    практически равны нулю, и растягивать шкалу на восемнадцать порядков
    ради них — значит сплющить единственную содержательную часть графика.
    Такие столбцы обрезаются до основания и подписываются «= 0».
    """
    dpi, _ = _style(config)
    subset = df[df["amplitude_deg"] == amplitude]
    regimes = ["только поворот", "только наклон", "поворот + наклон"]
    subset = subset.set_index("regime").loc[regimes]

    floor = 1e-6
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8), sharey=True)
    panels = [("G1_live", "G1_planar", "G1 — псевдоглубина носа"),
              ("G2_live", "G2_planar", "G2 — лево-правая асимметрия")]
    x = np.arange(len(regimes))
    width = 0.38

    for ax, (live_col, flat_col, title) in zip(axes, panels):
        pairs = [
            (x - width / 2 - 0.01, subset[live_col].to_numpy(dtype=float), SYN_BLUE, "живое лицо"),
            (x + width / 2 + 0.01, subset[flat_col].to_numpy(dtype=float), SYN_ORANGE, "плоская атака"),
        ]
        for positions, values, color, label in pairs:
            drawn = np.maximum(values, floor)
            ax.bar(positions, drawn, width, color=color, label=label)
            # значения ниже порога честно помечаем нулём, а не рисуем «почти ноль»
            for pos, raw in zip(positions, values):
                if raw < floor:
                    ax.annotate("= 0", (pos, floor), textcoords="offset points",
                                xytext=(0, 4), ha="center", fontsize=8.5, color=SYN_MUTED)
        ax.set_xticks(x, regimes, fontsize=9)
        ax.set_title(title, fontsize=10, color=SYN_INK)
        _syn_axes(ax)
        plt.setp(ax.get_xticklabels(), rotation=12, ha="right")

    axes[0].set_yscale("log")
    axes[0].set_ylim(floor, 0.45)
    axes[0].set_ylabel("Временная дисперсия (лог. шкала)")
    axes[0].legend(fontsize=9, frameon=False, loc="upper left")
    fig.suptitle(
        f"E3. Поворот включает G2, наклон включает G1 (амплитуда {amplitude:.0f}°)\n" + SYN_NOTE,
        fontsize=11, color=SYN_INK,
    )
    return _save(fig, _syn_dir(config) / "synthetic_motion_types.png", dpi)


def plot_synthetic_noise(config: Config, df: pd.DataFrame) -> Path:
    """E4: устойчивость классификации к шуму псевдоглубины."""
    dpi, _ = _style(config)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    styles = {
        "M1_pls": (SYN_BLUE, "o", "M1: PLS (временная геометрия)"),
        "B1_static2d": (SYN_ORANGE, "s", "B1: статический 2D-базлайн"),
        "B0_majority": (SYN_GRAY, "^", "B0: мажоритарный (ориентир)"),
    }
    for model, (color, marker, label) in styles.items():
        grp = df[df["model"] == model]
        if grp.empty:
            continue
        agg = grp.groupby("sigma")["acer"].mean().reset_index()
        ax.plot(agg["sigma"], agg["acer"], color=color, linewidth=2,
                marker=marker, markersize=5, label=label)
    _syn_axes(ax)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("sigma синтетического шума псевдоглубины z")
    ax.set_ylabel("ACER (меньше — лучше)")
    ax.set_ylim(0, 0.62)
    ax.set_title("E4. Шум в псевдоглубине почти не портит результат:\n"
                 "основной вклад даёт G2, который не использует z\n" + SYN_NOTE,
                 fontsize=11, color=SYN_INK)
    ax.legend(fontsize=9, frameon=False, loc="center right")
    return _save(fig, _syn_dir(config) / "synthetic_noise_robustness.png", dpi)


def plot_synthetic_roc(config: Config, predictions: pd.DataFrame) -> Path | None:
    """E4: ROC-кривые на синтетическом наборе."""
    dpi, _ = _style(config)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    styles = {"M1_pls": (SYN_BLUE, "M1: PLS"), "B1_static2d": (SYN_ORANGE, "B1: статический 2D")}
    plotted = False
    for model, (color, label) in styles.items():
        grp = predictions[predictions["model"] == model]
        if grp.empty:
            continue
        fpr, tpr = roc_curve_points(grp["y_true"].to_numpy(), grp["score"].to_numpy())
        if fpr.size == 0:
            continue
        auc = roc_auc(grp["y_true"].to_numpy(), grp["score"].to_numpy())
        ax.plot(fpr, tpr, color=color, linewidth=2, marker="o", markersize=4,
                label=f"{label} (AUC = {auc:.3f})")
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.plot([0, 1], [0, 1], color=SYN_MUTED, linewidth=1, linestyle="--",
            label="случайное угадывание")
    _syn_axes(ax)
    ax.set_xlabel("FPR — доля атак, принятых за живое лицо")
    ax.set_ylabel("TPR — доля верно принятых живых")
    ax.set_title("E4. ROC на синтетическом наборе (8 участников, LOSO)\n" + SYN_NOTE,
                 fontsize=11, color=SYN_INK)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    return _save(fig, _syn_dir(config) / "synthetic_roc.png", dpi)


def plot_synthetic_threshold_rules(config: Config, df: pd.DataFrame) -> Path | None:
    """E5: сравнение правил выбора порога при разном числе участников."""
    dpi, _ = _style(config)
    out = _syn_dir(config) / "synthetic_threshold_rules.png"
    pls = df[df["model"] == "M1_pls"]
    if pls.empty:
        return None

    order = ["min_acer", "min_acer_mid", "class_midpoint", "eer", "fixed_zero"]
    labels = {
        "min_acer": "min_acer\n(историческое)",
        "min_acer_mid": "min_acer_mid\n(середина плато)",
        "class_midpoint": "class_midpoint\n(между средними)",
        "eer": "eer\n(равные ошибки)",
        "fixed_zero": "fixed_zero\n(нуль)",
    }
    counts = sorted(pls["n_subjects"].unique())
    colors = {counts[0]: SYN_ORANGE, counts[-1]: SYN_BLUE}

    x = np.arange(len(order))
    width = 0.8 / max(1, len(counts))
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    for i, n_sub in enumerate(counts):
        subset = pls[pls["n_subjects"] == n_sub]
        means = [subset[subset["rule"] == r]["acer"].mean() for r in order]
        errors = [subset[subset["rule"] == r]["acer"].std() for r in order]
        bars = ax.bar(x + i * width, means, width, yerr=errors, capsize=3,
                      color=colors.get(n_sub, SYN_GRAY),
                      label=f"{n_sub} участников")
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    _syn_axes(ax)
    ax.set_xticks(x + width * (len(counts) - 1) / 2, [labels[r] for r in order], fontsize=8.5)
    ax.set_ylabel("ACER (меньше — лучше)")
    ax.set_title("E5. Чем меньше участников, тем сильнее правило выбора порога\n"
                 "влияет на результат — при 8 участниках разница почти исчезает\n" + SYN_NOTE,
                 fontsize=11, color=SYN_INK)
    ax.legend(fontsize=9, frameon=False)
    return _save(fig, out, dpi)


def generate_synthetic_plots(config: Config, results: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Построить все графики синтетического исследования."""
    made: dict[str, Any] = {}
    made["depth"] = plot_synthetic_depth_sweep(config, results["depth"])
    made["distance"] = plot_synthetic_distance_sweep(config, results["distance"])
    made["motion"] = plot_synthetic_motion_types(config, results["motion"])
    made["noise"] = plot_synthetic_noise(config, results["noise"])
    made["roc"] = plot_synthetic_roc(config, results["predictions"])
    if "rules" in results:
        made["rules"] = plot_synthetic_threshold_rules(config, results["rules"])
    return made
