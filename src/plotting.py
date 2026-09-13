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
from src.evaluation import confusion_counts, roc_curve_points
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
