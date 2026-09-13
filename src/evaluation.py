"""Метрики PAD и subject-disjoint протокол LOSO-CV.

Определения (label: 1 = live/bona fide, 0 = attack):

    APCER    = (# атак, классифицированных как live) / (# атак)
    BPCER    = (# live, классифицированных как attack) / (# live)
    ACER     = (APCER + BPCER) / 2
    Accuracy = (# верно классифицированных) / (# всех)

    Риск-ориентированная функция потерь:
        L(tau) = lambda_attack * APCER(tau) + lambda_live * BPCER(tau)

Правила протокола (см. protocol.md):
    * все видео одного участника попадают в один и тот же внешний тестовый фолд;
    * разбиение НИКОГДА не делается по кадрам;
    * нормализация признаков и базлайн обучаются ТОЛЬКО на обучающих субъектах;
    * порог tau выбирается ТОЛЬКО на обучающих/валидационных субъектах;
    * каждый тестовый субъект оценивается ровно один раз.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from src.utils import Config, ensure_dir, get_logger

EPS = 1e-12


# --------------------------------------------------------------------------
# Базовые метрики
# --------------------------------------------------------------------------
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Матрица ошибок в терминах PAD (1 = live, 0 = attack)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "n_live": int((y_true == 1).sum()),
        "n_attack": int((y_true == 0).sum()),
        "live_as_live": int(((y_true == 1) & (y_pred == 1)).sum()),
        "live_as_attack": int(((y_true == 1) & (y_pred == 0)).sum()),
        "attack_as_live": int(((y_true == 0) & (y_pred == 1)).sum()),
        "attack_as_attack": int(((y_true == 0) & (y_pred == 0)).sum()),
    }


def apcer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Доля атак, ошибочно принятых за live. NaN, если атак нет."""
    c = confusion_counts(y_true, y_pred)
    return float(c["attack_as_live"] / c["n_attack"]) if c["n_attack"] else float("nan")


def bpcer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Доля live, ошибочно отклонённых как атака. NaN, если live нет."""
    c = confusion_counts(y_true, y_pred)
    return float(c["live_as_attack"] / c["n_live"]) if c["n_live"] else float("nan")


def acer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ACER = (APCER + BPCER) / 2."""
    a, b = apcer(y_true, y_pred), bpcer(y_true, y_pred)
    return float(np.nanmean([a, b])) if not (np.isnan(a) and np.isnan(b)) else float("nan")


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Доля верно классифицированных образцов."""
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    return float((y_true == y_pred).mean()) if y_true.size else float("nan")


def risk(y_true: np.ndarray, y_pred: np.ndarray, lam_attack: float, lam_live: float) -> float:
    """L(tau) = lambda_attack * APCER + lambda_live * BPCER."""
    a, b = apcer(y_true, y_pred), bpcer(y_true, y_pred)
    a = 0.0 if np.isnan(a) else a
    b = 0.0 if np.isnan(b) else b
    return float(lam_attack * a + lam_live * b)


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC (live = положительный класс). NaN, если присутствует один класс."""
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, np.asarray(scores, dtype=float)))


def roc_curve_points(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Точки ROC-кривой (fpr, tpr); пустые массивы при одном классе."""
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return np.array([]), np.array([])
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, np.asarray(scores, dtype=float))
    return fpr, tpr


def compute_all_metrics(
    y_true: Iterable[int], y_pred: Iterable[int], scores: Iterable[float] | None = None
) -> dict[str, float]:
    """Полный набор метрик по сохранённым предсказаниям."""
    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)
    metrics: dict[str, float] = {
        "n": int(y_true.size),
        "apcer": apcer(y_true, y_pred),
        "bpcer": bpcer(y_true, y_pred),
        "acer": acer(y_true, y_pred),
        "accuracy": accuracy(y_true, y_pred),
    }
    metrics.update({k: float(v) for k, v in confusion_counts(y_true, y_pred).items()})
    metrics["roc_auc"] = (
        roc_auc(y_true, np.asarray(list(scores), dtype=float))
        if scores is not None
        else float("nan")
    )
    return metrics


# --------------------------------------------------------------------------
# Выбор порога (ТОЛЬКО на обучающих/валидационных данных!)
# --------------------------------------------------------------------------
def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    """Кандидаты порога: середины между отсортированными уникальными значениями."""
    s = np.unique(np.asarray(scores, dtype=float))
    s = s[np.isfinite(s)]
    if s.size == 0:
        return np.array([0.0])
    if s.size == 1:
        return np.array([s[0] - 1e-6, s[0], s[0] + 1e-6])
    mids = (s[:-1] + s[1:]) / 2.0
    return np.concatenate([[s[0] - 1e-6], mids, [s[-1] + 1e-6]])


def select_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    rule: str = "min_acer",
    lam_attack: float = 1.0,
    lam_live: float = 1.0,
) -> tuple[float, float]:
    """Подобрать tau, минимизируя ACER или L(tau).

    Предполагается ориентированный скор: ``live`` предсказывается при
    ``score >= tau``.

    Returns:
        (tau, значение целевой функции).
    """
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    best_tau, best_obj = 0.0, np.inf
    for tau in threshold_candidates(scores):
        y_pred = (scores >= tau).astype(int)
        obj = (
            acer(y_true, y_pred)
            if rule == "min_acer"
            else risk(y_true, y_pred, lam_attack, lam_live)
        )
        if np.isnan(obj):
            continue
        if obj < best_obj - 1e-12:
            best_tau, best_obj = float(tau), float(obj)
    if not np.isfinite(best_obj):
        return 0.0, float("nan")
    return best_tau, best_obj


# --------------------------------------------------------------------------
# LOSO-CV
# --------------------------------------------------------------------------
def loso_folds(subject_ids: pd.Series) -> list[str]:
    """Список субъектов = список внешних фолдов (каждый тестируется один раз)."""
    return sorted(subject_ids.astype(str).unique().tolist())


def run_loso(
    features: pd.DataFrame,
    config: Config,
    model_builders: dict[str, Callable[[], Any]] | None = None,
    save: bool = True,
    bootstrap: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Выполнить subject-disjoint LOSO-CV для всех моделей.

    Args:
        features: таблица признаков уровня видео (`results/video_features.csv`).
        config: конфигурация прототипа.
        model_builders: отображение "имя модели -> фабрика". По умолчанию
            B0 (majority), B1 (static 2D logreg), M1 (PLS).
        save: записывать ли основные CSV в results/. Вспомогательные прогоны
            (абляция, эксперимент с шумом) вызывают с ``save=False``, чтобы не
            затирать результаты основного эксперимента.
        bootstrap: считать ли bootstrap-доверительные интервалы. ``None`` —
            брать значение из конфигурации.

    Returns:
        (predictions, fold_metrics, pooled_metrics, thresholds_by_fold).
    """
    # Отложенный импорт: модули моделей используют метрики из этого файла.
    from src.baseline import MajorityBaseline, StaticBaseline2D
    from src.pls_model import PLSModel

    logger = get_logger()
    if model_builders is None:
        model_builders = {
            "B0_majority": lambda: MajorityBaseline(),
            "B1_static2d": lambda: StaticBaseline2D(config),
            "M1_pls": lambda: PLSModel(config),
        }

    subjects = loso_folds(features["subject_id"])
    pred_rows: list[dict[str, Any]] = []
    thr_rows: list[dict[str, Any]] = []

    for subject in subjects:
        test_mask = features["subject_id"].astype(str) == subject
        train_df = features[~test_mask].reset_index(drop=True)
        test_df = features[test_mask].reset_index(drop=True)

        if train_df.empty:
            logger.warning("Фолд %s пропущен: нет обучающих субъектов", subject)
            continue
        if train_df["y"].nunique() < 2:
            logger.warning(
                "Фолд %s: в обучающей части только один класс — "
                "модели используют фиксированное правило по умолчанию",
                subject,
            )

        for model_name, builder in model_builders.items():
            model = builder()
            model.fit(train_df)                    # нормализация/веса — только train
            scores = model.score(test_df)          # ориентированный скор (больше = live)
            y_pred = model.predict(test_df)
            thr_rows.append(
                {
                    "fold_subject": subject,
                    "model": model_name,
                    "tau": float(getattr(model, "tau_", np.nan)),
                    "direction": str(getattr(model, "direction_", "n/a")),
                    "threshold_rule": str(getattr(model, "threshold_rule_", "n/a")),
                    "n_train_videos": int(len(train_df)),
                    "n_train_subjects": int(train_df["subject_id"].nunique()),
                }
            )
            for i in range(len(test_df)):
                pred_rows.append(
                    {
                        "fold_subject": subject,
                        "model": model_name,
                        "video_id": test_df.loc[i, "video_id"],
                        "subject_id": test_df.loc[i, "subject_id"],
                        "label": test_df.loc[i, "label"],
                        "attack_type": test_df.loc[i, "attack_type"],
                        "y_true": int(test_df.loc[i, "y"]),
                        "score": float(scores[i]),
                        "y_pred": int(y_pred[i]),
                        "tau": float(getattr(model, "tau_", np.nan)),
                    }
                )

    predictions = pd.DataFrame(pred_rows)
    thresholds = pd.DataFrame(thr_rows)

    fold_rows: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []
    if not predictions.empty:
        for (model_name, subject), grp in predictions.groupby(["model", "fold_subject"]):
            fold_rows.append(
                {
                    "model": model_name,
                    "fold_subject": subject,
                    **compute_all_metrics(grp["y_true"], grp["y_pred"], grp["score"]),
                }
            )
        for model_name, grp in predictions.groupby("model"):
            pooled_rows.append(
                {
                    "model": model_name,
                    **compute_all_metrics(grp["y_true"], grp["y_pred"], grp["score"]),
                }
            )

    fold_metrics = pd.DataFrame(fold_rows)
    pooled_metrics = pd.DataFrame(pooled_rows)

    # Опционально: bootstrap-доверительные интервалы для pooled-метрик.
    eval_cfg = config.get("evaluation", {})
    if bootstrap is None:
        bootstrap = bool(eval_cfg.get("bootstrap_ci", False))
    if bootstrap and not predictions.empty:
        pooled_metrics = add_bootstrap_ci(
            predictions,
            pooled_metrics,
            n_boot=int(eval_cfg.get("bootstrap_n", 1000)),
            alpha=float(eval_cfg.get("bootstrap_alpha", 0.05)),
            seed=config.seed,
        )

    if save:
        results_dir = ensure_dir(config.path("results_dir"))
        predictions.to_csv(results_dir / "predictions.csv", index=False)
        fold_metrics.to_csv(results_dir / "loso_fold_metrics.csv", index=False)
        pooled_metrics.to_csv(results_dir / "pooled_metrics.csv", index=False)
        thresholds.to_csv(results_dir / "thresholds_by_fold.csv", index=False)
        logger.info("LOSO-CV завершён: %d фолдов, результаты в %s", len(subjects), results_dir)
    return predictions, fold_metrics, pooled_metrics, thresholds


def add_bootstrap_ci(
    predictions: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap-доверительные интервалы (ресэмплинг ПО СУБЪЕКТАМ).

    Ресэмплинг выполняется по субъектам, а не по видео: видео одного
    участника статистически зависимы.
    """
    rng = np.random.default_rng(seed)
    out = pooled_metrics.copy()
    lo_acer, hi_acer, lo_acc, hi_acc = [], [], [], []
    for model_name in out["model"]:
        grp = predictions[predictions["model"] == model_name]
        subjects = grp["subject_id"].astype(str).unique()
        acers, accs = [], []
        for _ in range(n_boot):
            picked = rng.choice(subjects, size=len(subjects), replace=True)
            sample = pd.concat(
                [grp[grp["subject_id"].astype(str) == s] for s in picked], ignore_index=True
            )
            acers.append(acer(sample["y_true"].to_numpy(), sample["y_pred"].to_numpy()))
            accs.append(accuracy(sample["y_true"].to_numpy(), sample["y_pred"].to_numpy()))
        lo_acer.append(float(np.nanpercentile(acers, 100 * alpha / 2)))
        hi_acer.append(float(np.nanpercentile(acers, 100 * (1 - alpha / 2))))
        lo_acc.append(float(np.nanpercentile(accs, 100 * alpha / 2)))
        hi_acc.append(float(np.nanpercentile(accs, 100 * (1 - alpha / 2))))
    out["acer_ci_low"] = lo_acer
    out["acer_ci_high"] = hi_acer
    out["accuracy_ci_low"] = lo_acc
    out["accuracy_ci_high"] = hi_acc
    return out
