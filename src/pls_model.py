"""M1: предложенная временная модель Parallax Liveness Score (PLS).

Модель полностью интерпретируема и не содержит обучаемых весов, кроме
статистик нормализации:

    Gk_norm = [Gk - mean_train(Gk)] / [std_train(Gk) + eps]
    PLS     = (G1_norm + G2_norm + G3_norm) / 3

Правило классификации:

    predict live, если PLS >= tau, иначе attack

Направление неравенства может быть изменено на противоположное ТОЛЬКО по
обучающим данным (`direction: auto` в конфигурации). Тестовый субъект
никогда не участвует ни в нормализации, ни в выборе tau, ни в выборе
направления.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.build_features import G_NAMES
from src.evaluation import select_threshold
from src.utils import Config, get_logger


class PLSModel:
    """Parallax Liveness Score с train-only нормализацией и выбором порога.

    Attributes:
        mean_, std_: статистики нормализации, оценённые только на train.
        direction_: 'live_high' (live при PLS >= tau) или 'live_low'.
        tau_: выбранный порог на ОРИЕНТИРОВАННОМ скоре.
        threshold_rule_: как именно был выбран порог (для отчётности).
    """

    def __init__(
        self,
        config: Config,
        g_features: Sequence[str] | None = None,
        threshold_rule: str | None = None,
    ) -> None:
        cfg = config.get("pls", {})
        self.epsilon = float(config.get("features", {}).get("epsilon", 1e-8))
        self.g_features = list(
            g_features if g_features is not None else config.get("features", {}).get("g_features", G_NAMES)
        )
        self.direction_cfg = str(cfg.get("direction", "auto"))
        self.rule = str(threshold_rule or cfg.get("threshold_rule", "min_acer_mid"))
        self.lam_attack = float(cfg.get("lambda_attack", 1.0))
        self.lam_live = float(cfg.get("lambda_live", 1.0))
        self.min_train_subjects = int(cfg.get("min_train_subjects_for_inner_val", 3))
        self.fallback_tau = float(cfg.get("fallback_threshold", 0.0))

        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.direction_: str = "live_high"
        self.tau_: float = self.fallback_tau
        self.threshold_rule_: str = "not_fitted"

    # ---------------------------------------------------------------- утилиты
    def _raw_matrix(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.g_features].to_numpy(dtype=np.float64)

    def _combine(self, g_norm: np.ndarray) -> np.ndarray:
        """Свернуть нормированные признаки в одну величину.

        Единственное место, где задан способ объединения G1..G3. И обычный
        скор, и внутренняя валидация для выбора порога обязаны пользоваться
        им же: иначе tau подбирается по одной шкале, а применяется к другой.
        """
        return g_norm.mean(axis=1)

    def _normalize(self, df: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("PLSModel не обучен: сначала вызовите fit().")
        return (self._raw_matrix(df) - self.mean_) / (self.std_ + self.epsilon)

    def _pls_raw(self, df: pd.DataFrame) -> np.ndarray:
        """PLS до применения направления."""
        return self._combine(self._normalize(df))

    @staticmethod
    def _fit_stats(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return g.mean(axis=0), g.std(axis=0)

    def _orient(self, pls: np.ndarray) -> np.ndarray:
        """Ориентированный скор: больше => более вероятно live."""
        return pls if self.direction_ == "live_high" else -pls

    # ---------------------------------------------------------------- обучение
    def fit(self, train_df: pd.DataFrame) -> "PLSModel":
        """Оценить нормализацию, направление и порог ТОЛЬКО на train-субъектах."""
        logger = get_logger()
        if train_df.empty:
            raise ValueError("PLSModel.fit получил пустую обучающую выборку.")

        g_train = self._raw_matrix(train_df)
        self.mean_, self.std_ = self._fit_stats(g_train)
        y_train = train_df["y"].to_numpy(dtype=int)
        pls_train = self._pls_raw(train_df)

        # --- направление неравенства (только по train) ---
        if self.direction_cfg in ("live_high", "live_low"):
            self.direction_ = self.direction_cfg
        elif len(np.unique(y_train)) < 2:
            self.direction_ = "live_high"  # документированное значение по умолчанию
        else:
            mean_live = float(np.mean(pls_train[y_train == 1]))
            mean_attack = float(np.mean(pls_train[y_train == 0]))
            self.direction_ = "live_high" if mean_live >= mean_attack else "live_low"

        oriented_train = self._orient(pls_train)

        # --- порог tau (только по train/внутренней валидации) ---
        n_subjects = int(train_df["subject_id"].nunique())
        if len(np.unique(y_train)) < 2:
            self.tau_ = self.fallback_tau
            self.threshold_rule_ = "fallback_single_class_train"
        elif n_subjects >= self.min_train_subjects:
            inner_scores, inner_y = self._inner_validation_scores(train_df)
            if inner_scores.size and len(np.unique(inner_y)) == 2:
                self.tau_, _ = select_threshold(
                    inner_scores, inner_y, self.rule, self.lam_attack, self.lam_live
                )
                self.threshold_rule_ = f"inner_loso_{self.rule}"
            else:
                self.tau_, _ = select_threshold(
                    oriented_train, y_train, self.rule, self.lam_attack, self.lam_live
                )
                self.threshold_rule_ = f"train_resubstitution_{self.rule}"
        else:
            # Слишком мало обучающих субъектов для внутренней валидации.
            # Используется фиксированное документированное правило (protocol.md).
            self.tau_, _ = select_threshold(
                oriented_train, y_train, self.rule, self.lam_attack, self.lam_live
            )
            self.threshold_rule_ = f"train_resubstitution_{self.rule}"
            logger.debug(
                "PLS: только %d обучающих субъектов — порог выбран на train (см. protocol.md)",
                n_subjects,
            )
        return self

    def _inner_validation_scores(self, train_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Внутренний LOSO по обучающим субъектам: out-of-fold ориентированные скоры.

        Нормализация каждого внутреннего фолда оценивается только на его
        внутренней обучающей части, поэтому тестовый субъект внешнего фолда
        не влияет на выбор tau.
        """
        scores: list[float] = []
        ys: list[int] = []
        subjects = sorted(train_df["subject_id"].astype(str).unique().tolist())
        for subj in subjects:
            mask = train_df["subject_id"].astype(str) == subj
            inner_train, inner_val = train_df[~mask], train_df[mask]
            if inner_train.empty or inner_val.empty:
                continue
            g_inner = self._raw_matrix(inner_train)
            mean_i, std_i = self._fit_stats(g_inner)
            g_val = (self._raw_matrix(inner_val) - mean_i) / (std_i + self.epsilon)
            pls_val = self._combine(g_val)
            oriented = pls_val if self.direction_ == "live_high" else -pls_val
            scores.extend(oriented.tolist())
            ys.extend(inner_val["y"].astype(int).tolist())
        return np.asarray(scores, dtype=float), np.asarray(ys, dtype=int)

    # ---------------------------------------------------------------- вывод
    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Ориентированный скор живости (больше => более вероятно live)."""
        return self._orient(self._pls_raw(df))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """1 = live, 0 = attack по правилу ``score >= tau``."""
        return (self.score(df) >= self.tau_).astype(int)

    def explain(self, df: pd.DataFrame) -> pd.DataFrame:
        """Покомпонентный вклад G1/G2/G3 — основа объяснимости модели."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("PLSModel не обучен: сначала вызовите fit().")
        g_norm = (self._raw_matrix(df) - self.mean_) / (self.std_ + self.epsilon)
        out = pd.DataFrame(g_norm, columns=[f"{n}_norm" for n in self.g_features])
        out.insert(0, "video_id", df["video_id"].to_numpy())
        out["PLS"] = g_norm.mean(axis=1)
        out["score_oriented"] = self._orient(out["PLS"].to_numpy())
        out["tau"] = self.tau_
        out["direction"] = self.direction_
        return out


class WeightedPLSModel(PLSModel):
    """PLS со взвешиванием признаков по их разделяющей способности.

    Мотивация — результат эксперимента с шумом (`research_log.md`, запись 10):
    равновесное среднее пропускает испорченный признак в итоговую оценку с
    полным весом ``1/3``, и порча одного признака из трёх уводит ``ACER`` с
    ``0.000`` до ``0.262``. Здесь вес признака определяется тем, насколько он
    в действительности разделяет классы:

        a_k = |2 * AUC_k - 1|                    (ранговое разделение классов)
        w_k = a_k / sum_j a_j,          PLS = sum_k w_k * Gk_norm

    ``AUC_k`` — площадь под ROC-кривой признака на обучающих субъектах, то
    есть доля пар «живое/атака», которые признак упорядочивает верно.
    Обучаемых весов в смысле градиентного обучения нет: веса суть
    описательная статистика обучающей выборки.

    ПОЧЕМУ НЕ РАЗНОСТЬ СРЕДНИХ. Первой была опробована стандартизованная
    разность средних (величина Коэна). Она оказалась непригодна: эта мера
    делит на разброс внутри класса, а у ``G2`` разброс велик
    (``4.98e-2 … 8.99e-1`` у живых записей), хотя классы он разделяет
    полностью. В результате наибольший вес получал ``G1`` — признак, про
    который в записи 9 журнала показано, что он классы не разделяет вовсе.
    Ранговая мера от разброса не зависит и этим дефектом не страдает.

    ВАЖНО О ЧЕСТНОСТИ ПРОТОКОЛА. Эта модель **не заменяет** предложенную:
    равные веса ``1/3`` зафиксированы протоколом до сбора данных, и основной
    результат работы приводится именно для них. Здесь — дополнительный,
    разведочный анализ, выполненный уже после того, как данные увидены.

    Веса оцениваются **только на обучающих субъектах** внешнего фолда.
    Внутренняя валидация для выбора порога переоценивает нормализацию на своей
    части, но пользуется весами внешнего обучения; тестовый субъект не влияет
    ни на то, ни на другое. Порог подбирается по той же свёртке признаков,
    которая затем применяется (см. :meth:`PLSModel._combine`).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weights_: np.ndarray | None = None

    def _separability_weights(self, g_norm: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Веса по ранговому разделению классов (|2*AUC - 1|) на обучающих данных."""
        from src.evaluation import roc_auc

        n_features = g_norm.shape[1]
        equal = np.full(n_features, 1.0 / n_features)
        if len(np.unique(y)) < 2:
            return equal

        strength = np.empty(n_features, dtype=float)
        for k in range(n_features):
            column = g_norm[:, k]
            if not np.isfinite(column).all() or np.ptp(column) <= self.epsilon:
                strength[k] = 0.0        # постоянный или испорченный признак
                continue
            strength[k] = abs(2.0 * roc_auc(y, column) - 1.0)

        total = float(strength.sum())
        if not np.isfinite(total) or total <= self.epsilon:
            # Ни один признак не разделяет классы — вырождаемся в равные веса.
            return equal
        return strength / total

    def _combine(self, g_norm: np.ndarray) -> np.ndarray:
        """Взвешенная сумма вместо равновесного среднего."""
        if self.weights_ is None:
            return g_norm.mean(axis=1)
        return g_norm @ self.weights_

    def fit(self, train_df: pd.DataFrame) -> "WeightedPLSModel":
        """Оценить веса на обучающих субъектах, затем обучить как обычный PLS."""
        if train_df.empty:
            raise ValueError("WeightedPLSModel.fit получил пустую обучающую выборку.")
        g = self._raw_matrix(train_df)
        mean, std = self._fit_stats(g)
        g_norm = (g - mean) / (std + self.epsilon)
        self.weights_ = self._separability_weights(
            g_norm, train_df["y"].to_numpy(dtype=int)
        )
        super().fit(train_df)
        return self

    def explain(self, df: pd.DataFrame) -> pd.DataFrame:
        """Покомпонентный разбор с фактическими весами признаков."""
        out = super().explain(df)
        if self.weights_ is not None:
            for name, weight in zip(self.g_features, self.weights_):
                out[f"{name}_weight"] = float(weight)
        return out
