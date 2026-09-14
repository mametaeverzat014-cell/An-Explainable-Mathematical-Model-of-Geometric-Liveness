"""Базовые модели сравнения.

    B0 — тривиальный базлайн большинства класса (majority);
    B1 — статический однокадровый 2D-базлайн: StandardScaler + LogisticRegression.

B1 использует ТОЛЬКО один средний валидный кадр каждого видео и простые
интерпретируемые 2D-признаки (без временной информации). Никаких CNN,
трансформеров и тяжёлых предобученных моделей.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.build_features import BASELINE_FEATURE_NAMES
from src.evaluation import select_threshold
from src.utils import Config


class MajorityBaseline:
    """B0: всегда предсказывает наиболее частый класс обучающей выборки."""

    def __init__(self) -> None:
        self.majority_: int = 1
        self.tau_: float = 0.5
        self.direction_: str = "live_high"
        self.threshold_rule_: str = "majority_class"

    def fit(self, train_df: pd.DataFrame) -> "MajorityBaseline":
        y = train_df["y"].to_numpy(dtype=int)
        self.majority_ = int(np.bincount(y, minlength=2).argmax()) if y.size else 1
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Константный скор (ROC-AUC для B0 не информативен — это ожидаемо)."""
        return np.full(len(df), 1.0 if self.majority_ == 1 else 0.0, dtype=float)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.majority_, dtype=int)


class StaticBaseline2D:
    """B1: логистическая регрессия на 2D-признаках одного кадра.

    Attributes:
        pipeline_: Imputer -> StandardScaler -> LogisticRegression.
        tau_: порог на предсказанной вероятности класса live.
    """

    def __init__(self, config: Config, threshold_rule: str | None = None) -> None:
        cfg = config.get("baseline", {})
        pls_cfg = config.get("pls", {})
        self.C = float(cfg.get("logreg_C", 1.0))
        self.max_iter = int(cfg.get("logreg_max_iter", 1000))
        self.class_weight = cfg.get("class_weight", "balanced")
        self.seed = config.seed
        self.rule = str(threshold_rule or pls_cfg.get("threshold_rule", "min_acer_mid"))
        self.lam_attack = float(pls_cfg.get("lambda_attack", 1.0))
        self.lam_live = float(pls_cfg.get("lambda_live", 1.0))

        self.feature_names = list(BASELINE_FEATURE_NAMES)
        self.pipeline_: Pipeline | None = None
        self.constant_prediction_: int | None = None
        self.tau_: float = 0.5
        self.direction_: str = "live_high"
        self.threshold_rule_: str = "not_fitted"

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.feature_names].to_numpy(dtype=np.float64)

    def fit(self, train_df: pd.DataFrame) -> "StaticBaseline2D":
        """Обучить масштабирование и логрегрессию ТОЛЬКО на обучающих субъектах."""
        y = train_df["y"].to_numpy(dtype=int)
        if len(np.unique(y)) < 2:
            # Вырожденный фолд: логрегрессию обучить нельзя.
            self.constant_prediction_ = int(y[0]) if y.size else 1
            self.threshold_rule_ = "fallback_single_class_train"
            return self

        self.constant_prediction_ = None
        self.pipeline_ = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        C=self.C,
                        max_iter=self.max_iter,
                        class_weight=self.class_weight,
                        random_state=self.seed,
                    ),
                ),
            ]
        )
        self.pipeline_.fit(self._matrix(train_df), y)
        train_scores = self.pipeline_.predict_proba(self._matrix(train_df))[:, 1]
        self.tau_, _ = select_threshold(
            train_scores, y, self.rule, self.lam_attack, self.lam_live
        )
        self.threshold_rule_ = f"train_{self.rule}"
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Вероятность класса live (больше => более вероятно live)."""
        if self.pipeline_ is None:
            return np.full(len(df), float(self.constant_prediction_ or 0), dtype=float)
        return self.pipeline_.predict_proba(self._matrix(df))[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            return np.full(len(df), int(self.constant_prediction_ or 1), dtype=int)
        return (self.score(df) >= self.tau_).astype(int)

    def coefficients(self) -> pd.DataFrame:
        """Коэффициенты логрегрессии — интерпретируемость базлайна."""
        if self.pipeline_ is None:
            return pd.DataFrame(columns=["feature", "coefficient"])
        coefs = self.pipeline_.named_steps["logreg"].coef_.ravel()
        return pd.DataFrame({"feature": self.feature_names, "coefficient": coefs})
