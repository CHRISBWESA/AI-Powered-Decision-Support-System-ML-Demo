from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd  # pyright: ignore[reportMissingModuleSource]
from sklearn.model_selection import train_test_split  # pyright: ignore[reportMissingModuleSource]


class DatasetError(ValueError):
    """Raised when an uploaded dataset can't be used by this demo."""


REQUIRED_COLUMNS = {
    # Targets / numeric scores
    "math score",
    "reading score",
    "writing score",
    # Categorical features expected by this demo
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
}


@dataclass(frozen=True)
class PreparedData:
    X_train_class: pd.DataFrame
    X_test_class: pd.DataFrame
    y_train_class: pd.Series
    y_test_class: pd.Series
    X_train_reg: pd.DataFrame
    X_test_reg: pd.DataFrame
    y_train_reg: pd.Series
    y_test_reg: pd.Series
    X: pd.DataFrame
    y_class: pd.Series
    y_reg: pd.Series


def prepare_datasets(csv_path: str) -> PreparedData:
    """
    Load and prepare a StudentsPerformance-like CSV for:
    - classification (pass_fail)
    - regression (average_score)
    - clustering (X)
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:  # pragma: no cover
        raise DatasetError(f"Could not read CSV: {e}") from e

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise DatasetError(
            "CSV is missing required columns for this demo: " + missing_list
        )

    df = df.copy()
    df["average_score"] = (df["math score"] + df["reading score"] + df["writing score"]) / 3
    df["pass_fail"] = df["average_score"].apply(lambda x: 1 if x >= 60 else 0)

    df_encoded = pd.get_dummies(
        df,
        columns=[
            "gender",
            "race/ethnicity",
            "parental level of education",
            "lunch",
            "test preparation course",
        ],
    )

    X = df_encoded.drop(
        ["math score", "reading score", "writing score", "average_score", "pass_fail"],
        axis=1,
    )
    y_class = df_encoded["pass_fail"]
    y_reg = df_encoded["average_score"]

    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    return PreparedData(
        X_train_class=X_train_class,
        X_test_class=X_test_class,
        y_train_class=y_train_class,
        y_test_class=y_test_class,
        X_train_reg=X_train_reg,
        X_test_reg=X_test_reg,
        y_train_reg=y_train_reg,
        y_test_reg=y_test_reg,
        X=X,
        y_class=y_class,
        y_reg=y_reg,
    )

