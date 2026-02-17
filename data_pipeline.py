from __future__ import annotations

from dataclasses import dataclass
import os
from typing import IO, Union

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


CsvSource = Union[str, "os.PathLike[str]", IO[bytes], IO[str]]


def prepare_datasets(csv_source: CsvSource) -> PreparedData:
    """
    Load and prepare a StudentsPerformance-like CSV for:
    - classification (pass_fail)
    - regression (average_score)
    - clustering (X)
    """
    try:
        df = pd.read_csv(csv_source)
    except Exception as e:  # pragma: no cover
        raise DatasetError(f"Could not read CSV: {e}") from e

    df = df.copy()

    # If the CSV looks like the original StudentsPerformance dataset,
    # use the domain-specific pipeline. Otherwise, fall back to a
    # flexible generic pipeline that works with "any" tabular data.
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if not missing:
        df["average_score"] = (
            df["math score"] + df["reading score"] + df["writing score"]
        ) / 3
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
    else:
        # Flexible mode:
        # - Take all numeric columns
        # - Use the LAST numeric column as a regression target
        # - Use median split on that column for classification
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            missing_list = ", ".join(sorted(missing))
            raise DatasetError(
                "CSV is missing the expected StudentsPerformance columns "
                f"({missing_list}) and also does not contain at least two numeric "
                "columns to run a generic demo."
            )

        target_col = num_cols[-1]
        feature_num_cols = num_cols[:-1]

        X_num = df[feature_num_cols]
        y_reg = df[target_col]
        # Binary label: above/below median of target
        threshold = y_reg.median()
        y_class = (y_reg >= threshold).astype(int)

        # One-hot encode any non-numeric columns and join to numeric features.
        cat_cols = [c for c in df.columns if c not in num_cols]
        if cat_cols:
            X_cat = pd.get_dummies(df[cat_cols], columns=cat_cols)
            X = pd.concat([X_num, X_cat], axis=1)
        else:
            X = X_num

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

