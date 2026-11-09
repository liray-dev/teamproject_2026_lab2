"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary

AGE_THRESHOLD = 30
TOP_FARES_COUNT = 5


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    passengers = pd.read_csv(data.csv_path)
    by_class = passengers.groupby("Pclass")
    return TitanicSummary(
        row_count=len(passengers),
        missing_by_column={str(column): int(count) for column, count in passengers.isna().sum().items()},
        adults_over_30_count=int((passengers["Age"] > AGE_THRESHOLD).sum()),
        mean_age_by_pclass=_by_class(by_class["Age"].mean()),
        survival_rate_by_pclass=_by_class(by_class["Survived"].mean()),
        highest_fares=[float(fare) for fare in passengers["Fare"].nlargest(TOP_FARES_COUNT)],
    )


def _by_class(values: pd.Series) -> dict[int, float]:
    return {int(pclass): float(value) for pclass, value in values.sort_index().items()}
