"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import math

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

FEATURES = ("FSIQ", "VIQ", "PIQ", "Weight", "Height")
TARGET = "MRI_Count"


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    brains = load_brain_data(data.csv_path)
    men, women = split_by_gender(brains)
    men_correlation = mri_correlations(men)
    women_correlation = mri_correlations(women)
    return BrainCorrelationSummary(
        men_count=len(men),
        women_count=len(women),
        women_mri_correlation=women_correlation,
        men_mri_correlation=men_correlation,
        strongest_mri_feature=_strongest_feature(men_correlation, women_correlation),
    )


def load_brain_data(path: str) -> pd.DataFrame:
    brains = pd.read_csv(path, sep="\t", na_values=["NA"]).dropna(how="all")
    brains["Gender"] = brains["Gender"].str.strip()
    return brains


def split_by_gender(brains: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return brains[brains["Gender"] == "Male"], brains[brains["Gender"] == "Female"]


def mri_correlations(group: pd.DataFrame) -> dict[str, float]:
    return {feature: float(group[feature].corr(group[TARGET], method="pearson")) for feature in FEATURES}


def _strongest_feature(*correlations: dict[str, float]) -> str:
    candidates = [
        (abs(value), feature)
        for correlation in correlations
        for feature, value in correlation.items()
        if feature != TARGET and not math.isnan(value)
    ]
    if not candidates:
        raise ValueError("no correlations with MRI_Count could be computed")
    return max(candidates)[1]
