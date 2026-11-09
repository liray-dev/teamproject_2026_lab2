import pytest

from grader_contracts.pandas_tasks import TitanicInput
from pandas_tasks import analyze_titanic

SAMPLE_CSV = """PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"A, Mr. One",male,22,1,0,T1,7.25,,S
2,1,1,"B, Mrs. Two",female,38,1,0,T2,71.28,C85,C
3,1,3,"C, Miss. Three",female,,0,0,T3,7.92,,S
4,1,1,"D, Mrs. Four",female,35,1,0,T4,53.1,C123,
5,0,3,"E, Mr. Five",male,30,0,0,T5,8.05,,S
6,0,2,"F, Mr. Six",male,54,0,0,T6,13.0,,S
7,1,2,"G, Mrs. Seven",female,,1,0,T7,30.0,,S
"""


@pytest.fixture
def sample_summary(tmp_path):
    path = tmp_path / "titanic.csv"
    path.write_text(SAMPLE_CSV, encoding="utf-8")
    return analyze_titanic(TitanicInput(str(path)))


def test_row_count(sample_summary):
    assert sample_summary.row_count == 7


def test_missing_by_column(sample_summary):
    assert sample_summary.missing_by_column["Age"] == 2
    assert sample_summary.missing_by_column["Cabin"] == 5
    assert sample_summary.missing_by_column["Embarked"] == 1
    assert sample_summary.missing_by_column["Fare"] == 0
    assert len(sample_summary.missing_by_column) == 12


def test_adults_over_30_is_strict(sample_summary):
    assert sample_summary.adults_over_30_count == 3


def test_mean_age_ignores_missing(sample_summary):
    assert sample_summary.mean_age_by_pclass == pytest.approx({1: 36.5, 2: 54.0, 3: 26.0})


def test_survival_rate_by_pclass(sample_summary):
    assert sample_summary.survival_rate_by_pclass == pytest.approx({1: 1.0, 2: 0.5, 3: 1 / 3})


def test_class_keys_are_int(sample_summary):
    for mapping in (sample_summary.mean_age_by_pclass, sample_summary.survival_rate_by_pclass):
        assert all(type(key) is int for key in mapping)


def test_highest_fares(sample_summary):
    assert sample_summary.highest_fares == pytest.approx([71.28, 53.1, 30.0, 13.0, 8.05])
    assert all(type(fare) is float for fare in sample_summary.highest_fares)


def test_real_dataset():
    summary = analyze_titanic(TitanicInput("titanic.csv"))
    assert summary.row_count == 891
    assert summary.missing_by_column["Age"] == 177
    assert summary.missing_by_column["Cabin"] == 687
    assert summary.missing_by_column["Embarked"] == 2
    assert summary.adults_over_30_count == 305
    assert set(summary.mean_age_by_pclass) == {1, 2, 3}
    assert summary.mean_age_by_pclass[1] == pytest.approx(38.233, abs=1e-3)
    assert summary.survival_rate_by_pclass[1] == pytest.approx(0.6296, abs=1e-4)
    assert summary.highest_fares == sorted(summary.highest_fares, reverse=True)
    assert summary.highest_fares[0] == pytest.approx(512.3292)
    assert len(summary.highest_fares) == 5
