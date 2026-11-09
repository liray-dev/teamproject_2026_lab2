import math

import pytest
from scipy import stats

from correlation_tasks import FEATURES, TARGET, analyze_brain_correlations, load_brain_data
from grader_contracts.correlation_tasks import BrainDataInput

SAMPLE_ROWS = [
    "Gender\tFSIQ\tVIQ\tPIQ\tWeight\tHeight\tMRI_Count",
    "Female\t100\t100\t100\t120\t60.0\t800",
    "Female\t110\t105\t102\t130\t62.0\t900",
    "Female\t120\t108\t90\t140\t64.0\t1000",
    "Male\t100\t90\t110\tNA\t70.0\t1000",
    "Male\t110\t95\t100\t160\t71.0\t950",
    "Male\t120\t130\t90\t170\tNA\t900",
    "Male\t130\t100\t80\t180\t73.0\t850",
    "",
]


@pytest.fixture
def sample_summary(tmp_path):
    path = tmp_path / "brainsize.txt"
    path.write_text("\r\n".join(SAMPLE_ROWS), encoding="utf-8")
    return analyze_brain_correlations(BrainDataInput(str(path)))


def test_group_counts(sample_summary):
    assert sample_summary.women_count == 3
    assert sample_summary.men_count == 4


def test_feature_keys(sample_summary):
    assert tuple(sample_summary.men_mri_correlation) == FEATURES
    assert tuple(sample_summary.women_mri_correlation) == FEATURES
    assert TARGET not in sample_summary.men_mri_correlation


def test_women_perfect_correlations(sample_summary):
    assert sample_summary.women_mri_correlation["FSIQ"] == pytest.approx(1.0)
    assert sample_summary.women_mri_correlation["Weight"] == pytest.approx(1.0)
    assert sample_summary.women_mri_correlation["Height"] == pytest.approx(1.0)


def test_men_correlation_skips_missing(sample_summary):
    assert sample_summary.men_mri_correlation["FSIQ"] == pytest.approx(-1.0)
    assert sample_summary.men_mri_correlation["Weight"] == pytest.approx(-1.0)
    assert not math.isnan(sample_summary.men_mri_correlation["Height"])


def test_strongest_feature_is_feature(sample_summary):
    assert sample_summary.strongest_mri_feature in FEATURES


@pytest.fixture(scope="module")
def real_summary():
    return analyze_brain_correlations(BrainDataInput("brainsize.txt"))


def test_real_counts(real_summary):
    assert real_summary.men_count == 20
    assert real_summary.women_count == 20


@pytest.mark.parametrize("gender", ["Male", "Female"])
def test_real_matches_scipy(real_summary, gender):
    brains = load_brain_data("brainsize.txt")
    group = brains[brains["Gender"] == gender]
    result = real_summary.men_mri_correlation if gender == "Male" else real_summary.women_mri_correlation
    for feature in FEATURES:
        pair = group[[feature, TARGET]].dropna()
        expected = stats.pearsonr(pair[feature], pair[TARGET]).statistic
        assert result[feature] == pytest.approx(expected)


def test_real_strongest_feature(real_summary):
    combined = {
        (gender, feature): abs(value)
        for gender, correlation in (("m", real_summary.men_mri_correlation), ("f", real_summary.women_mri_correlation))
        for feature, value in correlation.items()
    }
    assert real_summary.strongest_mri_feature == max(combined, key=combined.get)[1]
