import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from correlation_tasks import FEATURES, TARGET, load_brain_data, split_by_gender

GENDER_COLORS = {"Male": "tab:blue", "Female": "tab:red"}
GENDER_TITLES = {"Male": "Мужчины", "Female": "Женщины"}


def plot_scatter(brains, output_dir: Path) -> Path:
    figure, axes = plt.subplots(1, len(FEATURES), figsize=(4 * len(FEATURES), 4), sharey=True)
    for axis, feature in zip(axes, FEATURES):
        for gender, color in GENDER_COLORS.items():
            group = brains[brains["Gender"] == gender]
            axis.scatter(group[feature], group[TARGET], color=color, alpha=0.7, label=GENDER_TITLES[gender])
        axis.set_xlabel(feature)
    axes[0].set_ylabel(TARGET)
    axes[0].legend()
    figure.suptitle(f"Зависимость {TARGET} от признаков")
    figure.tight_layout()
    return _save(figure, output_dir / "scatter_mri.png")


def plot_heatmaps(brains, output_dir: Path) -> Path:
    columns = [*FEATURES, TARGET]
    men, women = split_by_gender(brains)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, (gender, group) in zip(axes, (("Male", men), ("Female", women))):
        sns.heatmap(
            group[columns].corr(method="pearson"), ax=axis, annot=True, fmt=".2f",
            cmap="coolwarm", vmin=-1, vmax=1, square=True,
        )
        axis.set_title(f"{GENDER_TITLES[gender]} (n={len(group)})")
    figure.suptitle("Корреляции Пирсона")
    figure.tight_layout()
    return _save(figure, output_dir / "heatmap_correlations.png")


def _save(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "brainsize.txt"
    brains = load_brain_data(source)
    output_dir = Path("plots")
    for path in (plot_scatter(brains, output_dir), plot_heatmaps(brains, output_dir)):
        print(path)


if __name__ == "__main__":
    main()
