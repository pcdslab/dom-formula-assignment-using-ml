import logging
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



DPI = 300
OUTCOME_COLORS = {
    "True annotation": "#8CC9C8",
    "False annotation": "#E6A0A8",
}
SAMPLE_NAMES = {
    "Table_Pahokee_River_Fulvic_Acid": "PPFA",
    "Table_Suwannee_River_Fulvic_Acid_2_v2": "SRFA2",
    "Table_Suwannee_River_Fulvic_Acid_3": "SRFA3",
}
PEAKLIST_FILES = {
    "PPFA": "PPFA_neg_8M_0.5s_5ppm_aFTk_PeakList_Rec.csv",
    "SRFA2": "SRFA2_neg_8M_0.5s_5ppm_aFTk_PeakList_Rec.csv",
    "SRFA3": "SRFA3_neg_8M_0.5s_5ppm_aFTk_PeakList_Rec.csv",
}
TEST_SET_FILES = {
    "PPFA": "Table_Pahokee_River_Fulvic_Acid.csv",
    "SRFA2": "Table_Suwannee_River_Fulvic_Acid_2_v2.csv",
    "SRFA3": "Table_Suwannee_River_Fulvic_Acid_3.csv",
}
COMPOSER_COUNTS = {"PPFA": 938, "SRFA2": 1431, "SRFA3": 1678}


def _formula_chons_key(formula):
    """Normalize formula strings to CHONS element-count tuples for uniqueness."""
    if not isinstance(formula, str) or not formula.strip():
        return None

    counts = {"C": 0, "H": 0, "O": 0, "N": 0, "S": 0}
    for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula.strip()):
        if element not in counts:
            counts[element] = 0
        counts[element] += int(count) if count else 1

    return tuple(sorted(counts.items()))


def _element_count(formula, target_element):
    """Return the count of one element in a molecular formula string."""
    if not isinstance(formula, str) or not formula.strip():
        return np.nan

    count_by_element = {}
    for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula.strip()):
        count_by_element[element] = count_by_element.get(element, 0) + (
            int(count) if count else 1
        )
    return count_by_element.get(target_element, 0)


def _formula_element_counts(formula):
    """Return parsed element counts for a molecular formula string."""
    if not isinstance(formula, str) or not formula.strip():
        return None

    count_by_element = {}
    for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula.strip()):
        count_by_element[element] = count_by_element.get(element, 0) + (
            int(count) if count else 1
        )
    return count_by_element


def _formula_class(formula):
    counts = _formula_element_counts(formula)
    if not counts:
        return None

    elements = {element for element, count in counts.items() if count > 0}
    if elements == {"C", "H", "O"}:
        return "CHO"
    if elements == {"C", "H", "O", "N"}:
        return "CHON"
    if elements == {"C", "H", "O", "N", "S"}:
        return "CHONS"
    if elements == {"C", "H", "O", "S"}:
        return "CHOS"
    return None


def _get_logger():
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("plotting")
    if not logger.handlers:
        handler = logging.FileHandler(log_dir / "plotting.log")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


plot_logger = _get_logger()


def _short_label(label):
    """Convert verbose configuration names into compact axis labels."""
    return (
        label.replace("Model-", "")
        .replace("_K", " · k=")
        .replace("_Euclidean", " · Euc")
        .replace("_Manhattan", " · Man")
    )


def _result_csvs(result_dir):
    result_path = Path(result_dir)
    if not result_path.is_dir():
        return []
    return sorted(result_path.glob("results_*.csv"))


def _sample_name(path):
    base = path.stem.removeprefix("results_").removesuffix(".csv")
    return SAMPLE_NAMES.get(base, base.replace("Table_", "").replace("_", " "))


def _collect_test_data(result_dirs, labels):
    outcome_rows = []
    sample_rows = []
    mass_rows = []

    for result_dir, full_label in zip(result_dirs, labels):
        label = _short_label(full_label)
        totals = {"Correct": 0, "New assignment": 0, "Wrong": 0}

        for path in _result_csvs(result_dir):
            try:
                frame = pd.read_csv(path)
            except Exception as exc:
                plot_logger.error("Could not read %s: %s", path, exc)
                continue

            correct = int(frame.get("predicted", pd.Series(dtype=float)).sum())
            new = int(frame.get("new_assignment", pd.Series(dtype=float)).sum())
            wrong = int(frame.get("wrong_prediction", pd.Series(dtype=float)).sum())
            total = len(frame)
            totals["Correct"] += correct
            totals["New assignment"] += new
            totals["Wrong"] += wrong

            sample_rows.append(
                {
                    "Model": label,
                    "Sample": _sample_name(path),
                    "Correct prediction (%)": 100 * correct / total if total else np.nan,
                }
            )

            if "mass_error_in_ppm" in frame:
                errors = pd.to_numeric(frame["mass_error_in_ppm"], errors="coerce")
                for value in errors[(errors >= 0) & (errors <= 1)].dropna():
                    mass_rows.append({"Model": label, "Mass error (ppm)": value})

        total_outcomes = sum(totals.values())
        for outcome, count in totals.items():
            outcome_rows.append(
                {
                    "Model": label,
                    "Outcome": outcome,
                    "Count": count,
                    "Percentage": 100 * count / total_outcomes if total_outcomes else 0,
                }
            )

    return pd.DataFrame(outcome_rows), pd.DataFrame(sample_rows), pd.DataFrame(mass_rows)


def _collect_peaklist_data(result_dirs, labels):
    rows = []
    sample_totals = {}

    for result_dir, full_label in zip(result_dirs, labels):
        label = _short_label(full_label)
        for sample, filename in PEAKLIST_FILES.items():
            path = Path(result_dir) / "peak_list" / filename
            if not path.is_file():
                continue
            try:
                frame = pd.read_csv(path)
            except Exception as exc:
                plot_logger.error("Could not read %s: %s", path, exc)
                continue

            total = len(frame)
            valid = int(frame.get("valid_prediction", pd.Series(dtype=float)).sum())
            if {"valid_prediction", "predicted_formula"}.issubset(frame.columns):
                unique_formulas = int(
                    frame.loc[frame["valid_prediction"].astype(bool), "predicted_formula"]
                    .dropna()
                    .map(_formula_chons_key)
                    .dropna()
                    .nunique()
                )
            else:
                unique_formulas = np.nan
            sample_totals.setdefault(sample, total)
            rows.append(
                {
                    "Model": label,
                    "Sample": sample,
                    "Total rows": total,
                    "Valid predictions": valid,
                    "Unique formulas": unique_formulas,
                    "Valid assignment (%)": 100 * valid / total if total else np.nan,
                }
            )

    for sample, count in COMPOSER_COUNTS.items():
        total = sample_totals.get(sample, 0)
        rows.append(
            {
                "Model": "Composer",
                "Sample": sample,
                "Total rows": total,
                "Valid predictions": count,
                "Unique formulas": count,
                "Valid assignment (%)": 100 * count / total if total else np.nan,
            }
        )

    return pd.DataFrame(rows)


def _summarize_median_error(mass_errors, model_order, n_bootstrap=1000):
    """Calculate median mass error and deterministic 95% bootstrap intervals."""
    rng = np.random.default_rng(42)
    rows = []
    for model in model_order:
        values = mass_errors.loc[
            mass_errors["Model"] == model, "Mass error (ppm)"
        ].dropna().to_numpy()
        if not len(values):
            continue
        samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
        bootstrap_medians = np.median(samples, axis=1)
        rows.append(
            {
                "Model": model,
                "Median": np.median(values),
                "CI low": np.percentile(bootstrap_medians, 2.5),
                "CI high": np.percentile(bootstrap_medians, 97.5),
            }
        )
    return pd.DataFrame(rows)


def _style_axis(ax, title, panel):
    ax.set_title(f"({panel})  {title}", loc="left", fontsize=22, fontweight="bold", pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#D9E1E8", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def _plot_assignment_radar(
    peaklists,
    plots_dir,
    value_column="Valid predictions",
    output_filename="valid_formula_assignments_radar.png",
):
    """Plot raw valid-formula counts for each sample as three radar panels."""
    previous_font_family = plt.rcParams["font.family"]
    previous_sans_serif = plt.rcParams["font.sans-serif"]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        }
    )

    configurations = ["k=1 · Euc", "k=1 · Man", "k=3 · Euc", "k=3 · Man"]
    families = ["L1", "L3", "L1-L3", "Synthetic"]
    colors = {
        "L1": "#FF7A7C",
        "L3": "#78BCE3",
        "L1-L3": "#F6CA6A",
        "Synthetic": "#6CC9AA",
    }

    model_rows = peaklists[peaklists["Model"] != "Composer"].copy()
    model_rows[["Family", "K", "Metric"]] = model_rows["Model"].str.split(
        " · ", n=2, expand=True
    )
    model_rows["Configuration"] = model_rows["K"] + " · " + model_rows["Metric"]

    angles = np.linspace(0, 2 * np.pi, len(configurations), endpoint=False)
    closed_angles = np.append(angles, angles[0])
    max_count = max(peaklists[value_column].dropna().max(), 1)
    radial_step = 1000
    radial_max = int(np.ceil(max_count / radial_step) * radial_step)
    radial_ticks = np.arange(radial_step, radial_max + 1, radial_step)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(24, 10),
        subplot_kw={"projection": "polar"},
    )
    for panel, (ax, sample) in enumerate(zip(axes, ["PPFA", "SRFA2", "SRFA3"]), start=1):
        sample_rows = model_rows[model_rows["Sample"] == sample]
        for family in families:
            family_rows = sample_rows[sample_rows["Family"] == family]
            values = (
                family_rows.set_index("Configuration")[value_column]
                .reindex(configurations)
                .fillna(0)
                .to_numpy()
            )
            closed_values = np.append(values, values[0])
            ax.plot(
                closed_angles,
                closed_values,
                color=colors[family],
                linewidth=3.2,
                marker="o",
                markersize=4.5,
                label=family,
            )
            ax.fill(closed_angles, closed_values, color=colors[family], alpha=0.055)

        composer = peaklists.loc[
            (peaklists["Model"] == "Composer") & (peaklists["Sample"] == sample),
            value_column,
        ]
        if not composer.empty and pd.notna(composer.iloc[0]):
            composer_values = np.repeat(composer.iloc[0], len(configurations) + 1)
            ax.plot(
                closed_angles,
                composer_values,
                color="#697780",
                linewidth=2.6,
                linestyle="--",
                label="Composer",
            )

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles)
        ax.set_xticklabels([])
        for index, (angle, configuration) in enumerate(zip(angles, configurations)):
            ax.text(
                angle,
                radial_max * 1.10,
                configuration,
                rotation=90 if index in (1, 3) else 0,
                rotation_mode="anchor",
                ha="center",
                va="center",
                fontsize=22,
                fontweight="bold",
                clip_on=False,
            )
        ax.set_ylim(0, radial_max)
        ax.set_yticks(radial_ticks)
        ax.set_yticklabels([])
        ax.set_rlabel_position(22.5)
        ax.grid(color="#4A4A4A", linewidth=0.7, alpha=0.75)
        ax.spines["polar"].set_color("#4A4A4A")
        ax.spines["polar"].set_linewidth(0.8)
        ax.set_title(
            f"({chr(96 + panel)})  {sample}",
            fontsize=22,
            fontweight="bold",
            pad=0,
            y=1.27,
            x=0.46,
        )
        ax.plot(
            [0.31, 0.61],
            [1.19, 1.19],
            transform=ax.transAxes,
            color="#263238",
            linewidth=2.2,
            clip_on=False,
        )
        for tick in radial_ticks:
            ax.text(
                np.deg2rad(22.5),
                tick,
                f"{tick:,}",
                color="#52616B",
                fontsize=22,
                fontweight="bold",
                ha="left",
                va="center",
                zorder=20,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.4},
            )

    handles, legend_labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=5,
        frameon=False,
        handlelength=2.6,
        prop={"size": 25, "weight": "bold"},
    )
    for legend_handle in legend.legend_handles:
        legend_handle.set_linewidth(4.8)
    fig.subplots_adjust(left=0.045, right=0.975, top=0.80, bottom=0.16, wspace=0.38)
    radar_path = plots_dir / output_filename
    fig.savefig(radar_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    plt.rcParams["font.family"] = previous_font_family
    plt.rcParams["font.sans-serif"] = previous_sans_serif
    return radar_path


def _oxygen_unique_formula_counts(formulas):
    unique_formulas = pd.Series(formulas).dropna().drop_duplicates()
    oxygen_counts = unique_formulas.map(lambda formula: _element_count(formula, "O")).dropna()
    if oxygen_counts.empty:
        return pd.Series(dtype=int)
    return oxygen_counts.astype(int).value_counts().sort_index()


def _oxygen_unique_formula_counts_by_class(formulas):
    unique_formulas = pd.Series(formulas).dropna().drop_duplicates()
    if unique_formulas.empty:
        return pd.DataFrame(columns=["Formula class", "O atoms", "Unique formulas"])

    formula_data = pd.DataFrame({"Formula": unique_formulas})
    formula_data["Formula class"] = formula_data["Formula"].map(_formula_class)
    formula_data["O atoms"] = formula_data["Formula"].map(lambda formula: _element_count(formula, "O"))
    formula_data = formula_data.dropna(subset=["Formula class", "O atoms"])
    if formula_data.empty:
        return pd.DataFrame(columns=["Formula class", "O atoms", "Unique formulas"])

    formula_data["O atoms"] = formula_data["O atoms"].astype(int)
    return (
        formula_data.groupby(["Formula class", "O atoms"])
        .size()
        .rename("Unique formulas")
        .reset_index()
    )


def _collect_k1_euclidean_oxygen_data(result_dirs, labels):
    rows = []
    for result_dir, full_label in zip(result_dirs, labels):
        if not full_label.endswith("_K1_Euclidean"):
            continue

        model = _short_label(full_label).split(" · ", maxsplit=1)[0]
        for sample, filename in PEAKLIST_FILES.items():
            path = Path(result_dir) / "peak_list" / filename
            if not path.is_file():
                continue

            try:
                frame = pd.read_csv(path)
            except Exception as exc:
                plot_logger.error("Could not read %s: %s", path, exc)
                continue

            if {"valid_prediction", "predicted_formula"}.issubset(frame.columns):
                valid_formulas = frame.loc[
                    frame["valid_prediction"].astype(bool), "predicted_formula"
                ]
                counts = _oxygen_unique_formula_counts_by_class(valid_formulas)
                for _, row in counts.iterrows():
                    rows.append(
                        {
                            "Sample": sample,
                            "Model": model,
                            "Formula class": row["Formula class"],
                            "O atoms": row["O atoms"],
                            "Unique formulas": row["Unique formulas"],
                        }
                    )

    data_dir = Path("data") / "DOM_testing_set"
    for sample, filename in TEST_SET_FILES.items():
        path = data_dir / filename
        if not path.is_file():
            continue

        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            plot_logger.error("Could not read %s: %s", path, exc)
            continue

        formula_column = "Chem. Formula" if "Chem. Formula" in frame.columns else "proposed_formula"
        if formula_column not in frame.columns:
            continue

        counts = _oxygen_unique_formula_counts_by_class(frame[formula_column])
        for _, row in counts.iterrows():
            rows.append(
                {
                    "Sample": sample,
                    "Model": "Composer",
                    "Formula class": row["Formula class"],
                    "O atoms": row["O atoms"],
                    "Unique formulas": row["Unique formulas"],
                }
            )

    return pd.DataFrame(rows)


def _plot_k1_euclidean_oxygen_unique_formulas(result_dirs, labels, plots_dir):
    oxygen_data = _collect_k1_euclidean_oxygen_data(result_dirs, labels)
    oxygen_path = plots_dir / "k1_euclidean_oxygen_unique_formulas.png"
    if oxygen_data.empty:
        plot_logger.warning("No K1 Euclidean oxygen-count data found; skipping %s", oxygen_path)
        return None

    model_order = ["Composer", "L1", "L3", "L1-L3", "Synthetic"]
    colors = {
        "Composer": "#697780",
        "L1": "#FF7A7C",
        "L3": "#78BCE3",
        "L1-L3": "#F6CA6A",
        "Synthetic": "#6CC9AA",
    }
    linestyles = {"Composer": "--"}

    formula_classes = ["CHO", "CHON", "CHONS", "CHOS"]
    samples = ["PPFA", "SRFA2", "SRFA3"]
    fig, axes = plt.subplots(
        len(formula_classes),
        len(samples),
        figsize=(18, 18.5),
        sharex=False,
        sharey="row",
    )
    legend_handles = None
    legend_labels = None
    for row_index, formula_class in enumerate(formula_classes):
        for column_index, sample in enumerate(samples):
            ax = axes[row_index, column_index]
            panel_data = oxygen_data[
                (oxygen_data["Sample"] == sample)
                & (oxygen_data["Formula class"] == formula_class)
            ]
            for model in model_order:
                model_data = panel_data[panel_data["Model"] == model].sort_values("O atoms")
                if model_data.empty:
                    continue

                ax.plot(
                    model_data["O atoms"],
                    model_data["Unique formulas"],
                    marker="o",
                    markersize=4.8,
                    linewidth=2.5 if model != "Composer" else 2.2,
                    linestyle=linestyles.get(model, "-"),
                    color=colors[model],
                    label=model,
                )

            if row_index == 0:
                ax.set_title(sample, fontsize=20, fontweight="bold", pad=12)
            if column_index == 0:
                ax.set_ylabel(f"{formula_class}\nUnique formulas", fontsize=16, fontweight="bold")
            if row_index == len(formula_classes) - 1:
                ax.set_xlabel("Number of oxygen atoms", fontsize=16)

            ax.tick_params(axis="both", labelsize=12)
            ax.grid(color="#D9E1E8", linewidth=0.8, alpha=0.85)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=5,
        frameon=False,
        fontsize=15,
    )
    fig.suptitle(
        "K1 Euclidean valid peaklist unique formulas by oxygen count",
        fontsize=24,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.945, bottom=0.085, hspace=0.38, wspace=0.16)
    fig.savefig(oxygen_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    oxygen_data.to_csv(plots_dir / "k1_euclidean_oxygen_unique_formulas.csv", index=False)
    return oxygen_path


def plot_publication_figures(result_dirs, labels, out_dir):
    plots_dir = Path(out_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    outcomes, sample_rates, mass_errors = _collect_test_data(result_dirs, labels)
    peaklists = _collect_peaklist_data(result_dirs, labels)
    model_order = [_short_label(label) for label in labels]

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

    fig_test = plt.figure(figsize=(15, 14))
    test_grid = fig_test.add_gridspec(
        2,
        2,
        height_ratios=[1.55, 0.75],
        width_ratios=[1.08, 0.92],
        hspace=0.60,
        wspace=0.08,
    )
    ax_outcomes = fig_test.add_subplot(test_grid[0, 0])
    ax_mass = fig_test.add_subplot(test_grid[0, 1], sharey=ax_outcomes)
    ax_regression = fig_test.add_subplot(test_grid[1, :])

    outcome_counts = outcomes.pivot(index="Model", columns="Outcome", values="Count")
    outcome_counts = outcome_counts.reindex(model_order).fillna(0)
    outcome_pivot = pd.DataFrame(index=outcome_counts.index)
    outcome_pivot["True annotation"] = (
        outcome_counts.get("Correct", 0) + outcome_counts.get("New assignment", 0)
    )
    outcome_pivot["False annotation"] = outcome_counts.get("Wrong", 0)
    outcome_pivot = outcome_pivot.div(outcome_pivot.sum(axis=1), axis=0) * 100
    left = np.zeros(len(outcome_pivot))
    for outcome in ["True annotation", "False annotation"]:
        values = outcome_pivot.get(outcome, pd.Series(0, index=outcome_pivot.index))
        bars = ax_outcomes.barh(
            outcome_pivot.index,
            values,
            left=left,
            label=outcome,
            color=OUTCOME_COLORS[outcome],
            edgecolor="white",
            linewidth=0.7,
            height=0.72,
        )
        for row, value in enumerate(values):
            if value < 1:
                continue
            if outcome == "False annotation" and value < 12:
                x_position = 99.4
                horizontal_alignment = "right"
            else:
                x_position = left[row] + value / 2
                horizontal_alignment = "center"
            ax_outcomes.text(
                x_position,
                row,
                f"{value:.1f}",
                ha=horizontal_alignment,
                va="center",
                color="#263238",
                fontsize=13,
                fontweight="bold",
            )
        left += values.to_numpy()
    ax_outcomes.set_xlim(0, 100)
    ax_outcomes.set_xlabel("Share of test observations (%)")
    ax_outcomes.set_ylabel("")
    ax_outcomes.invert_yaxis()
    ax_outcomes.tick_params(axis="x", labelsize=16)
    ax_outcomes.tick_params(axis="y", labelsize=17)
    ax_outcomes.xaxis.label.set_size(19)
    ax_outcomes.legend(
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        handlelength=1.6,
        fontsize=16,
    )
    _style_axis(ax_outcomes, "Test-set outcomes", "a")

    if not mass_errors.empty:
        error_summary = _summarize_median_error(mass_errors, model_order)
        error_summary = error_summary.set_index("Model").reindex(model_order).reset_index()
        y_positions = np.arange(len(error_summary))
        medians = error_summary["Median"].to_numpy()
        lower_errors = medians - error_summary["CI low"].to_numpy()
        upper_errors = error_summary["CI high"].to_numpy() - medians
        ax_mass.errorbar(
            medians,
            y_positions,
            xerr=np.vstack([lower_errors, upper_errors]),
            fmt="o",
            markersize=7,
            color="#5CA4A9",
            ecolor="#8A9BA5",
            elinewidth=3.6,
            capsize=5.5,
            capthick=3.2,
        )
        x_min = error_summary["CI low"].min()
        x_max = error_summary["CI high"].max()
        padding = max((x_max - x_min) * 0.18, 0.001)
        ax_mass.set_xlim(x_min - padding, x_max + padding * 2.2)
        for y, median in zip(y_positions, medians):
            ax_mass.annotate(
                f"{median:.3f}",
                (median, y),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=13,
                color="#263238",
            )
    ax_mass.set_xlabel("Median mass error (ppm)")
    ax_mass.set_ylabel("")
    ax_mass.tick_params(axis="x", labelsize=16)
    ax_mass.tick_params(axis="y", left=False, labelleft=False)
    ax_mass.xaxis.label.set_size(19)
    _style_axis(ax_mass, "Median mass error (95% CI)", "b")

    regression_categories = [
        "Formula assignment\naccuracy",
        "C accuracy",
        "H accuracy",
        "O accuracy",
        "S accuracy",
        "N accuracy",
    ]
    decision_tree = [86.5, 88.4, 89.5, 88.8, 96.6, 96.6]
    random_forest = [60.4, 77.6, 67.1, 80.4, 94.9, 98.2]
    x_positions = np.arange(len(regression_categories))
    bar_width = 0.36
    tree_bars = ax_regression.bar(
        x_positions - bar_width / 2,
        decision_tree,
        bar_width,
        label="Decision Tree Regressor",
        color="#AFCBE3",
        edgecolor="white",
        linewidth=0.8,
    )
    forest_bars = ax_regression.bar(
        x_positions + bar_width / 2,
        random_forest,
        bar_width,
        label="Random Forest Regressor",
        color="#D8BEDA",
        edgecolor="white",
        linewidth=0.8,
    )
    for bars in (tree_bars, forest_bars):
        ax_regression.bar_label(
            bars,
            labels=[f"{bar.get_height():.1f}" for bar in bars],
            padding=4,
            fontsize=17,
            fontweight="bold",
            color="#263238",
        )
    ax_regression.set_xticks(x_positions)
    ax_regression.set_xticklabels(regression_categories)
    ax_regression.set_ylim(0, 108)
    ax_regression.set_ylabel("Performance (%)", fontsize=19)
    ax_regression.tick_params(axis="x", labelsize=16)
    ax_regression.tick_params(axis="y", labelsize=16)
    ax_regression.legend(
        frameon=False,
        ncol=2,
        fontsize=16,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.03),
    )
    ax_regression.grid(axis="y", color="#D9E1E8", linewidth=0.7)
    ax_regression.spines[["top", "right"]].set_visible(False)
    ax_regression.set_axisbelow(True)
    ax_regression.set_title(
        "(c)  Formula-assignment and element-level accuracy",
        loc="left",
        fontsize=22,
        fontweight="bold",
        pad=14,
        y=1.22,
    )

    fig_test.subplots_adjust(left=0.20, right=0.98, top=0.96, bottom=0.07)
    test_path = plots_dir / "test_set_performance.png"
    fig_test.savefig(test_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig_test)
    fig_samples, (ax_samples, ax_peak) = plt.subplots(
        1,
        2,
        figsize=(15, 9),
        gridspec_kw={"wspace": 0.50},
    )

    sample_pivot = sample_rates.pivot_table(
        index="Model", columns="Sample", values="Correct prediction (%)", aggfunc="mean"
    ).reindex(model_order)
    sns.heatmap(
        sample_pivot,
        ax=ax_samples,
        cmap=sns.light_palette("#177E89", as_cmap=True),
        vmin=60,
        vmax=100,
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": 8.5},
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Correct prediction (%)", "shrink": 0.78, "pad": 0.03},
    )
    ax_samples.set_xlabel("")
    ax_samples.set_ylabel("")
    ax_samples.tick_params(axis="x", rotation=0)
    ax_samples.tick_params(axis="y", labelsize=8.5)
    ax_samples.set_title("(a)  Correct predictions by test sample", loc="left", fontweight="bold", pad=10)


    peak_order = ["Composer"] + model_order
    peak_pivot = peaklists.pivot_table(
        index="Model", columns="Sample", values="Valid assignment (%)", aggfunc="mean"
    ).reindex(peak_order)
    sns.heatmap(
        peak_pivot,
        ax=ax_peak,
        cmap=sns.light_palette("#E76F51", as_cmap=True),
        vmin=0,
        vmax=50,
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": 8.5},
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Valid assignment (%)", "shrink": 0.78, "pad": 0.03},
    )
    ax_peak.set_xlabel("")
    ax_peak.set_ylabel("")
    ax_peak.tick_params(axis="x", rotation=0)
    ax_peak.tick_params(axis="y", labelsize=8.5)
    ax_peak.set_title("(b)  Valid formula assignments by peak list", loc="left", fontweight="bold", pad=10)

    fig_samples.subplots_adjust(left=0.16, right=0.97, top=0.94, bottom=0.08)
    samples_path = plots_dir / "sample_assignment_performance.png"
    fig_samples.savefig(samples_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig_samples)

    radar_path = _plot_assignment_radar(peaklists, plots_dir)
    unique_radar_path = _plot_assignment_radar(
        peaklists,
        plots_dir,
        value_column="Unique formulas",
        output_filename="unique_formula_assignments_radar.png",
    )
    oxygen_path = _plot_k1_euclidean_oxygen_unique_formulas(result_dirs, labels, plots_dir)

    outcomes.to_csv(plots_dir / "metrics_test_results_combined.csv", index=False)
    peaklists.to_csv(plots_dir / "peaklist_combined_summary.csv", index=False)
    plot_logger.info(
        "Publication figures saved: %s, %s, %s, %s, and %s",
        test_path,
        samples_path,
        radar_path,
        unique_radar_path,
        oxygen_path,
    )
    return test_path, samples_path, radar_path, unique_radar_path, oxygen_path


def plot_testset_main(result_dirs, labels, out_dir):
    """Generate the two publication figures after test-set evaluation."""
    plot_logger.info("Starting publication plotting...")
    return plot_publication_figures(result_dirs, labels, out_dir)


def plot_peaklist_main(result_dirs, labels, out_dir):
    """Refresh the two publication figures after peak-list evaluation."""
    output_paths = plot_publication_figures(result_dirs, labels, out_dir)
    plot_logger.info("Finished publication plotting.")
    return output_paths
