import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =====================================================================
# Shared helpers
# =====================================================================

def get_dynamic_bins(data_list, bin_rule, fallback=50):
    """Resolve a bin spec against actual data.

    bin_rule can be an explicit array/int (passed straight through), or one
    of the strings "linspace" / "arange", in which case bins are computed
    from the min/max of data_list.
    """
    if not isinstance(bin_rule, str):
        return bin_rule
    if not data_list:
        return fallback

    if bin_rule == "linspace":
        return np.linspace(min(data_list), max(data_list), 51)
    if bin_rule == "arange":
        return np.arange(0, int(max(data_list)) + 1, 1)
    return fallback


# =====================================================================
# prepare_* : DataFrame -> plain lists/dicts/arrays (no plotting here)
# =====================================================================

def prepare_category_counts(df, column, categories):
    """Counts of `column` for each of `categories`, in that order."""
    return [int((df[column] == cat).sum()) for cat in categories]


def prepare_probabilities_by_category(df, group_col, prob_col, categories):
    """dict: category -> list of prob_col values, for the overlaid prob plots."""
    return {
        cat: df.loc[df[group_col] == cat, prob_col].dropna().tolist()
        for cat in categories
    }


def prepare_energy_spectrum_data(df, mc_processes, states1, feature="incident_energy"):
    """dict: mc_process -> l1_state -> list of feature values.

    Replaces the old metrics[proc][l1][l2][feature] walk — the old code
    flattened across l2 anyway (stacking was only ever by L1 status), so
    this is just a two-level groupby of the flat DataFrame.
    """
    out = {proc: {l1: [] for l1 in states1} for proc in mc_processes}
    subset = df[df["mc_process"].isin(mc_processes) & df["l1_state"].isin(states1)]
    for (proc, l1), group in subset.groupby(["mc_process", "l1_state"], observed=True):
        out[proc][l1] = group[feature].dropna().tolist()
    return out


def prepare_confusion_matrix(df, mc_col="mc_process", pred_col="l2_label",
                              true_order=("COMP", "PAIR", "PHOT", "OTHER"),
                              pred_order=("CO", "PA", "PH")):
    """Build the confusion matrix (rows=truth, cols=prediction) as a numpy array.

    Only meaningful for SIGNAL events with resolvable truth (matches the old
    accumulation, which only ever incremented for status == "SIGNAL").
    Any mc_process not in true_order[:-1] is folded into "OTHER".
    """
    sub = df.loc[df["l1_state"] == "SIGNAL", [mc_col, pred_col]].copy()
    known = set(true_order[:-1])
    sub["_truth_bucket"] = np.where(sub[mc_col].isin(known), sub[mc_col], "OTHER")

    cross = pd.crosstab(sub["_truth_bucket"], sub[pred_col])
    cross = cross.reindex(index=true_order, columns=pred_order, fill_value=0)
    return cross.to_numpy()


def prepare_particle_comparison_data(df, feature, calc=None):
    """true/miss/all dicts keyed by expected label (CO/PA/PH), for SIGNAL
    events with resolvable COMP/PAIR/PHOT truth.

    `calc`, if given, is a function of the filtered DataFrame returning a
    Series (used for derived ratio features like erat/nrat); rows where it
    yields NaN are dropped, matching the old np.divide(..., where=...) + 
    isnan-filter behavior.
    """
    process_map = {"PHOT": "PH", "PAIR": "PA", "COMP": "CO"}
    signal_df = df[(df["l1_state"] == "SIGNAL") & df["mc_process"].isin(process_map.keys())].copy()

    if calc is not None:
        signal_df[feature] = calc(signal_df)
    signal_df = signal_df.dropna(subset=[feature, "correct"])
    signal_df["category"] = signal_df["mc_process"].map(process_map)

    true_dict = {lbl: [] for lbl in process_map.values()}
    miss_dict = {lbl: [] for lbl in process_map.values()}

    for cat, group in signal_df.groupby("category", observed=True):
        true_dict[cat] = group.loc[group["correct"] == True, feature].tolist()
        miss_dict[cat] = group.loc[group["correct"] == False, feature].tolist()

    all_dict = {cat: true_dict[cat] + miss_dict[cat] for cat in true_dict}
    return true_dict, miss_dict, all_dict


def prepare_detector_comparison_data(df, feat_map):
    """true/miss/all dicts keyed by detector component (e.g. TRA/CAL/TOT).

    feat_map maps detector label -> column name, e.g.
    {"TRA": "E_tra", "CAL": "E_cal", "TOT": "E_tot"}.
    Categorization is the same correct/incorrect L2 split as the particle
    comparison, just reporting a different column per "category".
    """
    process_map = {"PHOT": "PH", "PAIR": "PA", "COMP": "CO"}
    signal_df = df[(df["l1_state"] == "SIGNAL") & df["mc_process"].isin(process_map.keys())].copy()
    signal_df = signal_df.dropna(subset=["correct"])
    correct_mask = signal_df["correct"] == True

    true_dict, miss_dict = {}, {}
    for det, col in feat_map.items():
        true_dict[det] = signal_df.loc[correct_mask, col].dropna().tolist()
        miss_dict[det] = signal_df.loc[~correct_mask, col].dropna().tolist()

    all_dict = {det: true_dict[det] + miss_dict[det] for det in feat_map}
    return true_dict, miss_dict, all_dict


# =====================================================================
# plot_* : pure plotting, takes lists/dicts/arrays only
# =====================================================================

def plot_bar_counts(categories, counts, colors, title, output_path, ylabel="Events"):
    """Simple bar chart of counts-per-category with value labels above each bar."""
    plt.figure(figsize=(8, 5))
    bars = plt.bar(categories, counts, color=colors, edgecolor="black", alpha=0.7)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.ylabel(ylabel)

    max_count = max(counts) if counts and max(counts) > 0 else 1
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + (max_count * 0.01), f"{yval}", ha="center", va="bottom")

    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_overlaid_probabilities(prob_dict, hist_configs, title, output_path, bins=50, value_range=(0, 1)):
    """Overlaid (stepfilled) probability histograms for a set of categories.

    hist_configs: list of (category_key, label, face_color, edge_color) tuples.
    """
    plt.figure(figsize=(9, 5))
    for cat, label, f_col, e_col in hist_configs:
        values = prob_dict.get(cat, [])
        if values:
            plt.hist(values, bins=bins, range=value_range, alpha=0.4, label=label, color=f_col, histtype="stepfilled", edgecolor=e_col)

    plt.title(title, fontsize=12, fontweight="bold")
    plt.yscale("log")
    plt.legend()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_stacked_energy_spectrum(spectrum_data, mc_processes, states1, output_path, bins=None,
                                  feature_label="Incident Energy (MeV)"):
    """1x3 stacked-histogram energy spectrum.

    spectrum_data: dict from prepare_energy_spectrum_data(), i.e.
    mc_process -> l1_state -> list of feature values.
    """
    l1_colors = {"UN": "gray", "MU": "crimson", "SIGNAL": "teal"}

    if bins is None:
        all_values = [v for proc in mc_processes for l1 in states1 for v in spectrum_data[proc][l1]]
        bins = np.linspace(min(all_values), max(all_values), 51) if all_values else 50

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, proc in zip(axes, mc_processes):
        stack_data, stack_labels, stack_colors = [], [], []
        for l1 in states1:
            values = spectrum_data[proc][l1]
            stack_data.append(values)
            stack_labels.append(f"{l1} ({len(values)})")
            stack_colors.append(l1_colors[l1])

        ax.hist(stack_data, bins=bins, stacked=True, color=stack_colors, label=stack_labels, edgecolor="black", linewidth=0.3, alpha=0.85)
        ax.set_title(proc, fontsize=11, fontweight="bold")
        ax.set_xlabel(feature_label, fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, loc="upper right")

    axes[0].set_ylabel("Number of Events", fontsize=10)
    fig.suptitle("Incident Energy Spectrum by MC Process (stacked by L1 status)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_confusion_matrix(confusion_matrix, output_path,
                           xlabels=("Compton", "Pair", "Photo"),
                           ylabels=("Compton", "Pair", "Photo", "Other")):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Render values onto the matrix squares: raw count, with the
    # row-normalized percentage (each true-label row sums to 100%)
    # underneath in parentheses.
    thresh = confusion_matrix.max() / 2. if confusion_matrix.max() > 0 else 1
    row_sums = confusion_matrix.sum(axis=1)
    for row in range(len(ylabels)):
        row_total = row_sums[row]
        for col in range(len(xlabels)):
            count = confusion_matrix[row, col]
            pct = (count / row_total * 100) if row_total > 0 else 0.0
            ax.text(col, row, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center",
                    color="white" if count > thresh else "black",
                    fontweight='bold')

    ax.set_title("Confusion Matrix (L2 Layer)", fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Process', fontweight='bold')
    ax.set_ylabel('True MC Label', fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_type_classification_comparison(true_by_category, miss_by_category, all_by_category, output_path,
                                         xlabel, title, bins=50, log_y=True, categories=None):
    """2x2 comparison layout with counts and misclassification ratio panels.

    Unchanged from the original — this function was already generic over
    plain dicts, so no pandas logic belongs in here.
    """
    if categories is None:
        categories = list(true_by_category.keys())

    palette = [
        ("royalblue", "darkblue"),
        ("forestgreen", "darkgreen"),
        ("darkorchid", "purple"),
    ]
    colors = {category: palette[idx % len(palette)] for idx, category in enumerate(categories)}

    def _plot_hist_panel(ax, data_by_category, panel_title):
        for category in categories:
            values = data_by_category.get(category, [])
            if not values:
                continue
            color, edgecolor = colors[category]
            ax.hist(
                values,
                bins=bins,
                alpha=0.45,
                color=color,
                edgecolor=edgecolor,
                histtype="stepfilled",
                label=f"{category} ({len(values)})",
            )

        ax.set_title(panel_title, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Number of Events", fontsize=10)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9)
        if log_y:
            ax.set_yscale("log")

    def _plot_ratio_panel(ax):
        for category in categories:
            true_values = true_by_category.get(category, [])
            miss_values = miss_by_category.get(category, [])
            if not true_values and not miss_values:
                continue

            combined_values = true_values + miss_values
            if not combined_values:
                continue

            # Must be binned in the same way, using the "true's" bins
            counts_true, bin_edges = np.histogram(true_values, bins=bins)
            counts_miss, _ = np.histogram(miss_values, bins=bin_edges)
            counts_total = counts_true + counts_miss
            ratios = np.divide(
                counts_miss,
                counts_total,
                out=np.zeros_like(counts_total, dtype=float),
                where=counts_total > 0,
            )
            centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            color, _ = colors[category]
            mean_misclassification = len(miss_values) / len(combined_values)

            # Binomial standard error PER BIN (uses that bin's own count).
            safe_counts = np.where(counts_total > 0, counts_total, 1)
            dy = np.sqrt(ratios * (1 - ratios) / safe_counts)
            dy = np.where(counts_total > 0, dy, 0.0)

            ax.errorbar(centers, ratios, dy, color=color, fmt=".", label=f"{category} (mean: {mean_misclassification:.2f})")
            ax.axhline(
                mean_misclassification,
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
            )

        ax.set_title("Misclassified / Total", fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Fraction Misclassified", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.28, wspace=0.22)
    ax_true = fig.add_subplot(gs[0, 0])
    ax_miss = fig.add_subplot(gs[0, 1])
    ax_all = fig.add_subplot(gs[1, 0])
    ax_ratio = fig.add_subplot(gs[1, 1])

    _plot_hist_panel(ax_true, true_by_category, "Correctly Classified")
    _plot_hist_panel(ax_miss, miss_by_category, "Misclassified")
    _plot_hist_panel(ax_all, all_by_category, "Correctly + Misclassified")
    _plot_ratio_panel(ax_ratio)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# =====================================================================
# make_* : high-level DataFrame -> saved figure, chaining prepare + plot.
# These are what you call directly from main().
# =====================================================================

def make_category_counts_plot(df, column, categories, colors, title, output_path, ylabel="Events"):
    '''
    Makes a bar plot of counts for specified categories in a given column of the DataFrame.
    Parameters:
    - df: pandas DataFrame containing the data.
    - column: str, the name of the column in df to count categories from.
    - categories: list of str, the categories to count in the specified column.
    - colors: list of str, colors for each category in the bar plot.
    - title: str, the title of the plot.
    - output_path: str, the file path to save the plot.
    - ylabel: str, the label for the y-axis (default is "Events").
    '''
    counts = prepare_category_counts(df, column, categories)
    plot_bar_counts(categories, counts, colors, title, output_path, ylabel=ylabel)


def make_probabilities_plot(df, group_col, prob_col, hist_configs, title, output_path, bins=50, value_range=(0, 1)):
    '''
    Makes an overlaid histogram plot of probability distributions for specified categories in a given column of the DataFrame.
    Parameters:
    - df: pandas DataFrame containing the data.
    - group_col: str, the name of the column in df to group by (e.g., l1_state or l2_label).
    - prob_col: str, the name of the column in df containing probability values.
    - hist_configs: list of tuples, each containing (category_key, label, face_color, edge_color) for the histogram.
    - title: str, the title of the plot.
    - output_path: str, the file path to save the plot.
    - bins: int or array-like, the number of bins or the bin edges for the histogram (default is 50).
    - value_range: tuple, the range of values for the histogram (default is (0, 1)).
    '''
    categories = [cfg[0] for cfg in hist_configs]
    prob_dict = prepare_probabilities_by_category(df, group_col, prob_col, categories)
    plot_overlaid_probabilities(prob_dict, hist_configs, title, output_path, bins=bins, value_range=value_range)


def make_energy_spectrum_plot(df, mc_processes, states1, output_path, feature="incident_energy", bins=None):
    '''
    Makes a stacked histogram plot of energy spectrum for specified MC processes and L1 states.
    Parameters:
    - df: pandas DataFrame containing the data.
    - mc_processes: list of str, the MC processes to include in the plot (e.g., ["COMP", "PAIR", "PHOT"]).
    - states1: list of str, the L1 states to include in the plot (e.g., ["UN", "MU", "SIGNAL"]).
    - output_path: str, the file path to save the plot.
    - feature: str, the name of the column in df containing the energy values (default is "incident_energy").
    - bins: int or array-like, the number of bins or the bin edges for the histogram (default is None, which computes bins dynamically).
    '''
    spectrum_data = prepare_energy_spectrum_data(df, mc_processes, states1, feature=feature)
    plot_stacked_energy_spectrum(spectrum_data, mc_processes, states1, output_path, bins=bins)


def make_confusion_matrix_plot(df, output_path, mc_col="mc_process", pred_col="l2_label"):
    '''
    Makes a confusion matrix plot for specified MC process and predicted label columns in the DataFrame.
    Parameters:
    - df: pandas DataFrame containing the data.
    - output_path: str, the file path to save the plot.
    - mc_col: str, the name of the column in df containing the true MC process labels (default is "mc_process").
    - pred_col: str, the name of the column in df containing the predicted labels (default is "l2_label").
    '''
    matrix = prepare_confusion_matrix(df, mc_col=mc_col, pred_col=pred_col)
    plot_confusion_matrix(matrix, output_path)


def make_particle_comparison_plot(df, feature, xlabel, title, output_path, bin_rule=50, log_y=True, calc=None):
    '''
    Makes a 2x2 comparison plot of correctly classified vs misclassified events for specified particle types (COMP, PAIR, PHOT).
    Parameters:
    - df: pandas DataFrame containing the data.
    - feature: str, the name of the column in df containing the feature values to compare (e.g., "incident_energy").
    - xlabel: str, the label for the x-axis.
    - title: str, the title of the plot.
    - output_path: str, the file path to save the plot.
    - bin_rule: int or array-like, the number of bins or the bin edges for the histogram (default is 50).
    - log_y: bool, whether to use a logarithmic scale for the y-axis (default is True).
    - calc: callable, an optional function that takes the filtered DataFrame and returns a Series of derived values for the feature (default is None).
    '''
    true_dict, miss_dict, all_dict = prepare_particle_comparison_data(df, feature, calc=calc)
    flat_data = sum(true_dict.values(), []) + sum(miss_dict.values(), [])
    bins = get_dynamic_bins(flat_data, bin_rule)
    plot_type_classification_comparison(
        true_by_category=true_dict, miss_by_category=miss_dict, all_by_category=all_dict,
        output_path=output_path, xlabel=xlabel, title=title, bins=bins, log_y=log_y,
    )


def make_detector_comparison_plot(df, feat_map, xlabel, title, output_path, bin_rule=50, log_y=False):
    '''
    Makes a 2x2 comparison plot of correctly classified vs misclassified events for specified detector components (e.g., TRA, CAL, TOT).
    Parameters:
    - df: pandas DataFrame containing the data.
    - feat_map: dict, mapping detector component labels to column names in df (e.g., {"TRA": "E_tra", "CAL": "E_cal", "TOT": "E_tot"}).
    - xlabel: str, the label for the x-axis.
    - title: str, the title of the plot.
    - output_path: str, the file path to save the plot.
    - bin_rule: int or array-like, the number of bins or the bin edges for the histogram (default is 50).
    - log_y: bool, whether to use a logarithmic scale for the y-axis (default is False).
    '''
    true_dict, miss_dict, all_dict = prepare_detector_comparison_data(df, feat_map)
    flat_data = sum(true_dict.values(), []) + sum(miss_dict.values(), [])
    bins = get_dynamic_bins(flat_data, bin_rule)
    plot_type_classification_comparison(
        true_by_category=true_dict, miss_by_category=miss_dict, all_by_category=all_dict,
        output_path=output_path, xlabel=xlabel, title=title, bins=bins, log_y=log_y,
        categories=list(feat_map.keys()),
    )