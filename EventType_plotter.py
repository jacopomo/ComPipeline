import matplotlib.pyplot as plt 
import numpy as np

def plot_type_classification_comparison(true_by_category, miss_by_category, all_by_category, output_path, xlabel, title, bins=50, log_y=True, categories=None):
    """Create a 2x2 comparison layout with counts and misclassification ratio panels."""

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

            # Binomial standard error PER BIN (uses that bin's own count, not
            # the dataset-wide total). Previously this used len(combined_values)
            # for every bin, which understates the uncertainty in sparsely
            # populated bins and overstates it in densely populated ones.
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


def plot_bar_counts(categories, counts, colors, title, output_path, ylabel="Events"):
    """Simple bar chart of counts-per-category with value labels above each bar.

    Shared by the L1 and L2 "counts" plots, which were previously two
    hand-copied blocks of identical code.
    """
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

    Shared by the L1 and L2 "probabilities" plots, which were previously two
    hand-copied blocks of identical code.

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


def plot_stacked_energy_spectrum(metrics, mc_processes, states1, output_path, bins=None):
    """1x3 stacked-histogram energy spectrum.

    One subplot per MC process (COMP, PAIR, PHOT). Within each subplot, the
    incident-energy histogram is stacked by L1 status (UN, MU, SIGNAL), so
    each bin shows how that process's true events split across the L1
    classifier's three possible outcomes.
    """
    l1_colors = {"UN": "gray", "MU": "crimson", "SIGNAL": "teal"}

    # If bins is not provided, compute it from the min/max of all incident energies across all processes and L1 states.
    if bins is None:
        all_energies = [
            e
            for proc in mc_processes
            for l1 in states1
            for l2 in metrics[proc][l1]
            for e in metrics[proc][l1][l2]["incident_energy"]
        ]
        bins = np.linspace(min(all_energies), max(all_energies), 51) if all_energies else 50

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, proc in zip(axes, mc_processes):
        stack_data, stack_labels, stack_colors = [], [], []
        for l1 in states1:
            energies = [e for l2 in metrics[proc][l1] for e in metrics[proc][l1][l2]["incident_energy"]]
            stack_data.append(energies)
            stack_labels.append(f"{l1} ({len(energies)})")
            stack_colors.append(l1_colors[l1])

        ax.hist(stack_data, bins=bins, stacked=True, color=stack_colors, label=stack_labels, edgecolor="black", linewidth=0.3, alpha=0.85)
        ax.set_title(proc, fontsize=11, fontweight="bold")
        ax.set_xlabel("Incident Energy (MeV)", fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, loc="upper right")

    axes[0].set_ylabel("Number of Events", fontsize=10)
    fig.suptitle("Incident Energy Spectrum by MC Process (stacked by L1 status)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_confusion_matrix(confusion_matrix, output_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    labels = ['Compton (CO)', 'Pair (PA)', 'Photo (PH)']
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Render values onto the matrix squares: raw count, with the
    # row-normalized percentage (each true-label row sums to 100%)
    # underneath in parentheses.
    thresh = confusion_matrix.max() / 2. if confusion_matrix.max() > 0 else 1
    row_sums = confusion_matrix.sum(axis=1)
    for row in range(3):
        row_total = row_sums[row]
        for col in range(3):
            count = confusion_matrix[row, col]
            pct = (count / row_total * 100) if row_total > 0 else 0.0
            ax.text(col, row, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center",
                    color="white" if count > thresh else "black",
                    fontweight='bold')
    
    ax.set_title("Confusion Matrix 3x3 (L2 Layer)", fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True MC Label', fontweight='bold')
    fig.tight_layout()
    
    matrix_path = output_path
    plt.savefig(matrix_path, dpi=300)
    plt.close()
