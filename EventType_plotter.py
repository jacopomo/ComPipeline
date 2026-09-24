import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
import numpy as np

def estimate_compton_pair_probability(compton_count, pair_count):
    """Estimate Compton/Pair probabilities and binomial standard errors.

    Only CO and PA events are included in the sample size. Other classifier
    outcomes, such as PH, MU, and UN, are intentionally excluded.
    """
    total_count = compton_count + pair_count
    if total_count <= 0:
        return {
            "compton_probability": np.nan,
            "compton_error": np.nan,
            "pair_probability": np.nan,
            "pair_error": np.nan,
            "compton_count": compton_count,
            "pair_count": pair_count,
            "total_count": total_count,
        }

    compton_probability = compton_count / total_count
    error = np.sqrt(compton_probability * (1 - compton_probability) / total_count)
    return {
        "compton_probability": compton_probability,
        "compton_error": error,
        "pair_probability": 1 - compton_probability,
        "pair_error": error,
        "compton_count": compton_count,
        "pair_count": pair_count,
        "total_count": total_count,
    }

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

    Shared by the L1 and L2 "counts" plots.
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

    Shared by the L1 and L2 "probabilities" plots.

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
    
    xlabels = ['Compton', 'Pair', 'Photo']
    ylabels = ['Compton', 'Pair', 'Photo', 'Other']
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Render values onto the matrix squares: raw count, with the
    # row-normalized percentage (each true-label row sums to 100%)
    # underneath in parentheses.
    thresh = confusion_matrix.max() / 2. if confusion_matrix.max() > 0 else 1
    row_sums = confusion_matrix.sum(axis=1)
    for row in range(4):
        row_total = row_sums[row]
        for col in range(3):
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
    
    matrix_path = output_path
    plt.savefig(matrix_path, dpi=300)
    plt.close()

def plot_energy_response_hist(measured_energy, incident_energy, output_path, bins=50):
    """Plot measured energy against Monte Carlo incident energy.

    Both energy sequences are expected in MeV and must contain one entry per
    selected SIGNAL event.
    """
    measured_energy = np.asarray(measured_energy, dtype=float)
    incident_energy = np.asarray(incident_energy, dtype=float)
    valid = np.isfinite(measured_energy) & np.isfinite(incident_energy)

    fig = plt.figure(figsize=(9, 8))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1, 4, 0.18),
        height_ratios=(4, 1),
        wspace=0.08,
        hspace=0.08,
    )
    ax = fig.add_subplot(grid[0, 1])
    ax_incident = fig.add_subplot(grid[0, 0], sharey=ax)
    ax_measured = fig.add_subplot(grid[1, 1], sharex=ax)
    cax = fig.add_subplot(grid[0, 2])

    if np.any(valid):
        histogram = ax.hist2d(
            measured_energy[valid],
            incident_energy[valid],
            bins=bins,
            cmap="viridis",
            cmin=1,
            norm=LogNorm(vmin=1),
        )
        fig.colorbar(histogram[3], cax=cax, label="Number of Events")
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        diagonal_min = max(x_min, y_min)
        diagonal_max = min(x_max, y_max)
        ax.plot(
            [diagonal_min, diagonal_max],
            [diagonal_min, diagonal_max],
            linestyle="--",
            color="white",
            linewidth=1.2,
            label="Perfect response",
        )
        x_edges, y_edges = histogram[1], histogram[2]
        ax_measured.hist(
            measured_energy[valid],
            bins=x_edges,
            color="steelblue",
            edgecolor="black",
            linewidth=0.4,
        )
        ax_measured.set_yscale("log")
        ax_incident.hist(
            incident_energy[valid],
            bins=y_edges,
            orientation="horizontal",
            color="darkorange",
            edgecolor="black",
            linewidth=0.4,
        )
        ax_incident.set_xscale("log")
    else:
        ax.text(0.5, 0.5, "No SIGNAL events", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Deposited Energy (MeV)", fontsize=10)
    ax.set_ylabel("Monte Carlo Incident Energy (MeV)", fontsize=10)
    ax.set_title("Energy Response for SIGNAL Events", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax_measured.set_xlabel("Deposited Energy (MeV)", fontsize=10)
    ax_measured.set_ylabel("Events", fontsize=10)
    ax_measured.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax_incident.set_xlabel("Events", fontsize=10)
    ax_incident.set_ylabel("Monte Carlo Incident Energy (MeV)", fontsize=10)
    ax_incident.grid(True, axis="x", linestyle="--", alpha=0.35)

    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax_incident.get_yticklabels(), visible=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_process_probability_vs_measured_energy(
    measured_energy_by_process,
    all_measured_energy,
    output_path,
    incident_energy_by_process=None,
    all_incident_energy=None,
    bins=50,
):
    """Plot true-process probabilities as a function of measured and MC energy.

    Each probability is estimated per energy bin as the number of events from
    that process divided by the number of all events in the bin. Solid curves
    use measured deposited energy; optional dotted curves use true MC initial
    energy.
    Consequently, events from processes not present in
    ``measured_energy_by_process`` (for example RAYL) make the three curves
    sum to less than one.
    """
    all_measured_energy = np.asarray(all_measured_energy, dtype=float)
    all_measured_energy = all_measured_energy[np.isfinite(all_measured_energy)]
    incident_energy_by_process = incident_energy_by_process or {}
    all_incident_energy = np.asarray(all_incident_energy if all_incident_energy is not None else [], dtype=float)
    all_incident_energy = all_incident_energy[np.isfinite(all_incident_energy)]

    fig, ax = plt.subplots(figsize=(9, 5))
    if all_measured_energy.size == 0:
        ax.text(0.5, 0.5, "No measured-energy events", ha="center", va="center", transform=ax.transAxes)
    else:
        all_energy = np.concatenate((all_measured_energy, all_incident_energy))
        _, bin_edges = np.histogram(all_energy, bins=bins)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        colors = {"COMP": "royalblue", "PAIR": "forestgreen", "PHOT": "darkorchid"}

        for process, color in colors.items():
            process_energy = np.asarray(measured_energy_by_process.get(process, []), dtype=float)
            process_energy = process_energy[np.isfinite(process_energy)]
            counts_total_measured, _ = np.histogram(all_measured_energy, bins=bin_edges)
            counts_process_measured, _ = np.histogram(process_energy, bins=bin_edges)
            measured_probabilities = np.divide(
                counts_process_measured,
                counts_total_measured,
                out=np.zeros_like(counts_process_measured, dtype=float),
                where=counts_total_measured > 0,
            )
            ax.plot(centers, measured_probabilities, marker=".", linewidth=1.5, label=f"{process} (measured)", color=color)

            if all_incident_energy.size > 0:
                process_incident_energy = np.asarray(incident_energy_by_process.get(process, []), dtype=float)
                process_incident_energy = process_incident_energy[np.isfinite(process_incident_energy)]
                counts_total_incident, _ = np.histogram(all_incident_energy, bins=bin_edges)
                counts_process_incident, _ = np.histogram(process_incident_energy, bins=bin_edges)
                incident_probabilities = np.divide(
                    counts_process_incident,
                    counts_total_incident,
                    out=np.zeros_like(counts_process_incident, dtype=float),
                    where=counts_total_incident > 0,
                )
                ax.plot(
                    centers,
                    incident_probabilities,
                    marker=".",
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.8,
                    label=f"{process} (true MC)",
                    color=color,
                )

        ax.set_xlim(bin_edges[0], bin_edges[-1])
        ax.set_ylim(0, 1.05)

    ax.set_xlabel("Measured Deposited Energy (MeV)", fontsize=10)
    ax.set_ylabel("Frequentist Process Probability", fontsize=10)
    ax.set_title("True MC Process Probability vs Measured Energy", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)