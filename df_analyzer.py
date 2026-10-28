import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MC_PROCESSES = ("COMP", "PAIR", "PHOT")
L2_LABELS = ("CO", "PA", "PH")
colors = ['royalblue', 'forestgreen', 'darkorchid', 'crimson', 'gray']

# Metadata mapping for the three known processes
process_config = {
    "COMP": {"l2": "CO", "name": "Compton", "color": colors[1]},
    "PAIR": {"l2": "PA", "name": "Pair", "color": colors[2]},
    "PHOT": {"l2": "PH", "name": "Photoelectric", "color": colors[3]},
}

def build_confusion_matrix(df):
    """Rows: true MC process (+ "OTHER"). Columns: predicted L2 label.

    Only SIGNAL events with resolvable MC truth are counted.
    """
    signal = df[(df["l1"] == "SIGNAL") & (df["mc_true"] != "UNKNOWN")]
    true_label = signal["mc_true"].where(signal["mc_true"].isin(MC_PROCESSES), "OTHER")
    matrix = pd.crosstab(true_label, signal["l2"])
    return matrix.reindex(index=[*MC_PROCESSES, "OTHER"], columns=L2_LABELS, fill_value=0)

def plot_confusion_matrix(confusion_matrix, output_path):
    # Extract numpy array and labels directly from the pandas DataFrame
    cm_array = confusion_matrix.values
    ylabels = list(confusion_matrix.index)
    xlabels = list(confusion_matrix.columns)

    n_rows, n_cols = cm_array.shape

    row_sums = cm_array.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm_array / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Render values onto the matrix squares (raw count + row-normalized percentage)
    thresh = 0.5

    for row in range(n_rows):
        row_total = row_sums[row, 0]
        for col in range(n_cols):
            count = cm_array[row, col]
            pct = (count / row_total * 100) if row_total > 0 else 0.0
            norm_val = cm_norm[row, col]
            ax.text(
                col,
                row,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color="white" if norm_val > thresh else "black",
                fontweight="bold",
        )

    ax.set_title("Confusion Matrix (L2 Layer)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Process", fontweight="bold")
    ax.set_ylabel("True MC Label", fontweight="bold")
    fig.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_bar_chart(df, column, output_path, color=colors, title=None, l1_type="ALL"):
    """Computes value counts from a DataFrame column and plots a bar chart

    with exact counts printed above each bar.
    """
    # Get counts and categories from the dataframe column

    valid_l1_types = ["SIGNAL", "UN", "MU"]

    if l1_type == "ALL":
        filtered_df = df
    elif l1_type in valid_l1_types:
        filtered_df = df[df["l1"] == l1_type]
    else:
        raise ValueError(f"Invalid l1_type: {l1_type}. Must be one of {valid_l1_types} or 'ALL'.")
    
    counts_series = filtered_df[column].value_counts()
    categories = counts_series.index.astype(str).tolist()
    counts = counts_series.values.tolist()

    plt.figure(figsize=(8, 5))
    bars = plt.bar(categories, counts, color=color, edgecolor="black", alpha=0.7)
    
    if title is None:
        title = f"Counts of {column} considering {l1_type} (L1) events"
    plt.title(title, fontsize=12, fontweight="bold")
    plt.ylabel("Counts", fontsize=11)
    plt.xticks(rotation=45, ha="right")

    # Calculate offset for printing values nicely above the bars
    max_count = max(counts) if counts and max(counts) > 0 else 1
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + (max_count * 0.01),
            yval,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_histogram(
    df,
    column,
    output_path=None,
    bins=50,
    color=colors[0],
    title=None,
    xlabel=None,
    ylabel=None,
    log_scale=False,
    ax=None,
    l1_type="ALL"
):
    """Plots a histogram of a DataFrame column.

    Supports an optional 'ax' argument for subplots.

    Ability to filter by L1 type (SIGNAL, UN, MU, or ALL) for the histogram.
    """
    # If no axis is provided, create a new standalone figure
    if ax is None:
        if output_path is None:
            raise ValueError("output_path must be provided when creating a stadalone histogram plot.")
        fig, ax = plt.subplots(figsize=(8, 5))
        created_fig = True
    else:
        created_fig = False
    
    valid_l1_types = ["SIGNAL", "UN", "MU"]

    if l1_type == "ALL":
        filtered_df = df
    elif l1_type in valid_l1_types:
        filtered_df = df[df["l1"] == l1_type]
    else:
        raise ValueError(f"Invalid l1_type: {l1_type}. Must be one of {valid_l1_types} or 'ALL'.")
    
    counts, edges, _ = ax.hist(filtered_df[column], bins=bins, color=color, edgecolor="black", alpha=0.7)
    total_count = int(counts.sum())

    if title is None:
        title = f"Histogram of {column}, considering {l1_type} (L1) events"
    ax.set_title(title, fontsize=12, fontweight="bold")
    
    if xlabel is None:
        xlabel = column
    ax.set_xlabel(xlabel, fontsize=11)

    if ylabel is None:
        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("Counts (log scale)", fontsize=11)
        else:
            ax.set_ylabel("Counts", fontsize=11)
    else:
        ax.set_ylabel(ylabel, fontsize=11)

    ax.text(
        0.95, 0.95,
        f"Total: {total_count}",    
        transform=ax.transAxes,
        ha="right", va="top", 
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.4", 
            facecolor="white", 
            edgecolor="gray", 
            alpha=0.8
            )
        )
    
    if created_fig:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    return counts, edges

def plot_misclassification_comparison(
    df,
    column,
    proc1="COMP",
    proc2="PAIR",
    output_path=None,
    bins=50,
    xlabel=None,
):
    """A generalized 2x2-subplot figure showing cross-misclassifications 
    between any two selected MC processes (e.g., COMP, PAIR, PHOT).
    """    

    if proc1 not in process_config or proc2 not in process_config:
        raise ValueError(f"Processes must be one of {list(process_config.keys())}")
    
    p1 = process_config[proc1]
    p2 = process_config[proc2]

    events_1 = df[(df["mc_true"] == proc1) & (df["l1"] == "SIGNAL")]
    events_2 = df[(df["mc_true"] == proc2) & (df["l1"] == "SIGNAL")]

    # Filter for cross-misclassifications
    mis_1 = events_1[events_1["l2"] == p2["l2"]]
    mis_2 = events_2[events_2["l2"] == p1["l2"]]

    if isinstance(bins, int):
        data_min = df[column].min()
        data_max = df[column].max()
        bin_edges = np.linspace(data_min, data_max, bins + 1)
    else:
        bin_edges = bins  

    # Total counts for normalization
    counts_1, _ = np.histogram(events_1[column], bins=bin_edges)
    counts_2, _ = np.histogram(events_2[column], bins=bin_edges)

    # Set up a 2-row, 2-column subplot figure
    fig, axes = plt.subplots(
            2, 2, 
            figsize=(12,8), 
            sharex="col", 
            sharey="row",
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0, "wspace": 0}
        )

    ax1, ax2 = axes[0, 0], axes[0, 1]  # Top row: Spectra
    ax3, ax4 = axes[1, 0], axes[1, 1]  # Bottom row: Thinner rates/residuals
    
    # Top left plot: True Compton misclassified as Pair
    mis_counts_1, _ = plot_histogram(
        mis_1,
        column,
        bins=bin_edges,
        color=p1["color"],
        title=f"Misclassified {p1['name']} events (tagged as {p2['name']}) vs {column}",
        xlabel="",
        ax=ax1,
        log_scale=True
    )
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Right plot: True Pair misclassified as Compton
    mis_counts_2, _ = plot_histogram(
        mis_2,
        column,
        bins=bin_edges,
        color=p2["color"],
        title=f"Misclassified {p2['name']} events (tagged as {p1['name']}) vs {column}",
        xlabel="",
        ylabel="",
        ax=ax2,
        log_scale=True
    )
    plt.setp(ax2.get_xticklabels(), visible=False)

    # Misclassification rates
    rate_1 = np.divide(
        mis_counts_1, counts_1, 
        out=np.zeros_like(mis_counts_1, dtype=float), 
        where=counts_1 != 0
    )
    
    rate_2 = np.divide(
        mis_counts_2, counts_2, 
        out=np.zeros_like(mis_counts_2, dtype=float), 
        where=counts_2 != 0
    )

    # Global population-weighted average rates
    avg_rate_1 = mis_counts_1.sum() / counts_1.sum() if counts_1.sum() > 0 else 0
    avg_rate_2 = mis_counts_2.sum() / counts_2.sum() if counts_2.sum() > 0 else 0
    
    if xlabel is None:
        xlabel = column

    # Left plot: Relative misclassification rate for process 1
    ax3.scatter(bin_edges[:-1], rate_1, color=p1["color"])
    ax3.axhline(avg_rate_1, color=p1["color"], linestyle="--", linewidth=1, alpha=0.7, label=f"Avg: {avg_rate_1:.3f}")
    ax3.set_xlabel(xlabel, fontsize=11)
    ax3.set_ylabel("Relative Misclassification Rate", fontsize=11)
    ax3.margins(y=0.25)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=8, frameon=True)

    # Right plot: Relative misclassification rate for process 2
    ax4.scatter(bin_edges[:-1], rate_2, color=p2["color"])
    ax4.axhline(avg_rate_2, color=p2["color"], linestyle="--", linewidth=1, alpha=0.7, label=f"Avg: {avg_rate_2:.3f}")
    ax4.set_xlabel(xlabel, fontsize=11)
    ax4.margins(y=0.25)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="upper right", fontsize=8, frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_classification_hist(
    df,
    column,
    output_path=None,
    process="COMP",
    bins=50,
    title=None,
    xlabel=None,
    ylabel=None,
    log_scale=False,
    ax=None,
):
    #check if process is valid
    if process not in process_config:
        raise ValueError(f"Process must be one of {list(process_config.keys())}")

    # If no axis is provided, create a new standalone figure
    if ax is None:
        if output_path is None:
            raise ValueError("output_path must be provided when creating a stadalone histogram plot.")
        fig, ax = plt.subplots(figsize=(8, 5))
        created_fig = True
    else:
        created_fig = False

    events = df[(df["mc_true"] == process) & (df["l1"] == "SIGNAL")]
    comp_events = events[events["l2"] == "CO"]
    pair_events = events[events["l2"] == "PA"]
    phot_events = events[events["l2"] == "PH"]

    ax.hist(comp_events[column], bins=bins, density=True,
            histtype="stepfilled", color=process_config["COMP"]["color"],
            alpha=0.5, label="Classified as Compton",
            edgecolor=process_config["COMP"]["color"]
        )
    
    ax.hist(pair_events[column], bins=bins, density=True,
            histtype="stepfilled", color=process_config["PAIR"]["color"],
            alpha=0.5, label="Classified as Pair",
            edgecolor=process_config["PAIR"]["color"]
        )
    
    ax.hist(phot_events[column], bins=bins, density=True,
            histtype="stepfilled", color=process_config["PHOT"]["color"],
            alpha=0.5, label="Classified as Photoelectric",
            edgecolor=process_config["PHOT"]["color"]
        )

    if title is None:
        title = f"Classification of {process}, considering {column}"
    ax.set_title(title, fontsize=12, fontweight="bold")


    if xlabel is None:
        xlabel = column
    ax.set_xlabel(xlabel, fontsize=11)

    if ylabel is None:
        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("Normalized counts (log scale)", fontsize=11)
        else:
            ax.set_ylabel("Normalized counts", fontsize=11)
    else:
        ax.set_ylabel(ylabel, fontsize=11)

    ax.legend(loc="upper right", fontsize=8, frameon=True)
    if created_fig:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

def plot_triple_classification_hist(
    df,
    column,
    output_path=None,
    bins=50,
    title=None,
    xlabel=None,
    log_scale=False,
    ax=None,
):

    # Set up a 1-row, 3-column subplot figure
    fig, axes = plt.subplots(
            1, 3, 
            figsize=(16,8), 
            sharey="row",
            gridspec_kw={"wspace": 0}
        )

    ax1, ax2, ax3 = axes[0], axes[1], axes[2]

    plot_classification_hist(
        df,
        column,
        output_path=None,
        process="COMP",
        bins=bins,
        title=f"Classification of Compton events, considering {column}",
        xlabel=xlabel,
        log_scale=log_scale,
        ax=ax1
    )

    plot_classification_hist(
        df,
        column,
        output_path=None,
        process="PAIR",
        bins=bins,
        title=f"Classification of Pair events, considering {column}",
        xlabel=xlabel,
        ylabel="",
        log_scale=log_scale,
        ax=ax2
    )
    plt.setp(ax2.get_yticklabels(), visible=False)

    plot_classification_hist(
        df,
        column,
        output_path=None,
        process="PHOT",
        bins=bins,
        title=f"Classification of Photoelectric events, considering {column}",
        xlabel=xlabel,
        ylabel="",
        log_scale=log_scale,
        ax=ax3
    )
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Event Classifier Pipeline for MEGAlib simulation files."
    )
    parser.add_argument(
        "-i", "--input", 
        type=str, 
        help="Path to the dataframe csv"
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        help="Path to directory where plots will be saved"
    )

    args = parser.parse_args()
    df = pd.read_csv(args.input)
    output_dir = args.output

    # All the plots
    ### Confusion Matrix
    matrix_df = build_confusion_matrix(df)
    plot_confusion_matrix(matrix_df, f"{output_dir}/confusion_matrix.png")
    print(f"OK: Confusion matrix plot saved to: {output_dir}/confusion_matrix.png")

    ### Bar Charts
    plot_bar_chart(df, "l1", f"{output_dir}/l1_counts.png", title = "L1 Layer Counts")
    print(f"OK: L1 counts plot saved to: {output_dir}/l1_counts.png")
    
    plot_bar_chart(df, "l2", f"{output_dir}/l2_counts.png", title = "L2 Layer Counts")
    print(f"OK: L2 counts plot saved to: {output_dir}/l2_counts.png")

    plot_bar_chart(df, "mc_true", f"{output_dir}/mc_true_counts.png", title = "MC True Counts considering SIGNAL events", l1_type="SIGNAL")
    print(f"OK: MC true counts plot saved to: {output_dir}/mc_true_counts.png")

    ### Histograms
    plot_histogram(df, "l1_prob", f"{output_dir}/l1_prob_histogram.png", bins=20, title = "L1 Probability Histogram")
    print(f"OK: L1 probability histogram plot saved to: {output_dir}/l1_prob_histogram.png")

    plot_histogram(df, "l2_prob", f"{output_dir}/l2_prob_histogram.png", bins=20, title = "L2 Probability Histogram")
    print(f"OK: L2 probability histogram plot saved to: {output_dir}/l2_prob_histogram.png")

    plot_histogram(df, "Ein", f"{output_dir}/Ein_histogram.png", bins=range(51), title = "Incident Energy Spectrum of SIGNAL events", xlabel="Incident Energy [MeV]", log_scale=True, l1_type="SIGNAL")
    print(f"OK: Incident energy spectrum histogram plot saved to: {output_dir}/Ein_histogram.png")

    plot_histogram(df, "E_dep_tot", f"{output_dir}/Edep_histogram.png", bins=range(51), title = "Total Deposited Energy Spectrum of SIGNAL events", xlabel="Deposited Energy [MeV]", log_scale=True, l1_type="SIGNAL")
    print(f"OK: Total deposited energy spectrum histogram plot saved to: {output_dir}/Edep_histogram.png")

    ### Misclassification plots
    plot_misclassification_comparison(df, "Ein", proc1="COMP", proc2="PAIR", output_path=f"{output_dir}/CoPa_misclassification_Ein.png", bins=range(51), xlabel="Incident Energy [MeV]")
    print(f"OK: Compton-Pair misclassification histogram plot saved to: {output_dir}/CoPa_misclassification_Ein.png")
    
    #plot_misclassification_comparison(df, "Ein", proc1="COMP", proc2="PHOT", output_path=f"{output_dir}/CoPh_misclassification_Ein.png", bins=range(51), xlabel="Incident Energy [MeV]")
    #print(f"OK: Compton-Photoelectric misclassification histogram plot saved to: {output_dir}/CoPa_misclassification_Ein.png")
    #plot_misclassification_comparison(df, "Ein", proc1="PAIR", proc2="PHOT", output_path=f"{output_dir}/PaPh_misclassification_Ein.png", bins=range(51), xlabel="Incident Energy [MeV]")
    #print(f"OK: Pair_Photoelectric misclassification histogram plot saved to: {output_dir}/CoPa_misclassification_Ein.png")
    
    # Classification type histograms
    #plot_classification_hist(df, "Ein", output_path=f"{output_dir}/classification_hist_Ein.png", process="COMP", bins=range(51), xlabel="Incident Energy [MeV]", log_scale=False)
    #print(f"OK: Classification histogram for Compton events saved to: {output_dir}/classification_hist_Ein.png")
    
    plot_triple_classification_hist(df, "Ein", output_path=f"{output_dir}/triple_classification_hist_Ein.png", bins=range(51), xlabel="Incident Energy [MeV]", log_scale=False)
    print(f"OK: Triple classification histogram plot saved to: {output_dir}/triple_classification_hist_Ein.png")
    print(f"Done! All plots have been saved to: {output_dir}")

    