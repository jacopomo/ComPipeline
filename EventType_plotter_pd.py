import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_dynamic_bins(data_list, bin_rule, fallback=50):
    ''' Determine bin edges for histograms based on a specified rule or fallback to a default number of bins.
    Parameters:
    - data_list: list or array-like, the data to determine the bin edges from.
    - bin_rule: str or int, the rule for determining bins. If str, can be "linspace" or "arange". If int, it specifies the number of bins.
    - fallback: int, the default number of bins to use if bin_rule is not recognized or data_list is empty (default is 50).
    Returns:
    - numpy array of bin edges if bin_rule is recognized and data_list is not empty; otherwise, returns the fallback value.
    '''
    if not isinstance(bin_rule, str):
        return bin_rule
    if not data_list:
        return fallback

    if bin_rule == "linspace":
        return np.linspace(int(np.floor(min(data_list))), int(np.ceil(max(data_list))), 51)
    if bin_rule == "arange":
        return np.arange(0, int(np.ceil(max(data_list))) + 1, 1)
    return fallback

def plot_bar_counts(df, column, colors, title, output_path, ylabel="Events"):
    ''' Simple bar plot of counts for a categorical column in a DataFrame.
    Parameters:
    - df: pandas DataFrame containing the data.
    - column: str, the name of the column in df to count categories from.
    - colors: list of str, colors for each category in the bar plot.
    - title: str, the title of the plot.
    - output_path: str, the file path to save the plot.
    - ylabel: str, the label for the y-axis (default is "Events").
    '''
    
    plt.figure(figsize=(8, 5))
    counts = df[column].value_counts()
    ax = counts.plot(kind="bar", color=colors, edgecolor="black", alpha=0.7)
    ax.bar_label(ax.containers[0], padding=3, fontsize=10)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.ylabel(ylabel)
    plt.xticks(rotation=0)

    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_overlaid_probabilities(df, val_col, cat_col, title, output_path, bins=50):
    ''' Overlaid histogram of probability distributions for different categories in a DataFrame.
    Parameters:
    - df: pandas DataFrame containing the data.
    - val_col: str, the name of the column in df containing probability values.
    - cat_col: str, the name of the column in df containing category labels.
    - title: str, the title of the plot.
    - output_path: str, the file path to save the plot.
    - bins: int or array-like, the number of bins or the bin edges for the histogram (default is 50).
    '''
    plt.figure(figsize=(9, 5))

    for category, group_df in df.groupby(cat_col):
            plt.hist(
                group_df[val_col], 
                bins=bins, 
                histtype="stepfilled", 
                alpha=0.4,            
                edgecolor="black",
                linewidth=1.5,
                label=str(category)
            )

    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel(f"Probability ({val_col})", fontsize=10)
    plt.ylabel("Event Counts", fontsize=10)
    plt.yscale("log")
    plt.legend(title=cat_col, frameon=True)
    plt.xlim(0, 1.1) 

    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_stacked_energy_spectrum(df, output_path, bins=None, feature_label="Incident Energy (MeV)"):
    ''' Stacked histogram of incident energy spectrum by MC process, stacked by L1 status.
    Parameters:
    - df: pandas DataFrame containing the data with columns "mc_process", "l1_output", and "incident_energy".
    - output_path: str, the file path to save the plot.
    - bins: int or array-like, the number of bins or the bin edges for the histogram (default is None, which computes bins dynamically).
    - feature_label: str, the label for the x-axis (default is "Incident Energy (MeV)").
    '''

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    processes = ["COMP", "PHOT", "PAIR"]
    l1_categories = ["UN", "MU", "SIGNAL"]
    colors = ["gray", "crimson", "teal"]

    if bins is None:
        # 1 MeV bins from 0 to the largest incident energy in the DataFrame
        bins = np.arange(0, df["incident_energy"].max() + 1, 1)

    for i, proc in enumerate(processes):
        ax = axes[i]
        
        # Filter the DataFrame for this specific physical process
        df_proc = df[df["mc_process"] == proc]
        
        # Split the incident energy column into 3 datasets based on L1 output
        datasets_to_stack = [
            df_proc[df_proc["l1_output"] == cat]["incident_energy"] 
            for cat in l1_categories
        ]
        
        # Draw the stacked histogram
        ax.hist(
            datasets_to_stack, 
            bins=bins, 
            stacked=True, 
            label=l1_categories, 
            color=colors, 
            edgecolor="black",
            linewidth=0.3, 
            alpha=0.7
        )

        # Title and axis styling
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
    '''Confusion matrix plot with counts and row-normalized percentages.
    Parameters:
    - confusion_matrix: 2D numpy array of shape (len(ylabels), len(xlabels)), representing the confusion matrix.
    - output_path: str, the file path to save the plot.
    - xlabels: tuple of str, labels for the x-axis (predicted classes) (default: ("Compton", "Pair", "Photo")).
    - ylabels: tuple of str, labels for the y-axis (true classes) (default: ("Compton", "Pair", "Photo", "Other")).
    '''

    # Row-normalize for coloring
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    cm_percent = np.divide(
        confusion_matrix,
        row_sums,
        out=np.zeros_like(confusion_matrix, dtype=float),
        where=row_sums != 0
    ) * 100

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot percentages
    im = ax.imshow(
        cm_percent,
        interpolation='nearest',
        cmap=plt.cm.Blues,
        vmin=0,
        vmax=100
    )

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized percentage")
    cbar.set_ticks(np.linspace(0, 100, 6))
    cbar.set_ticklabels([f"{int(x)}%" for x in np.linspace(0, 100, 6)])

    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Threshold based on percentages
    thresh = 50

    for row in range(len(ylabels)):
        for col in range(len(xlabels)):
            count = confusion_matrix[row, col]
            pct = cm_percent[row, col]

            ax.text(
                col, row,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color="white" if cm_percent[row, col] > thresh else "black",
                fontweight="bold"
            )

    ax.set_title("Confusion Matrix (L2 Layer)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Process", fontweight="bold")
    ax.set_ylabel("True MC Label", fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_particle_comparison_plot(
    df, 
    feature_col, 
    binning_strategy=50, 
    log_y=True, 
    output_path=None
):
    ''' Generates a 2x2 comparison plot of correctly classified vs misclassified events for specified particle types (COMP, PAIR, PHOT).
    1. Top-left: Histogram of correctly classified events for each particle type.
    2. Top-right: Histogram of misclassified events for each particle type.
    3. Bottom-left: Histogram of all events (correct + misclassified) for each particle type.
    4. Bottom-right: Ratio of misclassified to total events for each particle type, with error bars representing the binomial standard error.
    5. Each histogram uses a unified binning strategy across all particle types.
    
    Parameters:
    - df: The pandas DataFrame
    - feature_col: The continuous numerical column to plot on the X-axis
    - binning_strategy: str or int, the rule for determining bins. If str, can be "linspace" or "arange". If int, it specifies the number of bins (default 50)
    - log_y: bool, logarithmic scale for the vertical axis (default: True)
    - output_path: str, the file path to save the plot. If None, shows the plot instead (default: None)
    '''
    categories=["COMP","PAIR","PHOT"]
    colors={
                "COMP": ("royalblue", "darkblue"),
                "PAIR": ("forestgreen", "darkgreen"),
                "PHOT": ("darkorchid", "purple")
            }


    bin_edges = get_dynamic_bins(df[feature_col].dropna().tolist(), binning_strategy, fallback=np.arange(0,51,1))
    bin_edges = np.asarray(bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Set up 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_true, ax_miss = axes[0, 0], axes[0, 1]
    ax_all, ax_ratio = axes[1, 0], axes[1, 1]

    # Helper function to plot a single histogram panel
    def _plot_hist_panel(ax, sub_df, panel_title):
        for cat in categories:
            # Get values for this category in this subset
            values = sub_df[sub_df["mc_process"] == cat][feature_col].dropna()
            color, edgecolor = colors[cat]
   
            ax.hist(
                values,
                bins=bin_edges,
                alpha=0.4,
                color=color,
                edgecolor=edgecolor,
                histtype="stepfilled",
                label=f"{cat} ({len(values)})",
            )
        ax.set_title(panel_title, fontsize=11, fontweight="bold")
        ax.set_xlabel(feature_col, fontsize=10)
        ax.set_ylabel("Number of Events", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9)
        if log_y:
            ax.set_yscale("log")

    # Filter the dataframes using boolean masking
    df_true = df[df["classification_status"] == True]
    df_miss = df[df["classification_status"] == False]  # False represents misclassified
    df_tot = pd.concat([df_true, df_miss])

    # Plot the 3 Histograms
    _plot_hist_panel(ax_true, df_true, "Correctly Classified")
    _plot_hist_panel(ax_miss, df_miss, "Misclassified")
    _plot_hist_panel(ax_all, df_tot, "All Events (Correct + Misclassified)")

    # Plot the Ratio / Errorbar Panel 
    for cat in categories:
        # Extract features for this category
        true_vals = df_true[df_true["mc_process"] == cat][feature_col].dropna()
        miss_vals = df_miss[df_miss["mc_process"] == cat][feature_col].dropna()
        total_len = len(true_vals) + len(miss_vals)
        
        if total_len == 0:
            continue

        # Bin counts
        counts_true, _ = np.histogram(true_vals, bins=bin_edges)
        counts_miss, _ = np.histogram(miss_vals, bins=bin_edges)
        counts_total = counts_true + counts_miss

        # Ratio calculation (avoiding division by zero)
        ratios = np.divide(
            counts_miss,
            counts_total,
            out=np.zeros_like(counts_total, dtype=float),
            where=counts_total > 0
        )
        
        # Binomial Standard Error calculation
        safe_counts = np.where(counts_total > 0, counts_total, 1)
        dy = np.sqrt(ratios * (1 - ratios) / safe_counts)
        dy = np.where(counts_total > 0, dy, 0.0)

        # Draw error bar and mean line
        color, _ = colors[cat]
        mean_misclass = len(miss_vals) / total_len
        
        ax_ratio.errorbar(
            bin_centers, ratios, yerr=dy, fmt=".", color=color, 
            label=f"{cat} (mean: {mean_misclass:.2%})"
        )
        ax_ratio.axhline(
            mean_misclass, color=color, linestyle="--", linewidth=1.2, alpha=0.8
        )

    ax_ratio.set_title("Misclassified / Total Ratio", fontsize=11, fontweight="bold")
    ax_ratio.set_xlabel(feature_col, fontsize=10)
    ax_ratio.set_ylabel("Fraction Misclassified", fontsize=10)
    ax_ratio.set_ylim(-0.05, 1.05)
    ax_ratio.grid(True, linestyle="--", alpha=0.5)
    ax_ratio.legend(loc="upper right", fontsize=9)

    # Clean layout wrapping
    plt.suptitle(f"Classification Comparison for '{feature_col}'", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()