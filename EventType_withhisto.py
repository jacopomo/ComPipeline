import argparse
from tqdm import tqdm
import time
import os
import sys

import ROOT as M
import torch
import pca
from PointNetModels.pointnet2C import PointNet
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
    
M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")


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
            mean_misclassification = len(miss_values) / len(combined_values) if combined_values else 0.0

            dy = np.sqrt(ratios*(1-ratios)/len(combined_values)) if combined_values else 0.0
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


class EventClassifierPipeline:

    def __init__(self, model_traced_path, onlyACDVeto=True, random_forest_path=None, lookup_path=None):
        if not onlyACDVeto:
            if random_forest_path is not None:
                print(f"Loading RF model from {random_forest_path}...")
                self.pca_classifier = pca.VegaClassifier(random_forest_path)
            elif lookup_path is not None:
                print(f"Loading lookup table from {lookup_path}...")
                self.pca_classifier = pca.SimpleClassifier(lookup_path)
            else:
                print("No BKG classifier provided — events will return 0.0")
                self.pca_classifier = None
        else:
            self.pca_classifier = None

        print(f"Loading TorchScript model from {model_traced_path}...")
        self.model = PointNet(add_nhits=False)
        state_dict = torch.load(model_traced_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()  

    def extract_hit_data(self, event, detId=None):
        nhits = event.GetNHTs()
        if nhits == 0:
            return None, 0

        data = torch.zeros([1, 4, nhits])
        if detId is None:
            for i in range(nhits):
                data[0, 0, i] = event.GetHTAt(i).GetPosition().X()
                data[0, 1, i] = event.GetHTAt(i).GetPosition().Y()
                data[0, 2, i] = event.GetHTAt(i).GetPosition().Z()
                data[0, 3, i] = event.GetHTAt(i).GetEnergy()
            return data, nhits

        n_selected = 0
        for i in range(nhits):
            if event.GetHTAt(i).GetDetectorType() == detId:
                data[0, 0, n_selected] = event.GetHTAt(i).GetPosition().X()
                data[0, 1, n_selected] = event.GetHTAt(i).GetPosition().Y()
                data[0, 2, n_selected] = event.GetHTAt(i).GetPosition().Z()
                data[0, 3, n_selected] = event.GetHTAt(i).GetEnergy()
                n_selected += 1

        if n_selected > 0:
            return data[:, :, :n_selected], n_selected
        return None, 0
    

    def signal_background_classifier(self, event, onlyACDVeto=True, thr=0.99):
        """First layer: Separates signal from background"""
        
        if not event:
            return "UN", 1.00
            
        nhits = event.GetNHTs()
        if nhits == 0:
            return "UN", 1.00

        nhits_ACD  = 0
        for i in range(nhits):
            if event.GetHTAt(i).GetDetectorType() == 4:
                nhits_ACD += 1
        if nhits_ACD > 0:  
            return "MU", 0.99  

        if not onlyACDVeto:
            data, _ = self.extract_hit_data(event, 1)
            prob = pca.analyze(data, event.GetTotalEnergyDeposit(), rf=self.pca_classifier, thr=thr)
            if prob > 0.5:
                return "SIGNAL", prob
            return "MU", 1.-prob
        else:
            return "SIGNAL", 1.00

    def type_of_signal(self, event):
        """Second layer: Checks if the event is a Photoelectric effect.
        If not, use PointNet to discriminate between Compton and Pair.
        """
        if not event:
            return "UN", 1.00
        
        nhits = event.GetNHTs()
                
        if nhits == 0:
            return "UN", 1.00

        # 1.Cut for  Photoelectric effect (PHOT)
        if not nhits > 2:
            return "PH", 0.50  # 'PH' for Photoelectric

        # 2. If not Photoelectric, extract hit data and execute PointNet
        data_input, _ = self.extract_hit_data(event)

        if data_input is None or data_input.shape[2] == 0:
            return "UN", 1.00

        with torch.no_grad():
            logits, _ = self.model(data_input)
            prob = torch.sigmoid(logits).item()

        if logits >= 0:
            return "PA", prob  # 'PA' for Pair Production
        elif logits < 0:
            return "CO", 1.0 - prob  # 'CO' for Compton Scattering

        return "UN", 1.00

def main(input_path, output_dir, geometry_name, model_traced, onlyACDVeto=True, rf=None, lookup_path=None, debug=False):

    # Global MEGAlib initialization
    G = M.MGlobal()
    G.Initialize()

    # Load MEGAlib Geometry
    Geometry = M.MDGeometryQuest()
    if Geometry.ScanSetupFile(M.MString(geometry_name)) == True:
        print("Geometry " + geometry_name + " loaded successfully!")
    else:
        print("Unable to load geometry " + geometry_name + " - Aborting!")
        quit()

    # Input file
    path_in = Path(input_path)
    files_to_process = []

    if path_in.is_file():
        files_to_process.append(path_in)
    elif path_in.is_dir():
        files_to_process.extend(path_in.glob("*.sim"))
        files_to_process.extend(path_in.glob("*.sim.gz"))
    else:
        print(f"Error: input path '{input_path}' does not exist or is invalid.. Aborting!")
        quit()

    if not files_to_process:
        print(f"No .sim or .sim.gz files found in '{input_path}'. Exiting.")
        return

    # Initiate the pipeline object
    pipeline = EventClassifierPipeline(model_traced, onlyACDVeto=onlyACDVeto, random_forest_path=rf, lookup_path=lookup_path)

    path_out_dir = Path(output_dir)
    path_out_dir.mkdir(parents=True, exist_ok=True)

    print("Starting event processing loop...")

    for fn_in in files_to_process:
        base_name = fn_in.name.split('.')[0]
        if path_out_dir.suffix:
            clean_out_dir = path_out_dir.parent
        else:
            clean_out_dir = path_out_dir

        fn_out = clean_out_dir / f"{base_name}.etp"

        print(f"\n[INFO] Processing file: {fn_in.name} -> Target Output: {fn_out}")

        Reader = M.MFileEventsSim(Geometry)
        if Reader.Open(M.MString(str(fn_in))) == False:
            print(f"Unable to open file {fn_in}. Skipping!")
            continue

        # Level 1 and 2 probabilities
        prob_l1 = {"UN": [], "MU": [], "SIGNAL": []}
        prob_l2 = {"PH": [], "PA": [], "CO": [], "UN": []}
        
        # Master metrics dictionary combining true/miss, interaction types, and all features
        # Structure: metrics[outcome][process_key][feature]
        processes = ["PH", "PA", "CO"]
        outcomes = ["true", "miss"]
        features = ["incident_energy", "zpos", "E_tra", "E_cal", "E_tot", "nhits_tra", "nhits_cal", "nhits_tot"]
        
        metrics = {
            outcome: {proc: {feat: [] for feat in features} for proc in processes}
            for outcome in outcomes
        }

        confusion_matrix = np.zeros((3, 3), dtype=int)
        mc_mapping = {"COMP": 0, "PAIR": 1, "PHOT": 2}
        pred_mapping = {"CO": 0, "PA": 1, "PH": 2}
        process_map = {"PHOT": "PH", "PAIR": "PA", "COMP": "CO"}

        with open(fn_out, "w") as f_out:

            t_read = t_classify = t_write = 0.0
            i = 0
            with tqdm(desc="Events", unit=" evt") as pbar:
                while True:
                    t0 = time.perf_counter()
                    Event = Reader.GetNextEvent()
                    if not Event:
                        break

                    M.SetOwnership(Event, True)
                    t_read += time.perf_counter() - t0
                    
                    i += 1
                    id_event = Event.GetID()
                    
                    t0 = time.perf_counter()

                    status, prob_bkg = pipeline.signal_background_classifier(Event, onlyACDVeto)
                    
                    if status in prob_l1:
                        prob_l1[status].append(prob_bkg)

                    if status == "SIGNAL":
                        event_type, probability = pipeline.type_of_signal(Event)
                        if event_type in prob_l2:
                            prob_l2[event_type].append(probability)
                    else:
                        event_type, probability = status, prob_bkg
                    
                    t_classify += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    if debug:
                        mc_process = "UNKNOWN"
                        if Event.GetNIAs() > 1:
                            mc_process = str(Event.GetIAAt(1).GetProcess().Data())
                            # Update confusion matrix
                            if status == "SIGNAL" and mc_process in mc_mapping and event_type in pred_mapping:
                                true_idx = mc_mapping[mc_process]
                                pred_idx = pred_mapping[event_type]
                                confusion_matrix[true_idx, pred_idx] += 1

                            # Extract Data & Populate Unified Metrics Dict
                            if Event.GetNIAs() > 0 and mc_process in process_map:
                                ia_e = Event.GetIAAt(0).GetSecondaryEnergy() / 1000.0 # keV to MeV
                                zpos = Event.GetIAAt(1).GetPosition().Z()

                                hit_data_tra, n_hits_tra = pipeline.extract_hit_data(Event, detId=1)
                                hit_data_cal, n_hits_cal = pipeline.extract_hit_data(Event, detId=2)
                                edep_tra = hit_data_tra[0, 3, :].sum().item()/1000 if hit_data_tra is not None else np.nan # keV to MeV
                                edep_cal = hit_data_cal[0, 3, :].sum().item()/1000 if hit_data_cal is not None else np.nan # keV to MeV
                                
                                # Determine the dictionary keys
                                proc_key = process_map[mc_process]
                                outcome_key = 'true' if event_type == proc_key else "miss"
                                
                                # Target specific leaf inside metrics[outcome][process]
                                tgt = metrics[outcome_key][proc_key]
                                
                                # Append features cleanly
                                tgt["incident_energy"].append(ia_e)
                                tgt["zpos"].append(zpos)
                                tgt["E_tra"].append(edep_tra)
                                tgt["E_cal"].append(edep_cal)
                                tgt["E_tot"].append(edep_tra + edep_cal)
                                tgt["nhits_tra"].append(n_hits_tra)
                                tgt["nhits_cal"].append(n_hits_cal)
                                tgt["nhits_tot"].append(n_hits_tra + n_hits_cal)
                                    
                        print(
                            f"SE\nID {id_event}\nMC {mc_process}\nET {event_type}\nTP {probability:.4f}",
                            file=f_out,
                        )
                    else:
                        # Write output like Nathan
                        print(
                            f"SE\nID {id_event}\nET {event_type}\nTP {probability:.4f}",
                            file=f_out,
                        )
                    t_write += time.perf_counter() - t0
                    del Event
                    pbar.update(1)
                        
                    if i % 500 == 0:
                        pbar.set_postfix({
                            "read":     f"{t_read  / i * 1000:.1f}ms",
                            "classify": f"{t_classify / i * 1000:.1f}ms",
                            "write":    f"{t_write / i * 1000:.1f}ms",
                        })
                        f_out.flush()

            print(f"\nDONE. {i} events processed.")
            print(f"  avg read    : {t_read     / i * 1000:.2f} ms/evt")
            print(f"  avg classify: {t_classify / i * 1000:.2f} ms/evt")
            print(f"  avg write   : {t_write    / i * 1000:.2f} ms/evt")
            print(f"[OK] File {fn_in.name} completed successfully. Saved to {fn_out}")

          
            print("[INFO] Plots...")
          
            
            categories_l1 = ['MU', 'SIGNAL', 'UN']
            counts_l1 = [len(prob_l1[cat]) for cat in categories_l1]
            plt.figure(figsize=(8, 5))
            bars = plt.bar(categories_l1, counts_l1, color=['orange', 'crimson', 'teal'], edgecolor='black', alpha=0.7)
            plt.title('L1: Signal vs Background Counts', fontsize=12, fontweight='bold')
            plt.ylabel('Events')
            
            max_c1 = max(counts_l1) if counts_l1 and max(counts_l1) > 0 else 1
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval + (max_c1*0.01), f'{yval}', ha='center', va='bottom')
            plt.savefig(clean_out_dir / f"{base_name}_L1_counts.png", dpi=300)
            plt.close()

        
            plt.figure(figsize=(9, 5))
            bins = 50
            l1_hist_configs = [
                ('MU', 'crimson', 'darkred'),
                ('SIGNAL', 'teal', 'darkslategray'),
                ('UN', 'gray', 'dimgray')
            ]
            for cat, f_col, e_col in l1_hist_configs:
                if prob_l1[cat]: 
                    plt.hist(prob_l1[cat], bins=bins, range=(0, 1), alpha=0.4, label=cat, color=f_col, histtype='stepfilled', edgecolor=e_col)

            
            plt.title('L1: Probability Distribution (TP)', fontsize=12, fontweight='bold')
            plt.yscale('log')
            plt.legend()
            plt.savefig(clean_out_dir / f"{base_name}_L1_probabilities.png", dpi=300)
            plt.close()

            
            categories_l2 = ['CO', 'PA', 'PH', 'UN']
            counts_l2 = [len(prob_l2[cat]) for cat in categories_l2]

            plt.figure(figsize=(8, 5))
            bars = plt.bar(categories_l2, counts_l2, color=['royalblue', 'forestgreen', 'darkorchid', 'gray'], edgecolor='black', alpha=0.7)
            plt.title('L2: Photon Type Classification Counts', fontsize=12, fontweight='bold')
            plt.ylabel('Events')

            max_c2 = max(counts_l2) if counts_l2 and max(counts_l2) > 0 else 1
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval + (max_c2*0.01), f'{yval}', ha='center', va='bottom')
            plt.savefig(clean_out_dir / f"{base_name}_L2_counts.png", dpi=300)
            plt.close()

            # --- PLOT LAYER 2: PROBABILITÀ ---
            plt.figure(figsize=(9, 5))
            l2_hist_configs = [
                ('CO', 'CO (Compton)', 'royalblue', 'darkblue'),
                ('PA', 'PA (Pair)', 'forestgreen', 'darkgreen'),
                ('PH', 'PH (Photo)', 'darkorchid', 'purple'),
                ('UN', 'UN', 'gray', 'dimgray')
            ]
            for cat, label, f_col, e_col in l2_hist_configs:
                if prob_l2[cat]:
                    plt.hist(prob_l2[cat], bins=bins, range=(0, 1), alpha=0.4, label=label, color=f_col, histtype='stepfilled', edgecolor=e_col)

                    
            plt.title('L2: Probability Distribution (TP)', fontsize=12, fontweight='bold')
            plt.yscale('log')
            plt.legend()
            plt.savefig(clean_out_dir / f"{base_name}_L2_probabilities.png", dpi=300)
            plt.close()

            
            if debug:
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
                ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                labels = ['Compton (CO)', 'Pair (PA)', 'Photo (PH)']
                ax.set_xticks(np.arange(3))
                ax.set_yticks(np.arange(3))
                ax.set_xticklabels(labels)
                ax.set_yticklabels(labels)
                
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                
                # Render values onto the matrix squares
                thresh = confusion_matrix.max() / 2. if confusion_matrix.max() > 0 else 1
                for row in range(3):
                    for col in range(3):
                        ax.text(col, row, format(confusion_matrix[row, col], 'd'),
                                ha="center", va="center",
                                color="white" if confusion_matrix[row, col] > thresh else "black",
                                fontweight='bold')
                
                ax.set_title("Confusion Matrix 3x3 (L2 Layer)", fontsize=12, fontweight='bold')
                ax.set_xlabel('Predicted Label', fontweight='bold')
                ax.set_ylabel('True MC Label', fontweight='bold')
                fig.tight_layout()
                
                matrix_path = clean_out_dir / f"{base_name}_confusion_matrix.png"
                plt.savefig(matrix_path, dpi=300)
                plt.close()
                print(f"[OK] Confusion Matrix : {matrix_path}")    
        
            if debug:
                processes = ["CO", "PA", "PH"]
                
                # Helper function to generate dynamic bins across all plot types safely
                def get_dynamic_bins(data_list, bin_rule, fallback=50):
                    if not isinstance(bin_rule, str):
                            return bin_rule
                    if not data_list:
                        return fallback
                    
                    if bin_rule == "linspace":
                        return np.linspace(min(data_list), max(data_list), 51)
                    if bin_rule == "arange":
                        return np.arange(0, int(max(data_list)) + 1, 1)
                    return fallback # Default fallback if an explicit bin array was provided
                
                # =================================================================
                # TYPE 1: Metrics split by Particle Type (CO, PA, PH)
                # =================================================================
                particle_plots = [
                    {
                        "feat": "incident_energy",
                        "xlabel": "Incident Energy (MeV)",
                        "title": "Incident Energy Distribution by Category (Layer 2)",
                        "bin_rule": np.arange(0, 51, 1),
                        "log_y": True,
                        "suffix": "energy",
                        "calc": None
                    },
                    {
                        "feat": "zpos",
                        "xlabel": "Incidence Z Position (cm)",
                        "title": "Z Vertex Distribution by Category (Layer 2)",
                        "bin_rule": np.linspace(-15, 30, 51),
                        "log_y": False,
                        "suffix": "zpos",
                        "calc": None
                    }, 
                    {
                        "feat": "erat", 
                        "xlabel": "Energy Ratio (E_tra / E_cal)",
                        "title": "Deposited Energy Ratio (TRA / CAL) by Category (Layer 2)",
                        "bin_rule": "linspace", # Now dynamic based on data limits!
                        "log_y": False,
                        "suffix": "erat",
                        "calc": lambda b: np.divide(np.array(b["E_tra"]), np.array(b["E_cal"]), 
                                                    out=np.full_like(np.array(b["E_tra"]), np.nan, dtype=float), 
                                                    where=np.array(b["E_cal"]) > 0)
                    }, # Added missing comma here
                    {
                        "feat": "nrat", 
                        "xlabel": "Number of Hits Ratio (n_tra / n_cal)",
                        "title": "Number of Hits Ratio (TRA / CAL) by Category (Layer 2)",
                        "bin_rule": "linspace", # Ratios are continuous floats, so linspace fits best
                        "log_y": False,
                        "suffix": "nrat",
                        "calc": lambda b: np.divide(np.array(b["nhits_tra"]), np.array(b["nhits_cal"]), 
                                                    out=np.full_like(np.array(b["nhits_tra"]), np.nan, dtype=float), 
                                                    where=np.array(b["nhits_cal"]) > 0)
                    }
                ]

                for p in particle_plots:
                    feat = p["feat"]
                    true_dict, miss_dict = {}, {}
                    
                    for proc in processes:
                        if p.get("calc") is not None:
                            raw_true = p["calc"](metrics["true"][proc])
                            raw_miss = p["calc"](metrics["miss"][proc])
                            true_dict[proc] = raw_true[~np.isnan(raw_true)].tolist()
                            miss_dict[proc] = raw_miss[~np.isnan(raw_miss)].tolist()
                        else:
                            true_dict[proc] = metrics["true"][proc][feat]
                            miss_dict[proc] = metrics["miss"][proc][feat]
                   
                    all_dict = {proc: true_dict[proc] + miss_dict[proc] for proc in processes}
                    
                    # Unify dynamic flattening across all channels for evaluation
                    flat_data = sum(true_dict.values(), []) + sum(miss_dict.values(), [])
                    bins = get_dynamic_bins(flat_data, p["bin_rule"])
                   
                    plot_path = clean_out_dir / f"{base_name}_wrong_predictions_by_category_{p['suffix']}.png"
                    plot_type_classification_comparison(
                        true_by_category=true_dict, miss_by_category=miss_dict, all_by_category=all_dict,
                        output_path=plot_path, xlabel=p["xlabel"], title=p["title"], bins=bins, log_y=p["log_y"]
                    )
                    print(f"[OK] {p['title']} salvato in: {plot_path}")
                # =================================================================
                # TYPE 2: Metrics split by Detector Component (TRA, CAL, TOT)
                # =================================================================
                detector_plots = [
                    {
                        "feats": {"TRA": "E_tra", "CAL": "E_cal", "TOT": "E_tot"},
                        "xlabel": "Deposited energy (MeV)",
                        "title": "Deposited Energy Distribution by Detector (Layer 2)",
                        "suffix": "edep",
                        "bin_rule": "linspace"
                    },
                    {
                        "feats": {"TRA": "nhits_tra", "CAL": "nhits_cal", "TOT": "nhits_tot"},
                        "xlabel": "Number of hits",
                        "title": "Distribution of Number of Hits by Detector (Layer 2)",
                        "suffix": "nhits",
                        "bin_rule": "arange" # Hits are whole integers, so arange fits perfectly
                    }
                ]

                for d in detector_plots:
                    true_dict = {det: [] for det in ["TRA", "CAL", "TOT"]}
                    miss_dict = {det: [] for det in ["TRA", "CAL", "TOT"]}
                    
                    for det, feat_key in d["feats"].items():
                        for proc in processes:
                            true_dict[det].extend(metrics["true"][proc][feat_key])
                            miss_dict[det].extend(metrics["miss"][proc][feat_key])
                            
                    all_dict = {det: true_dict[det] + miss_dict[det] for det in ["TRA", "CAL", "TOT"]}
                    
                    # Compute flat dataset to automatically extract boundaries
                    flat_data = true_dict["TRA"] + true_dict["CAL"] + true_dict["TOT"] + miss_dict["TRA"] + miss_dict["CAL"] + miss_dict["TOT"]
                    bins = get_dynamic_bins(flat_data, d["bin_rule"])
                        
                    plot_path = clean_out_dir / f"{base_name}_wrong_predictions_by_detector_{d['suffix']}.png"
                    plot_type_classification_comparison(
                        true_by_category=true_dict, miss_by_category=miss_dict, all_by_category=all_dict,
                        output_path=plot_path, xlabel=d["xlabel"], title=d["title"], bins=bins, log_y=False,
                        categories=["TRA", "CAL", "TOT"]
                    )
                    print(f"[OK] {d['title']} salvato in: {plot_path}")
                detector_plots = [
                    {
                        "feats": {"TRA": "E_tra", "CAL": "E_cal", "TOT": "E_tot"},
                        "xlabel": "Deposited energy (MeV)",
                        "title": "Deposited Energy Distribution by Detector (Layer 2)",
                        "suffix": "edep",
                        "bin_type": "linspace"
                    },
                    {
                        "feats": {"TRA": "nhits_tra", "CAL": "nhits_cal", "TOT": "nhits_tot"},
                        "xlabel": "Number of hits",
                        "title": "Distribution of Number of Hits by Detector (Layer 2)",
                        "suffix": "nhits",
                        "bin_type": "arange"
                    }
                ]

                for d in detector_plots:
                    true_dict = {det: [] for det in ["TRA", "CAL", "TOT"]}
                    miss_dict = {det: [] for det in ["TRA", "CAL", "TOT"]}
                    
                    # Aggregate data across ALL processes (CO + PA + PH) for each detector type
                    for det, feat_key in d["feats"].items():
                        for proc in processes:
                            true_dict[det].extend(metrics["true"][proc][feat_key])
                            miss_dict[det].extend(metrics["miss"][proc][feat_key])
                            
                    all_dict = {det: true_dict[det] + miss_dict[det] for det in ["TRA", "CAL", "TOT"]}
                    
                    # Calculate bins dynamically based on data content
                    combined_flat = true_dict["TRA"] + true_dict["CAL"] + true_dict["TOT"] + miss_dict["TRA"] + miss_dict["CAL"] + miss_dict["TOT"]
                    
                    if combined_flat:
                        if d["bin_type"] == "linspace":
                            bins = np.linspace(min(combined_flat), max(combined_flat), 51)
                        else: # arange
                            bins = np.arange(0, int(max(combined_flat)) + 1, 1)
                    else:
                        bins = 50 # fallback
                        
                    plot_path = clean_out_dir / f"{base_name}_wrong_predictions_by_detector_{d['suffix']}.png"
                    plot_type_classification_comparison(
                        true_by_category=true_dict, miss_by_category=miss_dict, all_by_category=all_dict,
                        output_path=plot_path, xlabel=d["xlabel"], title=d["title"], bins=bins, log_y=False,
                        categories=["TRA", "CAL", "TOT"]
                    )
                    print(f"[OK] {d['title']} salvato in: {plot_path}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Event Classifier Pipeline for MEGAlib simulation files."
    )

    parser.add_argument(
        "-i", "--input", 
        type=str, 
        default="./ComPair23_1MeV_50MeV_powerlaw.p1.inc10.id1.sim.gz",
        help="Path to a single .sim/.sim.gz file OR to a directory containing them."
    )
    parser.add_argument(
        "-o", "--output-dir", 
        type=str, 
        default="./output_etp",
        help="Directory where output .etp files will be saved."
    )
    parser.add_argument(
        "-g", "--geometry", 
        type=str, 
        default="../../simuComPair/Geometry/ComPair_23/ComPair23.geo.setup",
        help="Path to the MEGAlib geometry setup file (.geo.setup)."
    )
    parser.add_argument(
        "-m", "--model", 
        type=str, 
        default="./PointNetModels/test_torch_model_params_final_26-06.pth",
        help="Path to the PointNet model weights file (.pt)."
    )
    
    parser.add_argument(
        "--disable-onlyacd", 
        action="store_false", 
        dest="only_acd_veto",
        help="Disable the strict ACD-only veto and enable the Random Forest/PCA layer."
    )
    parser.add_argument(
        "-rf", "--random-forest", 
        type=str, 
        default=None, #"./RandomForest/vega_model.pkl",
        help="Path to the Random Forest model pickle file (used only if ACD-only veto is disabled)."
    )
    parser.add_argument(
        "-pca", "--pca", 
        type=str, 
        default=None, #"./pca_files",
        help="LookupTable pca."
    )

    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode to print MC true processes into the output file."
    )

    # Parse the arguments from command line
    args = parser.parse_args()

    # Pass the parsed arguments directly to the main function
    main(
        input_path=args.input, 
        output_dir=args.output_dir, 
        geometry_name=args.geometry, 
        model_traced=args.model, 
        onlyACDVeto=args.only_acd_veto, 
        rf=None if args.pca is not None else args.random_forest,
        lookup_path=args.pca,
        debug=args.debug
    )
