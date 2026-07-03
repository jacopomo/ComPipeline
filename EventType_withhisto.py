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

        prob_MU_l1 = []
        prob_SIGNAL_l1 = []
        prob_UN_l1 = []

        prob_PH_l2 = []
        prob_PA_l2 = []
        prob_CO_l2 = []
        prob_UN_l2 = []

        incident_energy_PH_true, incident_energy_PH_miss = [], []
        incident_energy_PA_true, incident_energy_PA_miss = [], []
        incident_energy_CO_true, incident_energy_CO_miss = [], []

        zpos_PH_true, zpos_PH_miss = [], []
        zpos_PA_true, zpos_PA_miss = [], []
        zpos_CO_true, zpos_CO_miss = [], []

        E_deposited_tra_true, E_deposited_tra_miss = [], []
        E_deposited_cal_true, E_deposited_cal_miss = [], []
        E_deposited_tot_true, E_deposited_tot_miss = [], []

        nhits_tra_true, nhits_tra_miss = [], []
        nhits_cal_true, nhits_cal_miss = [], []
        nhits_tot_true, nhits_tot_miss = [], []


        confusion_matrix = np.zeros((3, 3), dtype=int)
        mc_mapping = {"COMP": 0, "PAIR": 1, "PHOT": 2}
        pred_mapping = {"CO": 0, "PA": 1, "PH": 2}
        
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
                    
                    if status == "UN": prob_UN_l1.append(prob_bkg)
                    elif status == "MU": prob_MU_l1.append(prob_bkg)
                    elif status == "SIGNAL": prob_SIGNAL_l1.append(prob_bkg)

                    if status == "SIGNAL":
                        event_type, probability = pipeline.type_of_signal(Event)
                        if event_type == "PH": prob_PH_l2.append(probability)
                        elif event_type == "PA": prob_PA_l2.append(probability)
                        elif event_type == "CO": prob_CO_l2.append(probability)
                        elif event_type == "UN": prob_UN_l2.append(probability)
                    else:
                        event_type, probability = status, prob_bkg
                    
                    t_classify += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    if debug:
                        mc_process = "UNKNOWN"
                        if Event.GetNIAs() > 1:
                            mc_process = str(Event.GetIAAt(1).GetProcess().Data())
                            # Confusion matrix
                            if status == "SIGNAL":
                                if mc_process in mc_mapping and event_type in pred_mapping:
                                    true_idx = mc_mapping[mc_process]
                                    pred_idx = pred_mapping[event_type]
                                    confusion_matrix[true_idx, pred_idx] += 1

                                if Event.GetNIAs() > 0:
                                    ia_e = Event.GetIAAt(0).GetSecondaryEnergy() / 1000.0
                                    zpos = Event.GetIAAt(1).GetPosition().Z()

                                    hit_data_tra, n_hits_tra = pipeline.extract_hit_data(Event, detId=1)
                                    hit_data_cal, n_hits_cal = pipeline.extract_hit_data(Event, detId=2)
                                    edep_tra = hit_data_tra[0, 3, :].sum().item() if hit_data_tra is not None else 0.0
                                    edep_cal = hit_data_cal[0, 3, :].sum().item() if hit_data_cal is not None else 0.0
                                # Canale 1: True Photo
                                if mc_process == "PHOT":
                                    if event_type == "PH":
                                        incident_energy_PH_true.append(ia_e)
                                        zpos_PH_true.append(zpos)
                                        E_deposited_tra_true.append(edep_tra)
                                        E_deposited_cal_true.append(edep_cal)
                                        E_deposited_tot_true.append(edep_tra+edep_cal)
                                        nhits_tra_true.append(n_hits_tra)
                                        nhits_cal_true.append(n_hits_cal)
                                        nhits_tot_true.append(n_hits_tra+n_hits_cal)
                                    else:
                                        incident_energy_PH_miss.append(ia_e)
                                        zpos_PH_miss.append(zpos)
                                        E_deposited_tra_miss.append(edep_tra)
                                        E_deposited_cal_miss.append(edep_cal)
                                        E_deposited_tot_miss.append(edep_tra+edep_cal)
                                        nhits_tra_miss.append(n_hits_tra)
                                        nhits_cal_miss.append(n_hits_cal)
                                        nhits_tot_miss.append(n_hits_tra+n_hits_cal)
                                # Canale 2: True PAIR
                                if mc_process == "PAIR":
                                    if event_type == "PA":
                                        incident_energy_PA_true.append(ia_e)
                                        zpos_PA_true.append(zpos)
                                        E_deposited_tra_true.append(edep_tra)
                                        E_deposited_cal_true.append(edep_cal)
                                        E_deposited_tot_true.append(edep_tra+edep_cal)
                                        nhits_tra_true.append(n_hits_tra)
                                        nhits_cal_true.append(n_hits_cal)
                                        nhits_tot_true.append(n_hits_tra+n_hits_cal)
                                    else:
                                        incident_energy_PA_miss.append(ia_e)
                                        zpos_PA_miss.append(zpos)
                                        E_deposited_tra_miss.append(edep_tra)
                                        E_deposited_cal_miss.append(edep_cal)
                                        E_deposited_tot_miss.append(edep_tra+edep_cal)
                                        nhits_tra_miss.append(n_hits_tra)
                                        nhits_cal_miss.append(n_hits_cal)
                                        nhits_tot_miss.append(n_hits_tra+n_hits_cal)

                                # Canale 3: True COMP
                                if mc_process == "COMP":
                                    if event_type == "CO":
                                        incident_energy_CO_true.append(ia_e)
                                        zpos_CO_true.append(zpos)
                                        E_deposited_tra_true.append(edep_tra)
                                        E_deposited_cal_true.append(edep_cal)
                                        E_deposited_tot_true.append(edep_tra+edep_cal)
                                        nhits_tra_true.append(n_hits_tra)
                                        nhits_cal_true.append(n_hits_cal)
                                        nhits_tot_true.append(n_hits_tra+n_hits_cal)
                                    else:
                                        incident_energy_CO_miss.append(ia_e)
                                        zpos_CO_miss.append(zpos)
                                        E_deposited_tra_miss.append(edep_tra)
                                        E_deposited_cal_miss.append(edep_cal)
                                        E_deposited_tot_miss.append(edep_tra+edep_cal)
                                        nhits_tra_miss.append(n_hits_tra)
                                        nhits_cal_miss.append(n_hits_cal)
                                        nhits_tot_miss.append(n_hits_tra+n_hits_cal)
                                    
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
            counts_l1 = [len(prob_MU_l1), len(prob_SIGNAL_l1), len(prob_UN_l1)]
            plt.figure(figsize=(8, 5))
            bars = plt.bar(categories_l1, counts_l1, color=['orange', 'crimson', 'teal'], edgecolor='black', alpha=0.7)
            plt.title('L1: Signal vs Background Counts', fontsize=12, fontweight='bold')
            plt.ylabel('Events')
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval + (max(counts_l1)*0.01), f'{yval}', ha='center', va='bottom')
            plt.savefig(clean_out_dir / f"{base_name}_L1_counts.png", dpi=300)
            plt.close()

        
            plt.figure(figsize=(9, 5))
            bins = 50
            if prob_MU_l1: plt.hist(prob_MU_l1, bins=bins, range=(0, 1), alpha=0.4, label='MU', color='crimson', histtype='stepfilled', edgecolor='darkred')
            if prob_SIGNAL_l1: plt.hist(prob_SIGNAL_l1, bins=bins, range=(0, 1), alpha=0.4, label='SIGNAL', color='teal', histtype='stepfilled', edgecolor='darkslategray')
            if prob_UN_l1: plt.hist(prob_UN_l1, bins=bins, range=(0, 1), alpha=0.4, label='UN', color='gray', histtype='stepfilled', edgecolor='dimgray')
            plt.title('L1: Probability Distribution (TP)', fontsize=12, fontweight='bold')
            plt.yscale('log')
            plt.legend()
            plt.savefig(clean_out_dir / f"{base_name}_L1_probabilities.png", dpi=300)
            plt.close()

            
            categories_l2 = ['CO', 'PA', 'PH', 'UN']
            counts_l2 = [len(prob_CO_l2), len(prob_PA_l2), len(prob_PH_l2), len(prob_UN_l2)]
            plt.figure(figsize=(8, 5))
            bars = plt.bar(categories_l2, counts_l2, color=['royalblue', 'forestgreen', 'darkorchid', 'gray'], edgecolor='black', alpha=0.7)
            plt.title('L2: Photon Type Classification Counts', fontsize=12, fontweight='bold')
            plt.ylabel('Events')
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval + (max(counts_l2)*0.01), f'{yval}', ha='center', va='bottom')
            plt.savefig(clean_out_dir / f"{base_name}_L2_counts.png", dpi=300)
            plt.close()

            # --- PLOT LAYER 2: PROBABILITÀ ---
            plt.figure(figsize=(9, 5))
            if prob_CO_l2: plt.hist(prob_CO_l2, bins=bins, range=(0, 1), alpha=0.4, label='CO (Compton)', color='royalblue', histtype='stepfilled', edgecolor='darkblue')
            if prob_PA_l2: plt.hist(prob_PA_l2, bins=bins, range=(0, 1), alpha=0.4, label='PA (Pair)', color='forestgreen', histtype='stepfilled', edgecolor='darkgreen')
            if prob_PH_l2: plt.hist(prob_PH_l2, bins=bins, range=(0, 1), alpha=0.4, label='PH (Photo)', color='darkorchid', histtype='stepfilled', edgecolor='purple')
            if prob_UN_l2: plt.hist(prob_UN_l2, bins=bins, range=(0, 1), alpha=0.4, label='UN', color='gray', histtype='stepfilled', edgecolor='dimgray')
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
                
                # Inserisci i valori numerici dentro i quadrati della matrice
                thresh = confusion_matrix.max() / 2.
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
                # Incident energy 4-panel subplot
                energy_true_by_category = {
                    "CO": incident_energy_CO_true,
                    "PA": incident_energy_PA_true,
                    "PH": incident_energy_PH_true,
                }
                energy_miss_by_category = {
                    "CO": incident_energy_CO_miss,
                    "PA": incident_energy_PA_miss,
                    "PH": incident_energy_PH_miss,
                }
                energy_all_by_category = {
                    "CO": incident_energy_CO_true + incident_energy_CO_miss,
                    "PA": incident_energy_PA_true + incident_energy_PA_miss,
                    "PH": incident_energy_PH_true + incident_energy_PH_miss,
                }

                energy_plot_path = clean_out_dir / f"{base_name}_wrong_predictions_by_category_energy.png"
                energy_bins = np.arange(0, 51, 1) # 50 1 MeV bins
                plot_type_classification_comparison(
                    true_by_category=energy_true_by_category,
                    miss_by_category=energy_miss_by_category,
                    all_by_category=energy_all_by_category,
                    output_path=energy_plot_path,
                    xlabel="Incident Energy (MeV)",
                    title="Incident Energy Distribution by Category (Layer 2)",
                    bins=energy_bins,
                    log_y=True,
                )
                print(f"[OK] Spettro degli errori per categoria salvato in: {energy_plot_path}")
                
                # Incident zpos 4-panel subplot
                zpos_true_by_category = {
                    "CO": zpos_CO_true,
                    "PA": zpos_PA_true,
                    "PH": zpos_PH_true,
                }
                zpos_miss_by_category = {
                    "CO": zpos_CO_miss,
                    "PA": zpos_PA_miss,
                    "PH": zpos_PH_miss,
                }
                zpos_all_by_category = {
                    "CO": zpos_CO_true + zpos_CO_miss,
                    "PA": zpos_PA_true + zpos_PA_miss,
                    "PH": zpos_PH_true + zpos_PH_miss,
                }
                
                z_bins = np.linspace(-15, 30, 51) # 50 equally spaced bins defined for consistency
                zpos_plot_path = clean_out_dir / f"{base_name}_wrong_predictions_by_category_zpos.png"
                plot_type_classification_comparison(
                    true_by_category=zpos_true_by_category,
                    miss_by_category=zpos_miss_by_category,
                    all_by_category=zpos_all_by_category,
                    output_path=zpos_plot_path,
                    xlabel="Incidence Z Position (cm)",
                    title="Z Vertex Distribution by Category (Layer 2)",
                    bins=z_bins,
                    log_y=False,
                )
                print(f"[OK] Grafico delle posizioni Z degli errori salvato in: {zpos_plot_path}")
                
                # Deposited energy 4-panel subplot
                edep_true_by_category = {
                    "TRA": E_deposited_tra_true,
                    "CAL": E_deposited_cal_true,
                    "TOT": E_deposited_tot_true,
                }
                edep_miss_by_category = {
                    "TRA": E_deposited_tra_miss,
                    "CAL": E_deposited_cal_miss,
                    "TOT": E_deposited_tot_miss,
                }
                edep_tot_by_category = {
                    "TRA": E_deposited_tra_true + E_deposited_tra_miss,
                    "CAL": E_deposited_cal_true + E_deposited_cal_miss,
                    "TOT": E_deposited_tot_true + E_deposited_tot_miss,
                }

                # Extract all the integer values from the dictionaries and flatten
                all_values = []
                for d in [edep_true_by_category, edep_miss_by_category, edep_tot_by_category]:
                    for values_list in d.values():
                        if values_list:
                            all_values.extend(values_list)

                # Safeguard in case ALL dictionaries were empty
                if all_values:
                    bins_min = min(all_values)
                    bins_max = max(all_values)
                    edep_bins = np.linspace(bins_min, bins_max, 51)
                else:
                    print("Warning: No deposited energy found!")
                    edep_bins = 50  # Default fallback

                edep_plot_path = clean_out_dir / f"{base_name}_wrong_predictions_by_detector_edep.png"
                plot_type_classification_comparison(
                    true_by_category=edep_true_by_category,
                    miss_by_category=edep_miss_by_category,
                    all_by_category=edep_tot_by_category,
                    output_path=edep_plot_path,
                    xlabel="Deposited energy (MeV)",
                    title="Deposited Energy Distribution by Detector (Layer 2)",
                    bins=edep_bins,
                    log_y=False,
                    categories=["TRA", "CAL", "TOT"],
                ) 
                print(f"[OK] Grafico delle energie depositate degli errori salvato in: {edep_plot_path}")

                # Deposited energy 4-panel subplot
                nhits_true_by_category = {
                    "TRA": nhits_tra_true,
                    "CAL": nhits_cal_true,
                    "TOT": nhits_tot_true,
                }
                nhits_miss_by_category = {
                    "TRA": nhits_tra_miss,
                    "CAL": nhits_cal_miss,
                    "TOT": nhits_tot_miss,
                }
                nhits_tot_by_category = {
                    "TRA": nhits_tra_true + nhits_tra_miss,
                    "CAL": nhits_cal_true + nhits_cal_miss,
                    "TOT": nhits_tot_true + nhits_tot_miss,
                }
                
                # Extract all the integer values from the dictionaries and flatten
                all_values = []
                for d in [nhits_true_by_category, nhits_miss_by_category, nhits_tot_by_category]:
                    for values_list in d.values():
                        if values_list:
                            all_values.extend(values_list)

                # Safeguard in case ALL dictionaries were empty
                if all_values:
                    bins_min = 0
                    bins_max = int(max(all_values))
                    nhits_bins = np.linspace(bins_min, bins_max, 51)
                else:
                    print("Warning: No hit counts found!")
                    nhits_bins = 50  # Default fallback

                nhits_plot_path = clean_out_dir / f"{base_name}_wrong_predictions_by_detector_nhits.png"
                plot_type_classification_comparison(
                    true_by_category=nhits_true_by_category,
                    miss_by_category=nhits_miss_by_category,
                    all_by_category=nhits_tot_by_category,
                    output_path=nhits_plot_path,
                    xlabel="Number of hits",
                    title="Distribution of Number of Hits by Detector (Layer 2)",
                    bins=nhits_bins,
                    log_y=False,
                    categories=["TRA", "CAL", "TOT"],
                )
                print(f"[OK] Grafico delle nhits degli errori salvato in: {nhits_plot_path}")
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
