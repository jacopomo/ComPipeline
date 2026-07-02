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
            return None

        data = torch.zeros([1, 4, nhits])
        if detId is None:
            for i in range(nhits):
                data[0, 0, i] = event.GetHTAt(i).GetPosition().X()
                data[0, 1, i] = event.GetHTAt(i).GetPosition().Y()
                data[0, 2, i] = event.GetHTAt(i).GetPosition().Z()
                data[0, 3, i] = event.GetHTAt(i).GetEnergy()
            return data 
        else:
            j=0
            for i in range(nhits):
                if event.GetHTAt(i).GetDetectorType() == detId:
                    data[0, 0, j] = event.GetHTAt(i).GetPosition().X()
                    data[0, 1, j] = event.GetHTAt(i).GetPosition().Y()
                    data[0, 2, j] = event.GetHTAt(i).GetPosition().Z()
                    data[0, 3, j] = event.GetHTAt(i).GetEnergy()
                    j += 1
            return data[:, :, :j] if j > 0 else None
    

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
            data = self.extract_hit_data(event, 1)
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
        data_input = self.extract_hit_data(event)

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

        energy_true_PH_wrong = []
        energy_true_PA_wrong = []
        energy_true_CO_wrong = []

        zpos_true_PH_wrong = []
        zpos_true_PA_wrong = []
        zpos_true_CO_wrong = []

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
                                    
                                # Canale 1: True Photo misclassified (CO o PA)
                                if mc_process == "PHOT" and event_type != "PH":
                                    energy_true_PH_wrong.append(ia_e)
                                    zpos_true_PH_wrong.append(zpos)
                                    
                                # Canale 2: True Pair misclassified (PH o CO)
                                elif mc_process == "PAIR" and event_type != "PA":
                                    energy_true_PA_wrong.append(ia_e)
                                    zpos_true_PA_wrong.append(zpos)
                                    
                                # Canale 3: True Compton misclassified (PA o PH)
                                elif mc_process == "COMP" and event_type != "CO":
                                    energy_true_CO_wrong.append(ia_e)
                                    zpos_true_CO_wrong.append(zpos)
                                    
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
                # --- NUOVO PLOT 4 SPETTRO ENERGETICO DEGLI ERRORI PER CATEGORIA ---
                plt.figure(figsize=(10, 6))
                bins = 50
                
                # Plottiamo le tre distribuzioni sovrapposte usando 'step' o barre trasparenti
                plt.hist(energy_true_CO_wrong, bins=bins, alpha=0.5, 
                         label=f'True Compton pred. PA/PH (Total: {len(energy_true_CO_wrong)})', 
                         color='royalblue', histtype='stepfilled', edgecolor='darkblue')
                         
                plt.hist(energy_true_PA_wrong, bins=bins, alpha=0.5, 
                         label=f'True Pair pred. PH/CO (Total: {len(energy_true_PA_wrong)})', 
                         color='forestgreen', histtype='stepfilled', edgecolor='darkgreen')
                         
                plt.hist(energy_true_PH_wrong, bins=bins, alpha=0.5, 
                         label=f'True Photo pred. CO/PA (Total: {len(energy_true_PH_wrong)})', 
                         color='darkorchid', histtype='stepfilled', edgecolor='purple')
                
                plt.title('Energy Distribution of Misclassified Events (Layer 2)', fontsize=12, fontweight='bold')
                plt.xlabel('Incident Energy (MeV)', fontsize=11)
                plt.ylabel('Number of Wrong Predicted Events', fontsize=11)
                plt.grid(True, which="both", linestyle='--', alpha=0.5)
                plt.legend(loc='upper right', fontsize=10)
                
                # Consigliato in fisica per gli spettri energetici a larga banda
                plt.yscale('log') 
                
                energy_plot_path = clean_out_dir / f"{base_name}_wrong_predictions_by_category_energy.png"
                plt.savefig(energy_plot_path, dpi=300)
                plt.close()
                print(f"[OK] Spettro degli errori per categoria salvato in: {energy_plot_path}")

                plt.figure(figsize=(10, 6))
                bins = 50
                
                plt.hist(zpos_true_CO_wrong, bins=bins, alpha=0.5, 
                         label=f'True Compton pred. PA/PH (Total: {len(zpos_true_CO_wrong)})', 
                         color='royalblue', histtype='stepfilled', edgecolor='darkblue')
                         
                plt.hist(zpos_true_PA_wrong, bins=bins, alpha=0.5, 
                         label=f'True Pair pred. PH/CO (Total: {len(zpos_true_PA_wrong)})', 
                         color='forestgreen', histtype='stepfilled', edgecolor='darkgreen')
                         
                plt.hist(zpos_true_PH_wrong, bins=bins, alpha=0.5, 
                         label=f'True Photo pred. CO/PA (Total: {len(zpos_true_PH_wrong)})', 
                         color='darkorchid', histtype='stepfilled', edgecolor='purple')
                
                plt.title('Z Vertex Distribution of Misclassified Events (Layer 2)', fontsize=12, fontweight='bold')
                plt.xlabel('Incidence Z Position (cm)', fontsize=11)
                plt.ylabel('Number of Wrong Predicted Events', fontsize=11)
                plt.grid(True, which="both", linestyle='--', alpha=0.5)
                plt.legend(loc='upper right', fontsize=10)
                
                # Per la coordinata spaziale Z di solito si preferisce la scala lineare, 
                # ma se vedi picchi enormi puoi scommentare la riga sotto:
                # plt.yscale('log') 
                
                zpos_plot_path = clean_out_dir / f"{base_name}_wrong_predictions_by_category_zpos.png"
                plt.savefig(zpos_plot_path, dpi=300)
                plt.close()
                print(f"[OK] Grafico delle posizioni Z degli errori salvato in: {zpos_plot_path}")
                
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
