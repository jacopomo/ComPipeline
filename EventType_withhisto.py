import argparse
from tqdm import tqdm
import time
import numpy as np

import ROOT as M
from pathlib import Path
from EventType_plotter import plot_type_classification_comparison, plot_bar_counts, plot_overlaid_probabilities, plot_stacked_energy_spectrum, plot_confusion_matrix
from EventClassifierPipeline import EventClassifierPipeline

M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")

def record_debug_info(event, status, event_type, metrics, confusion_matrix, mc_mapping, pred_mapping, pipeline, log_file=None):
    """Debug-mode bookkeeping for a single event.

    Called once per event, only when --debug is set. Returns the true MC
    process string (or "UNKNOWN"), which the caller prints to the .etp file.

    Logs to file:
      - Events where GetNIAs() <= 1 (unresolvable MC truth)
      - Events where a SIGNAL event (L1) does not classify as PH, CO, or PA (L2)
    """
    mc_process = "UNKNOWN"

    # GetIAAt(0) is always the primary particle's own generation record, not a
    # real interaction — the *first actual interaction* is GetIAAt(1). That
    # means we need at least 2 recorded interactions (GetNIAs() > 1) before
    # GetIAAt(1) is safe to call at all.
    if event.GetNIAs() > 1:
        mc_process = str(event.GetIAAt(1).GetProcess().Data())

        # --- confusion_matrix: predicted L2 label vs. true MC process ---
        # Only meaningful for events that actually reached the L2 classifier (i.e., SIGNAL).
        if status == "SIGNAL" and mc_process in mc_mapping and event_type in pred_mapping:
            true_idx = mc_mapping[mc_process]
            pred_idx = pred_mapping[event_type]
            confusion_matrix[true_idx, pred_idx] += 1
        if status == "SIGNAL" and mc_process not in mc_mapping:
            true_idx = mc_mapping["OTHER"]
            pred_idx = pred_mapping[event_type]
            confusion_matrix[true_idx, pred_idx] += 1

        # --- metrics: per (true_process, L1 status, L2 label) feature store ---
        # Only store metrics for the three MC processes we care about (COMP, PAIR, PHOT) not RAYL!
        if mc_process in metrics: 

            # Get the info (features)
            ia_e = event.GetIAAt(0).GetSecondaryEnergy() / 1000.0  # keV -> MeV
            zpos = event.GetIAAt(1).GetPosition().Z()
            hit_data_tra = pipeline.extract_hit_data(event, detId=1)
            hit_data_cal = pipeline.extract_hit_data(event, detId=2)
            n_hits_tra = 0 if hit_data_tra is None else hit_data_tra.shape[2]
            n_hits_cal = 0 if hit_data_cal is None else hit_data_cal.shape[2]
            edep_tra = hit_data_tra[0, 3, :].sum().item() / 1000 if hit_data_tra is not None else np.nan  # keV -> MeV
            edep_cal = hit_data_cal[0, 3, :].sum().item() / 1000 if hit_data_cal is not None else np.nan  # keV -> MeV

            extracted_features = {
                "incident_energy": ia_e,
                "zpos": zpos,
                "E_tra": edep_tra,
                "E_cal": edep_cal,
                "E_tot": edep_tra + edep_cal,
                "nhits_tra": n_hits_tra,
                "nhits_cal": n_hits_cal,
                "nhits_tot": n_hits_tra + n_hits_cal,
            }

            # Store the extracted features in the metrics dictionary:
            # status is the L1 classification (UN, MU, SIGNAL)
            # event_type is the L2 classification (PH, PA, CO, UN)
            target_leaf = metrics[mc_process][status][event_type]
            for feat, val in extracted_features.items():
                target_leaf[feat].append(val)

        else:
            # Log events with unrecognized MC process (not COMP, PAIR, PHOT)
            if log_file:
                log_file.write(f"ID {event.GetID()}: Unrecognized MC process '{mc_process}' classified as {status} (L1) - {event_type} (L2)\n")
    else:
        # Log events with unresolvable MC truth (GetNIAs() <= 1)
        if log_file:
            log_file.write(f"ID {event.GetID()}: GetNIAs() = {event.GetNIAs()} (not > 1) - cannot resolve MC truth\n")

    # Log SIGNAL events that don't classify as PH, CO, or PA
    if status == "SIGNAL" and event_type not in ["PH", "CO", "PA"]:
        if log_file:
            log_file.write(f"ID {event.GetID()}: SIGNAL event classified as '{event_type}' (expected PH, CO, or PA)\n")

    # Return the true MC process string (or "UNKNOWN") for logging in the .etp output file.
    return mc_process

# ====================================================================================
# Main function to process input files, classify events, and generate output and plots
# ====================================================================================

def main(input_path, output_dir, geometry_name, model_traced, onlyACDVeto=True, rf=None, lookup_path=None, debug=False, three_class=False):

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
    pipeline = EventClassifierPipeline(model_traced, onlyACDVeto=onlyACDVeto, random_forest_path=rf, lookup_path=lookup_path, three_class=three_class)
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
        mc_processes = ["COMP", "PAIR", "PHOT"]                # possible TRUE MC processes (from simulation truth)
        states1 = ["UN", "MU", "SIGNAL"]                       # possible L1 (signal-vs-background) outcomes
        states2 = ["PH", "PA", "CO", "UN", "MU"]               # possible L2 (photon-type) outcomes
        features = ["incident_energy", "zpos", "E_tra", "E_cal", "E_tot", "nhits_tra", "nhits_cal", "nhits_tot"]

        # --------------------------------------------------------------------
        # Data-flow legend — every dict below is populated in exactly one
        # place, referenced here so it's easy to trace:
        #
        #   prob_l1[status]  -> list of L1 probabilities, keyed by L1 state ("UN", "MU", "SIGNAL").
        #       Populated for EVERY event, right after signal_background_classifier() runs
        #       (see "prob_l1[status].append(...)" below). Always in sync
        #       with the number of events read, regardless of --debug.
        #
        #   prob_l2[event_type] -> list of L2 probabilities, keyed by L2 label ("PH", "PA", "CO", "UN").
        #       Populated only for events with L1 state == "SIGNAL", right
        #       after type_of_signal() runs (see "prob_l2[event_type].append(...)").
        #
        #   metrics[mc_process][l1_state][l2_label][feature] -> list of values.
        #       Populated only in --debug mode, and only for events whose true
        #       MC process resolved to one of COMP/PAIR/PHOT. All the writing
        #       happens inside record_debug_info() (defined above main()).
        #       Feeds: the stacked energy spectrum (uses all 3 l1_state) and
        #       the "wrong predictions" Layer-2 plots (SIGNAL only).
        #
        #   confusion_matrix[true_idx, pred_idx] -> int counts.
        #       Populated only in --debug mode, only for SIGNAL events with a
        #       resolvable true process. Also written inside record_debug_info().
        # --------------------------------------------------------------------

        # metrics is a nested dict-of-dict-of-dict-of-dict structure, keyed by:
        #   mc_processes (COMP, PAIR, PHOT)
        #     -> states1 (UN, MU, SIGNAL)
        #       -> states2 (PH, PA, CO, UN)
        #         -> features (incident_energy, zpos, E_tra, E_cal, E_tot, nhits_tra, nhits_cal, nhits_tot)
      
        metrics = {
            proc: {l1: {l2: {feat: [] for feat in features} for l2 in states2} for l1 in states1}
            for proc in mc_processes
        }

        # Prepare the probability dictionaries for L1 and L2 classifications. 
        # These will store the probabilities for each event, indexed by their respective states.
        prob_l1 = {state: [] for state in states1} # UN, MU, SIGNAL
        prob_l2 = {state: [] for state in states2} # PH, PA, CO, UN

        confusion_matrix = np.zeros((4, 3), dtype=int)
        mc_mapping = {"COMP": 0, "PAIR": 1, "PHOT": 2, "OTHER": 3}
        pred_mapping = {"CO": 0, "PA": 1, "PH": 2}
        
        # Open log file for debug output
        log_file = None
        if debug:
            log_path = clean_out_dir / f"{base_name}_log.txt"
            log_file = open(log_path, "w")

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

                    # LAYER 1 — classify signal vs. background
                    # status can be "SIGNAL", "MU", or "UN".
                    # prob_bkg is the probability of that status.
                    status, prob_bkg = pipeline.signal_background_classifier(Event, debug=debug, onlyACDVeto=onlyACDVeto, log_file=log_file)

                    # prob_l1 gets populated HERE, for every single event, regardless of --debug.
                    if status in prob_l1:
                        prob_l1[status].append(prob_bkg)

                    # Only the SIGNAL events reach L2, so we only call type_of_signal() for those. 
                    # MU and UN events are never classified further, and their "event_type" is just their L1 status.
                    if status == "SIGNAL":
                        # LAYER 2 — only run for events that passed L1 as signal.
                        event_type, probability = pipeline.type_of_signal(Event, debug=debug, log_file=log_file)
                        # prob_l2 gets populated HERE, only for L1 status == "SIGNAL".
                        if event_type in prob_l2:
                            prob_l2[event_type].append(probability)
                    else:
                        # MU/UN events never reach L2, so their "event_type" is just their L1 status.
                        # However, prob_l2 never gets populated with these events. 
                        event_type, probability = status, prob_bkg
                    
                    t_classify += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    if debug:
                        # All of metrics / confusion_matrix get written inside this one call.
                        # Returns the true MC process string (or "UNKNOWN"), which the caller prints to the .etp file.
                        mc_process = record_debug_info(
                            Event, status, event_type,
                            metrics=metrics,
                            confusion_matrix=confusion_matrix,
                            mc_mapping=mc_mapping,
                            pred_mapping=pred_mapping,
                            pipeline=pipeline,
                            log_file=log_file,
                        )
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

            if debug:
                # Close the log file
                log_file.write(f"\nEnd of problems, total events processed: {i}\n")
                log_file.close()

            # ================
            # Plotting section
            # ================
            print("[INFO] Plots...")

            # L1 categories plot
            categories_l1 = ['MU', 'SIGNAL', 'UN']
            counts_l1 = [len(prob_l1[cat]) for cat in categories_l1]
            plot_bar_counts(
                categories_l1, counts_l1, ['orange', 'crimson', 'teal'],
                'L1: Signal vs Background Counts',
                clean_out_dir / f"{base_name}_L1_counts.png",
            )

            # L1 probabilities plot
            l1_hist_configs = [
                ('MU', 'MU', 'crimson', 'darkred'),
                ('SIGNAL', 'SIGNAL', 'teal', 'darkslategray'),
                ('UN', 'UN', 'gray', 'dimgray'),
            ]
            plot_overlaid_probabilities(
                prob_l1, l1_hist_configs, 'L1: Probability Distribution (TP)',
                clean_out_dir / f"{base_name}_L1_probabilities.png",
            )

            # L2 categories plot
            categories_l2 = ['CO', 'PA', 'PH', 'UN']
            counts_l2 = [len(prob_l2[cat]) for cat in categories_l2]
            plot_bar_counts(
                categories_l2, counts_l2, ['royalblue', 'forestgreen', 'darkorchid', 'gray'],
                'L2: Photon Type Classification Counts',
                clean_out_dir / f"{base_name}_L2_counts.png",
            )

            # L2 probabilities plot
            l2_hist_configs = [
                ('CO', 'CO (Compton)', 'royalblue', 'darkblue'),
                ('PA', 'PA (Pair)', 'forestgreen', 'darkgreen'),
                ('PH', 'PH (Photo)', 'darkorchid', 'purple'),
                ('UN', 'UN', 'gray', 'dimgray'),
            ]
            plot_overlaid_probabilities(
                prob_l2, l2_hist_configs, 'L2: Probability Distribution (TP)',
                clean_out_dir / f"{base_name}_L2_probabilities.png",
            )

            # Energy spectrum plot: 1x3 stacked histogram (stacked by L1 status),
            # one subplot per MC process (COMP, PAIR, PHOT).
            if debug:
                spectrum_path = clean_out_dir / f"{base_name}_energy_spectrum_stacked.png"
                plot_stacked_energy_spectrum(metrics, mc_processes, states1, spectrum_path)
                print(f"[OK] Stacked energy spectrum saved to: {spectrum_path}")

            # Confusion matrix
            if debug:
                matrix_path = clean_out_dir / f"{base_name}_confusion_matrix.png"
                plot_confusion_matrix(confusion_matrix, matrix_path)
                print(f"[OK] Confusion Matrix : {matrix_path}")    

            # 4-panel mis/classification plots
            if debug:
                process_map = {"PHOT": "PH", "PAIR": "PA", "COMP": "CO"}

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

                # Prepare a list of tuples (proc, l2, leaf) for all SIGNAL events across all MC processes
                signal_leaves = [
                    (proc, l2, metrics[proc]["SIGNAL"][l2])
                    for proc in mc_processes
                    for l2 in metrics[proc]["SIGNAL"]
                ]

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
                    },
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
                    true_dict = {process_map[proc]: [] for proc in mc_processes}
                    miss_dict = {process_map[proc]: [] for proc in mc_processes}

                    for proc, l2, leaf in signal_leaves:
                        lbl = process_map[proc]

                        if p["calc"] is not None:
                            vals = p["calc"](leaf)
                            vals = vals[~np.isnan(vals)].tolist()
                        else:
                            vals = leaf[feat]

                        if l2 == lbl:
                            true_dict[lbl].extend(vals)
                        else:
                            miss_dict[lbl].extend(vals)
                   
                    all_dict = {cat: true_dict[cat] + miss_dict[cat] for cat in true_dict}
                    
                    # Unify dynamic flattening across all channels for evaluation
                    flat_data = sum(true_dict.values(), []) + sum(miss_dict.values(), [])
                    plot_bins = get_dynamic_bins(flat_data, p["bin_rule"])
                   
                    plot_path = clean_out_dir / f"{base_name}_wrong_predictions_by_category_{p['suffix']}.png"
                    plot_type_classification_comparison(
                        true_by_category=true_dict, miss_by_category=miss_dict, all_by_category=all_dict,
                        output_path=plot_path, xlabel=p["xlabel"], title=p["title"], bins=plot_bins, log_y=p["log_y"]
                    )
                    print(f"[OK] {p['title']} saved to: {plot_path}")
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
                        for proc, l2, leaf in signal_leaves:
                            lbl = process_map[proc]
                            vals = leaf[feat_key]

                            if l2 == lbl:
                                true_dict[det].extend(vals)
                            else:
                                miss_dict[det].extend(vals)
                            
                    all_dict = {det: true_dict[det] + miss_dict[det] for det in ["TRA", "CAL", "TOT"]}
                    
                    # Compute flat dataset to automatically extract boundaries
                    flat_data = true_dict["TRA"] + true_dict["CAL"] + true_dict["TOT"] + miss_dict["TRA"] + miss_dict["CAL"] + miss_dict["TOT"]
                    plot_bins = get_dynamic_bins(flat_data, d["bin_rule"])
                        
                    plot_path = clean_out_dir / f"{base_name}_wrong_predictions_by_detector_{d['suffix']}.png"
                    plot_type_classification_comparison(
                        true_by_category=true_dict, miss_by_category=miss_dict, all_by_category=all_dict,
                        output_path=plot_path, xlabel=d["xlabel"], title=d["title"], bins=plot_bins, log_y=False,
                        categories=["TRA", "CAL", "TOT"]
                    )
                    print(f"[OK] {d['title']} saved to: {plot_path}")

# =========================
# Main function and parsing
# =========================
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Event Classifier Pipeline for MEGAlib simulation files."
    )
    parser.add_argument(
        "-i", "--input", 
        type=str, 
        default="./mini_test.sim.gz",
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
        default=None,
        help="Path to the PointNet model weights file (.pt). If omitted, defaults to "
             "the 2-class or 3-class checkpoint depending on whether -3c/--three-class is set."
    )
    parser.add_argument(
        "--disable-onlyacd", 
        action="store_false", 
        dest="only_acd_veto",
        help="Disable the strict ACD-only veto and enable the Random Forest/PCA layer "
             "(implied automatically if -rf or -pca is provided)."
    )
    parser.add_argument(
        "-rf", "--random-forest", 
        type=str, 
        default=None,
        help="Path to the Random Forest model file (.skops). Providing this flag "
             "automatically disables the ACD-only veto."
    )
    parser.add_argument(
        "-pca", "--pca", 
        type=str, 
        default=None, #"./pca_files",
        help="Path to the lookup-table pca file. Providing this flag automatically "
             "disables the ACD-only veto."
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode to print MC true processes into the output file."
    )
    parser.add_argument(
        "-3c", "--three-class",
        action="store_true",
        dest="three_class",
        help="Use the 3-class PointNet model (PointNetModels/pointnet3C.py) instead "
             "of the default 2-class model (PointNetModels/pointnet2C.py)."
    )

    # Parse the arguments from command line
    args = parser.parse_args()

    DEFAULT_RF_PATH = "./RandomForest/vega_model.skops"
    DEFAULT_MODEL_PATH_2C = "./PointNetModels/test_torch_model_params_final_26-06.pth"
    DEFAULT_MODEL_PATH_3C = "./PointNetModels/test_torch_model_params_3c_nhits.pth"

    if args.random_forest is not None and args.pca is not None:
        parser.error(
            "--random-forest/-rf and --pca/-pca are mutually exclusive: "
            "choose only one background classifier."
        )

    # Explicitly asking for -rf or -pca implies you want the veto disabled:
    # no need to also pass --disable-onlyacd.
    if args.random_forest is not None or args.pca is not None:
        args.only_acd_veto = False

    # If the veto is disabled (via --disable-onlyacd alone) but neither -rf nor
    # -pca was given, fall back to the default Random Forest model path.
    rf_path = args.random_forest
    if not args.only_acd_veto and rf_path is None and args.pca is None:
        rf_path = DEFAULT_RF_PATH

    # If -m/--model wasn't given, pick the default checkpoint matching -3c/--three-class.
    model_path = args.model
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH_3C if args.three_class else DEFAULT_MODEL_PATH_2C

    # Pass the parsed arguments directly to the main function
    main(
        input_path=args.input, 
        output_dir=args.output_dir, 
        geometry_name=args.geometry, 
        model_traced=model_path, 
        onlyACDVeto=args.only_acd_veto, 
        rf=None if args.pca is not None else rf_path,
        lookup_path=args.pca,
        debug=args.debug,
        three_class=args.three_class
    )