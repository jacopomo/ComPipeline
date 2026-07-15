import argparse
from tqdm import tqdm
import time
import numpy as np
import pandas as pd

import ROOT as M
from pathlib import Path
from EventType_plotter_pd import plot_bar_counts, plot_overlaid_probabilities, plot_stacked_energy_spectrum, plot_confusion_matrix, plot_particle_comparison_plot
from EventClassifierPipeline import EventClassifierPipeline

M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")
def record_debug_info(event, status, event_type, row_dict, confusion_matrix, mc_mapping, pred_mapping, pipeline, log_file=None):
    """Debug-mode bookkeeping for a single event.

    Called once per event, only when --debug is set. Returns the true MC
    process string (or "UNKNOWN"), which the caller prints to the .etp file.

    Logs to file:
      - Events where GetNIAs() <= 1 (unresolvable MC truth)
      - Events where a SIGNAL event (L1) does not classify as PH, CO, or PA (L2)
    """
    mc_process = "UNKNOWN"

    if event.GetNIAs() > 1:
        mc_process = str(event.GetIAAt(1).GetProcess().Data())

        # --- confusion_matrix: predicted L2 label vs. true MC process ---
        if status == "SIGNAL" and mc_process in mc_mapping and event_type in pred_mapping:
            true_idx = mc_mapping[mc_process]
            pred_idx = pred_mapping[event_type]
            confusion_matrix[true_idx, pred_idx] += 1
        if status == "SIGNAL" and mc_process not in mc_mapping:
            true_idx = mc_mapping["OTHER"]
            pred_idx = pred_mapping[event_type]
            confusion_matrix[true_idx, pred_idx] += 1

        # Only store metrics for the three MC processes we care about (COMP, PAIR, PHOT) not RAYL!
        if mc_process in ["COMP", "PAIR", "PHOT"]:

            # classification_status is only meaningful for events that actually
            # reached L2 (i.e. SIGNAL). For MU/UN background events, event_type
            # is NaN (never classified), so mark classification_status as NaN too
            # instead of silently comparing against NaN or crashing.
            classification_map = {"COMP": "CO", "PAIR": "PA", "PHOT": "PH"}
            if status == "SIGNAL":
                classification_status = classification_map[mc_process] == event_type
            else:
                classification_status = np.nan

            # Get the info (features)
            ia_e = event.GetIAAt(0).GetSecondaryEnergy() / 1000.0  # keV -> MeV
            zpos = event.GetIAAt(1).GetPosition().Z()
            hit_data_tra = pipeline.extract_hit_data(event, detId=1)
            hit_data_cal = pipeline.extract_hit_data(event, detId=2)
            n_hits_tra = 0 if hit_data_tra is None else hit_data_tra.shape[2]
            n_hits_cal = 0 if hit_data_cal is None else hit_data_cal.shape[2]
            edep_tra = hit_data_tra[0, 3, :].sum().item() / 1000 if hit_data_tra is not None else np.nan
            edep_cal = hit_data_cal[0, 3, :].sum().item() / 1000 if hit_data_cal is not None else np.nan

            extracted_features = {
                "mc_process": mc_process,
                "classification_status": classification_status,
                "incident_energy": ia_e,
                "zpos": zpos,
            }

        else:
            # Log events with unrecognized MC process (not COMP, PAIR, PHOT) —
            # only meaningful/interesting for SIGNAL events; background events
            # with e.g. RAYL are expected and not worth logging.
            if status == "SIGNAL" and log_file:
                log_file.write(f"ID {event.GetID()}: Unrecognized MC process '{mc_process}' classified as {status} (L1) - {event_type} (L2)\n")
    else:
        # Log events with unresolvable MC truth (GetNIAs() <= 1)
        if log_file:
            log_file.write(f"ID {event.GetID()}: GetNIAs() = {event.GetNIAs()} (not > 1) - cannot resolve MC truth\n")

    # Log SIGNAL events that don't classify as PH, CO, or PA
    if status == "SIGNAL" and event_type not in ["PH", "CO", "PA"]:
        if log_file:
            log_file.write(f"ID {event.GetID()}: SIGNAL event classified as '{event_type}' (expected PH, CO, or PA)\n")

    combined_dict = row_dict.copy()

    if mc_process in ["COMP", "PAIR", "PHOT"]:
        combined_dict.update(extracted_features)

    return mc_process, combined_dict

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

        data = []

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
                    # status can be "SIGNAL", "MU", or "UN", prob_bkg is the probability of that status.
                    status, prob_bkg = pipeline.signal_background_classifier(Event, debug=debug, onlyACDVeto=onlyACDVeto, log_file=log_file)


                    # Only the SIGNAL events reach L2, so we only call type_of_signal() for those. 
                    # MU and UN events are never classified further, and their "event_type" is just their L1 status.
                    if status == "SIGNAL":
                        # LAYER 2 — only run for events that passed L1 as signal.
                        event_type, probability = pipeline.type_of_signal(Event, debug=debug, log_file=log_file)
                    else:
                        # MU/UN events never reach L2, so their "event_type" is just nan
                        event_type, probability = np.nan, np.nan
                    
                    # If debug is disabled, this is all the data we have for the event. 
                    row_dict = {"id": id_event, "l1_output": status, "l1_prob": prob_bkg, "l2_output": event_type, "l2_prob": probability}
                    
                    t_classify += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    if debug:
                        # Since debug is on we can store MC process and many more features in each row
                        # Returns the true MC process string (or "UNKNOWN"), which the caller prints to the .etp file.
                        # Also returns the combined row_dict with additional features for the dataframe.
                        mc_process, row_dict = record_debug_info(
                            Event, status, event_type,
                            row_dict=row_dict,
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
                    data.append(row_dict)
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
    
            # Create the pandas dataframe
            df = pd.DataFrame(data)
            print(f"\nDONE. {i} events processed.")
            print(f"  avg read    : {t_read     / i * 1000:.2f} ms/evt")
            print(f"  avg classify: {t_classify / i * 1000:.2f} ms/evt")
            print(f"  avg write   : {t_write    / i * 1000:.2f} ms/evt")
            print(f"[OK] File {fn_in.name} completed successfully. Saved to {fn_out}")

            if debug:
                # Close the log file
                log_file.write(f"\nEnd of problems, total events processed: {i}\n")
                log_file.close()

            plot_results(df, confusion_matrix, base_name, clean_out_dir, debug=debug)

def plot_results(df, confusion_matrix, base_name, clean_out_dir, debug=False):
    print("[INFO] Plots...")

    # L1 categories plot
    l1_cat_path = clean_out_dir / f"{base_name}_L1_counts.png"
    plot_bar_counts(
        df, "l1_output", ['orange', 'crimson', 'teal'],
        'L1: Signal vs Background Counts',
        l1_cat_path,
    )
    print(f"[OK] L1 categories plot saved to: {l1_cat_path}")

    # L1 probabilities plot
    l1_prob_path = clean_out_dir / f"{base_name}_L1_probabilities.png"
    plot_overlaid_probabilities(
        df, 'l1_prob', 'l1_output', 'L1: Probability Distribution (TP)',
        l1_prob_path,
    )
    print(f"[OK] L1 probabilities plot saved to: {l1_prob_path}")

    # L2 categories plot
    l2_cat_path = clean_out_dir / f"{base_name}_L2_counts.png"
    plot_bar_counts(
        df, "l2_output", ['royalblue', 'forestgreen', 'darkorchid', 'gray'],
        'L2: Photon Type Classification Counts',
        l2_cat_path,
    )
    print(f"[OK] L2 categories plot saved to: {l2_cat_path}")

    # L2 probabilities plot
    l2_prob_path = clean_out_dir / f"{base_name}_L2_probabilities.png"
    plot_overlaid_probabilities(
        df, 'l2_prob', 'l2_output', 'L2: Probability Distribution (TP)',
        l2_prob_path,
    )
    print(f"[OK] L2 probabilities plot saved to: {l2_prob_path}")

    if debug:
        # Energy spectrum plot: 1x3 stacked histogram (stacked by L1 status), 
        # one subplot per MC process (COMP, PAIR, PHOT).
        spectrum_path = clean_out_dir / f"{base_name}_energy_spectrum_stacked.png"
        plot_stacked_energy_spectrum(df, spectrum_path)
        print(f"[OK] Stacked energy spectrum saved to: {spectrum_path}")

        # Confusion matrix
        matrix_path = clean_out_dir / f"{base_name}_confusion_matrix.png"
        plot_confusion_matrix(confusion_matrix, matrix_path)
        print(f"[OK] Confusion Matrix : {matrix_path}")    
        
        # Confusion plot for incident energy
        plot_path = clean_out_dir / f"{base_name}_incident_energy_confusion_plot.png"
        plot_particle_comparison_plot(df, feature_col="incident_energy", binning_strategy="arange", output_path=plot_path)
        print(f"[OK] Incident Energy Comparison Plot saved to: {plot_path}")

        # Confusion plot for zpos 
        plot_path = clean_out_dir / f"{base_name}_zpos_confusion_plot.png"
        plot_particle_comparison_plot(df, feature_col="zpos", binning_strategy="linspace", log_y=False, output_path=plot_path)
        print(f"[OK] Z Position Comparison Plot saved to: {plot_path}")

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