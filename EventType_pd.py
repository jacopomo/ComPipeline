import argparse
from tqdm import tqdm
import time
import numpy as np
import pandas as pd

import ROOT as M
from pathlib import Path
from EventType_plotter_pd import (
    make_category_counts_plot,
    make_probabilities_plot,
    make_energy_spectrum_plot,
    make_confusion_matrix_plot,
    make_particle_comparison_plot,
    make_detector_comparison_plot,
)
from EventClassifierPipeline import EventClassifierPipeline

M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")


def extract_debug_info(event, status, event_type, pipeline, log_file=None):
    """Debug-mode truth resolution + feature extraction for a single event.

    Returns (mc_process, feature_dict).
    feature_dict is None when truth is unresolvable (GetNIAs() <= 1) or when
    the true process isn't one of COMP/PAIR/PHOT (matches old behavior, which
    only ever populated `metrics` for those three).
    """
    mc_process = "UNKNOWN"
    feat_dict = None

    # GetIAAt(0) is always the primary particle's own generation record, not a
    # real interaction — the first actual interaction is GetIAAt(1). That
    # means we need at least 2 recorded interactions (GetNIAs() > 1) before
    # GetIAAt(1) is safe to call at all.
    if event.GetNIAs() > 1:
        mc_process = str(event.GetIAAt(1).GetProcess().Data())

        if mc_process in ("COMP", "PAIR", "PHOT"):
            ia_e = event.GetIAAt(0).GetSecondaryEnergy() / 1000.0  # keV -> MeV
            zpos = event.GetIAAt(1).GetPosition().Z()
            hit_data_tra = pipeline.extract_hit_data(event, detId=1)
            hit_data_cal = pipeline.extract_hit_data(event, detId=2)
            n_hits_tra = 0 if hit_data_tra is None else hit_data_tra.shape[2]
            n_hits_cal = 0 if hit_data_cal is None else hit_data_cal.shape[2]
            edep_tra = hit_data_tra[0, 3, :].sum().item() / 1000 if hit_data_tra is not None else np.nan
            edep_cal = hit_data_cal[0, 3, :].sum().item() / 1000 if hit_data_cal is not None else np.nan

            feat_dict = {
                "incident_energy": ia_e,
                "zpos": zpos,
                "E_tra": edep_tra,
                "E_cal": edep_cal,
                "E_tot": edep_tra + edep_cal,
                "nhits_tra": n_hits_tra,
                "nhits_cal": n_hits_cal,
                "nhits_tot": n_hits_tra + n_hits_cal,
            }
        else:
            if log_file:
                log_file.write(
                    f"ID {event.GetID()}: Unrecognized MC process '{mc_process}' "
                    f"classified as {status} (L1) - {event_type} (L2)\n"
                )
    else:
        if log_file:
            log_file.write(
                f"ID {event.GetID()}: GetNIAs() = {event.GetNIAs()} (not > 1) - cannot resolve MC truth\n"
            )

    if status == "SIGNAL" and event_type not in ["PH", "CO", "PA"]:
        if log_file:
            log_file.write(
                f"ID {event.GetID()}: SIGNAL event classified as '{event_type}' (expected PH, CO, or PA)\n"
            )

    return mc_process, feat_dict


def build_events_dataframe(event_rows, debug):
    """Build the per-file events DataFrame from a list of row dicts and
    downcast dtypes so this stays memory-friendly at millions-of-rows scale.
    """
    df = pd.DataFrame(event_rows)

    # --- categoricals: repeated small-vocabulary strings -> category dtype ---
    for col in ("l1_state", "l2_label", "mc_process"):
        if col in df.columns:
            df[col] = df[col].astype("category")

    # --- downcast numeric columns ---
    float_cols = ["prob_l1", "prob_l2", "incident_energy", "zpos", "E_tra", "E_cal", "E_tot"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    int_cols = ["nhits_tra", "nhits_cal", "nhits_tot"]
    for col in int_cols:
        if col in df.columns:
            # nullable Int32 since these are NaN for events without resolvable features
            df[col] = df[col].astype("Int32")

    # --- derived column: was the L2 prediction correct, given MC truth? ---
    if debug and "mc_process" in df.columns:
        truth_to_pred = {"COMP": "CO", "PAIR": "PA", "PHOT": "PH"}
        known_truth = df["mc_process"].isin(truth_to_pred.keys())
        df["correct"] = pd.NA
        df.loc[known_truth, "correct"] = (
            df.loc[known_truth, "mc_process"].map(truth_to_pred) == df.loc[known_truth, "l2_label"]
        )

    return df

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

        # --------------------------------------------------------------------
        # event_rows collects ONE dict per event. This single list replaces:
        #   - prob_l1 / prob_l2        (derive via df.groupby / df['prob_l1'])
        #   - metrics[...] nested dict (this IS the tidy replacement)
        #   - confusion_matrix         (derive via pd.crosstab / prepare_confusion_matrix
        #                               at the end, since it was never read during the run)
        # It's built once into a DataFrame after the loop — never inside it.
        # --------------------------------------------------------------------
        event_rows = []

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
                    status, prob_bkg = pipeline.signal_background_classifier(
                        Event, debug=debug, onlyACDVeto=onlyACDVeto, log_file=log_file
                    )

                    if status == "SIGNAL":
                        # LAYER 2 — only run for events that passed L1 as signal.
                        event_type, probability = pipeline.type_of_signal(Event, debug=debug, log_file=log_file)
                    else:
                        # MU/UN events never reach L2, so their "event_type" is just their L1 status.
                        event_type, probability = status, prob_bkg

                    t_classify += time.perf_counter() - t0

                    t0 = time.perf_counter()

                    row = {
                        "id_event": id_event,
                        "l1_state": status,
                        "prob_l1": prob_bkg,
                        "l2_label": event_type,
                        # only meaningful when L1 == SIGNAL; NaN otherwise (matches
                        # old prob_l2, which was only ever populated for SIGNAL events)
                        "prob_l2": probability if status == "SIGNAL" else np.nan,
                    }

                    if debug:
                        mc_process, feat_dict = extract_debug_info(
                            Event, status, event_type, pipeline=pipeline, log_file=log_file
                        )
                        row["mc_process"] = mc_process
                        if feat_dict is not None:
                            row.update(feat_dict)

                        print(
                            f"SE\nID {id_event}\nMC {mc_process}\nET {event_type}\nTP {probability:.4f}",
                            file=f_out,
                        )
                    else:
                        print(
                            f"SE\nID {id_event}\nET {event_type}\nTP {probability:.4f}",
                            file=f_out,
                        )

                    event_rows.append(row)

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
                log_file.write(f"\nEnd of problems, total events processed: {i}\n")
                log_file.close()

            # ================================================
            # Build this file's DataFrame — ONE per input file
            # ================================================
            print("[INFO] Building DataFrame...")
            df = build_events_dataframe(event_rows, debug=debug)
            df_path = clean_out_dir / f"{base_name}_events.parquet"
            df.to_parquet(df_path, index=False)
            print(f"[OK] Events DataFrame ({len(df)} rows, {df.memory_usage(deep=True).sum() / 1e6:.1f} MB) saved to: {df_path}")

            # ================
            # Plotting section
            # ================
            print("[INFO] Plots...")

            # --- L1 ---
            make_category_counts_plot(
                df, "l1_state", ['MU', 'SIGNAL', 'UN'],
                ['orange', 'crimson', 'teal'],
                'L1: Signal vs Background Counts',
                clean_out_dir / f"{base_name}_L1_counts.png",
            )

            make_probabilities_plot(
                df, "l1_state", "prob_l1",
                [
                    ('MU', 'MU', 'crimson', 'darkred'),
                    ('SIGNAL', 'SIGNAL', 'teal', 'darkslategray'),
                    ('UN', 'UN', 'gray', 'dimgray'),
                ],
                'L1: Probability Distribution (TP)',
                clean_out_dir / f"{base_name}_L1_probabilities.png",
            )

            # --- L2 ---
            make_category_counts_plot(
                df, "l2_label", ['CO', 'PA', 'PH', 'UN'],
                ['royalblue', 'forestgreen', 'darkorchid', 'gray'],
                'L2: Photon Type Classification Counts',
                clean_out_dir / f"{base_name}_L2_counts.png",
            )

            make_probabilities_plot(
                df, "l2_label", "prob_l2",
                [
                    ('CO', 'CO (Compton)', 'royalblue', 'darkblue'),
                    ('PA', 'PA (Pair)', 'forestgreen', 'darkgreen'),
                    ('PH', 'PH (Photo)', 'darkorchid', 'purple'),
                    ('UN', 'UN', 'gray', 'dimgray'),
                ],
                'L2: Probability Distribution (TP)',
                clean_out_dir / f"{base_name}_L2_probabilities.png",
            )

            if debug:
                mc_processes = ["COMP", "PAIR", "PHOT"]
                states1 = ["UN", "MU", "SIGNAL"]

                # --- Energy spectrum: stacked by L1 status, one subplot per MC process ---
                make_energy_spectrum_plot(
                    df, mc_processes, states1,
                    clean_out_dir / f"{base_name}_energy_spectrum_stacked.png",
                )
                print(f"[OK] Stacked energy spectrum saved to: {clean_out_dir / f'{base_name}_energy_spectrum_stacked.png'}")

                # --- Confusion matrix: derived + plotted, not accumulated during the run ---
                matrix_path = clean_out_dir / f"{base_name}_confusion_matrix.png"
                make_confusion_matrix_plot(df, matrix_path)
                print(f"[OK] Confusion Matrix : {matrix_path}")

                # --- 4-panel mis/classification plots ---

                # TYPE 1: Metrics split by Particle Type (CO, PA, PH)
                make_particle_comparison_plot(
                    df, "incident_energy", "Incident Energy (MeV)",
                    "Incident Energy Distribution by Category (Layer 2)",
                    clean_out_dir / f"{base_name}_wrong_predictions_by_category_energy.png",
                    bin_rule=np.arange(0, 51, 1), log_y=True,
                )
                print(f"[OK] Incident Energy Distribution by Category (Layer 2) saved to: "
                      f"{clean_out_dir / f'{base_name}_wrong_predictions_by_category_energy.png'}")

                make_particle_comparison_plot(
                    df, "zpos", "Incidence Z Position (cm)",
                    "Z Vertex Distribution by Category (Layer 2)",
                    clean_out_dir / f"{base_name}_wrong_predictions_by_category_zpos.png",
                    bin_rule=np.linspace(-15, 30, 51), log_y=False,
                )
                print(f"[OK] Z Vertex Distribution by Category (Layer 2) saved to: "
                      f"{clean_out_dir / f'{base_name}_wrong_predictions_by_category_zpos.png'}")

                make_particle_comparison_plot(
                    df, "erat", "Energy Ratio (E_tra / E_cal)",
                    "Deposited Energy Ratio (TRA / CAL) by Category (Layer 2)",
                    clean_out_dir / f"{base_name}_wrong_predictions_by_category_erat.png",
                    bin_rule="linspace", log_y=False,
                    calc=lambda d: (d["E_tra"] / d["E_cal"]).where(d["E_cal"] > 0),
                )
                print(f"[OK] Deposited Energy Ratio (TRA / CAL) by Category (Layer 2) saved to: "
                      f"{clean_out_dir / f'{base_name}_wrong_predictions_by_category_erat.png'}")

                make_particle_comparison_plot(
                    df, "nrat", "Number of Hits Ratio (n_tra / n_cal)",
                    "Number of Hits Ratio (TRA / CAL) by Category (Layer 2)",
                    clean_out_dir / f"{base_name}_wrong_predictions_by_category_nrat.png",
                    bin_rule="linspace", log_y=False,
                    calc=lambda d: (d["nhits_tra"] / d["nhits_cal"]).where(d["nhits_cal"] > 0),
                )
                print(f"[OK] Number of Hits Ratio (TRA / CAL) by Category (Layer 2) saved to: "
                      f"{clean_out_dir / f'{base_name}_wrong_predictions_by_category_nrat.png'}")

                # TYPE 2: Metrics split by Detector Component (TRA, CAL, TOT)
                make_detector_comparison_plot(
                    df, {"TRA": "E_tra", "CAL": "E_cal", "TOT": "E_tot"},
                    "Deposited energy (MeV)",
                    "Deposited Energy Distribution by Detector (Layer 2)",
                    clean_out_dir / f"{base_name}_wrong_predictions_by_detector_edep.png",
                    bin_rule="linspace",
                )
                print(f"[OK] Deposited Energy Distribution by Detector (Layer 2) saved to: "
                      f"{clean_out_dir / f'{base_name}_wrong_predictions_by_detector_edep.png'}")

                make_detector_comparison_plot(
                    df, {"TRA": "nhits_tra", "CAL": "nhits_cal", "TOT": "nhits_tot"},
                    "Number of hits",
                    "Distribution of Number of Hits by Detector (Layer 2)",
                    clean_out_dir / f"{base_name}_wrong_predictions_by_detector_nhits.png",
                    bin_rule="arange",
                )
                print(f"[OK] Distribution of Number of Hits by Detector (Layer 2) saved to: "
                      f"{clean_out_dir / f'{base_name}_wrong_predictions_by_detector_nhits.png'}")

# =========================
# Main function and parsing
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Event Classifier Pipeline for MEGAlib simulation files."
    )

    parser.add_argument("-i", "--input", type=str,
                         default="./ComPair23_1MeV_50MeV_powerlaw.p1.inc10.id1.sim.gz",
                         help="Path to a single .sim/.sim.gz file OR to a directory containing them.")
    parser.add_argument("-o", "--output-dir", type=str, default="./output_etp",
                         help="Directory where output .etp files will be saved.")
    parser.add_argument("-g", "--geometry", type=str,
                         default="../../simuComPair/Geometry/ComPair_23/ComPair23.geo.setup",
                         help="Path to the MEGAlib geometry setup file (.geo.setup).")
    parser.add_argument("-m", "--model", type=str,
                         default="./PointNetModels/test_torch_model_params_final_26-06.pth",
                         help="Path to the PointNet model weights file (.pt).")
    parser.add_argument("--disable-onlyacd", action="store_false", dest="only_acd_veto",
                         help="Disable the strict ACD-only veto and enable the Random Forest/PCA layer.")
    parser.add_argument("-rf", "--random-forest", type=str, default=None,
                         help="Path to the Random Forest model pickle file (used only if ACD-only veto is disabled).")
    parser.add_argument("-pca", "--pca", type=str, default=None, help="LookupTable pca.")
    parser.add_argument("--debug", action="store_true",
                         help="Enable debug mode to print MC true processes into the output file.")

    args = parser.parse_args()

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