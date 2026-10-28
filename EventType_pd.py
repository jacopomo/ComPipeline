"""Run the two-layer event classifier over MEGAlib .sim files.

For every input file this writes an .etp file (plus a log file in debug mode)
and returns ONE pandas DataFrame with one row per event.

Columns
-------
file, id                      : which file / event
l1, l1_prob                   : layer 1 label ("SIGNAL", "MU", "UN") and its probability
l2, l2_prob                   : layer 2 label ("CO", "PA", "PH") and probability (None if not SIGNAL)

Debug mode only (needs MC truth); NaN where not applicable:
mc_true                       : true first-interaction process, or "UNKNOWN"
Ein                           : primary energy [MeV]
zpos                          : z of first interaction
n_hits_tot / _tra / _cal      : number of hits (total / tracker / calorimeter)
E_dep_tot / _tra / _cal       : deposited energy [MeV]
"""
import argparse
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path

import pandas as pd
import ROOT
from tqdm import tqdm

from EventClassifierPipeline import EventClassifierPipeline, csv_file as DEFAULT_ENERGY_LOOKUP_PATH

KEV_TO_MEV = 1e-3
TRACKER_ID, CALORIMETER_ID = 1, 2
PROGRESS_INTERVAL = 500  # events between progress-bar / file flushes

# True MC processes we extract features for. Any other process counts as "OTHER".
MC_PROCESSES = ("COMP", "PAIR", "PHOT")
# Layer-2 predictions we expect for SIGNAL events.
L2_LABELS = ("CO", "PA", "PH")


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #
def init_megalib():
    """Load MEGAlib and initialise it. The returned object must stay alive."""
    ROOT.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")
    megalib_global = ROOT.MGlobal()
    megalib_global.Initialize()
    return megalib_global


def load_geometry(geometry_name):
    geometry = ROOT.MDGeometryQuest()
    if not geometry.ScanSetupFile(ROOT.MString(geometry_name)):
        raise RuntimeError(f"Unable to load geometry {geometry_name}")
    print(f"Geometry {geometry_name} loaded successfully!")
    return geometry


def find_input_files(input_path):
    """Return a single file, or all .sim / .sim.gz files in a directory."""
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted([*path.glob("*.sim"), *path.glob("*.sim.gz")])
    raise FileNotFoundError(f"Input path '{input_path}' does not exist or is invalid.")


def strip_sim_suffix(filename):
    """'run.v2.sim.gz' -> 'run.v2'"""
    return filename.removesuffix(".gz").removesuffix(".sim")


@contextmanager
def timed(totals, stage):
    """Add the elapsed time of the `with` block to totals[stage]."""
    start = time.perf_counter()
    try:
        yield
    finally:
        totals[stage] += time.perf_counter() - start


# --------------------------------------------------------------------------- #
# Per-event steps
# --------------------------------------------------------------------------- #
def classify_event(event, pipeline, only_acd_veto, debug, log_file):
    """Run both classifier layers. L2 only runs for events L1 calls SIGNAL."""
    l1, l1_prob = pipeline.signal_background_classifier(
        event, debug=debug, onlyACDVeto=only_acd_veto, log_file=log_file
    )
    if l1 == "SIGNAL":
        l2, l2_prob = pipeline.type_of_signal(event, debug=debug, log_file=log_file)
    else:
        l2, l2_prob = None, None  # MU / UN events never reach L2
    return {"l1": l1, "l1_prob": l1_prob, "l2": l2, "l2_prob": l2_prob}


def summarize_hits(hit_data):
    """Return (number of hits, deposited energy in MeV); (0, 0.0) if no hits."""
    if hit_data is None:
        return 0, 0.0
    return hit_data.shape[2], hit_data[0, 3, :].sum().item() * KEV_TO_MEV


def extract_features(event, pipeline):
    """Features for one event with a known MC process."""
    n_tra, e_tra = summarize_hits(pipeline.extract_hit_data(event, detId=TRACKER_ID))
    n_cal, e_cal = summarize_hits(pipeline.extract_hit_data(event, detId=CALORIMETER_ID))
    return {
        "Ein": event.GetIAAt(0).GetSecondaryEnergy() * KEV_TO_MEV,
        "zpos": event.GetIAAt(1).GetPosition().Z(),
        "n_hits_tot": n_tra + n_cal,
        "n_hits_tra": n_tra,
        "n_hits_cal": n_cal,
        "E_dep_tot": e_tra + e_cal,
        "E_dep_tra": e_tra,
        "E_dep_cal": e_cal,
    }


def extract_mc_truth(event, pipeline):
    """True MC process, plus features if it is one of MC_PROCESSES."""
    # GetIAAt(0) is the primary's own generation record, not a real interaction.
    # The first actual interaction is GetIAAt(1), which needs GetNIAs() > 1.
    if event.GetNIAs() <= 1:
        return {"mc_true": "UNKNOWN"}

    mc_true = str(event.GetIAAt(1).GetProcess().Data())
    truth = {"mc_true": mc_true}
    if mc_true in MC_PROCESSES:
        truth.update(extract_features(event, pipeline))
    return truth


def log_oddities(event, row, log_file):
    """Write unusual events to the log file (debug mode)."""
    if log_file is None:
        return
    event_id, mc_true, l1, l2 = row["id"], row["mc_true"], row["l1"], row["l2"]

    if mc_true == "UNKNOWN":
        log_file.write(f"ID {event_id}: GetNIAs() = {event.GetNIAs()} (not > 1) - cannot resolve MC truth\n")
    elif mc_true not in MC_PROCESSES:
        log_file.write(f"ID {event_id}: Unrecognized MC process '{mc_true}' classified as {l1} (L1) - {l2} (L2)\n")

    if l1 == "SIGNAL" and l2 not in L2_LABELS:
        log_file.write(f"ID {event_id}: SIGNAL event classified as '{l2}' (expected {', '.join(L2_LABELS)})\n")


def format_etp_entry(row):
    """One event's block in the .etp file. MC line only if truth is present."""
    # Non-SIGNAL events have no L2 result, so fall back to the L1 label/probability.
    label = row["l2"] if row["l2"] is not None else row["l1"]
    prob = row["l2_prob"] if row["l2_prob"] is not None else row["l1_prob"]

    lines = ["SE", f"ID {row['id']}"]
    if "mc_true" in row:
        lines.append(f"MC {row['mc_true']}")
    lines += [f"ET {label}", f"TP {prob:.4f}"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Per-file loop
# --------------------------------------------------------------------------- #
def process_file(path, out_dir, geometry, pipeline, only_acd_veto, debug):
    """Classify every event in one file. Returns a list of per-event row dicts."""
    base_name = strip_sim_suffix(path.name)
    etp_path = out_dir / f"{base_name}.etp"
    print(f"\n[INFO] Processing file: {path.name} -> Target Output: {etp_path}")

    reader = ROOT.MFileEventsSim(geometry)
    if not reader.Open(ROOT.MString(str(path))):
        print(f"Unable to open file {path}. Skipping!")
        return []

    rows = []
    timers = defaultdict(float)

    with ExitStack() as stack:
        stack.callback(reader.Close)
        f_out = stack.enter_context(open(etp_path, "w"))
        log_file = (
            stack.enter_context(open(out_dir / f"{base_name}_log.txt", "w")) if debug else None
        )

        with tqdm(desc="Events", unit=" evt") as pbar:
            while True:
                with timed(timers, "read"):
                    event = reader.GetNextEvent()
                if not event:
                    break
                ROOT.SetOwnership(event, True)  # Python now owns (and frees) the event

                row = {"id": event.GetID()}

                with timed(timers, "classify"):
                    row.update(classify_event(event, pipeline, only_acd_veto, debug, log_file))

                if debug:
                    with timed(timers, "mc_truth"):
                        row.update({"file": base_name})
                        row.update(extract_mc_truth(event, pipeline))
                        log_oddities(event, row, log_file)

                with timed(timers, "write"):
                    f_out.write(format_etp_entry(row))

                rows.append(row)
                pbar.update(1)

                n_events = len(rows)
                if n_events % PROGRESS_INTERVAL == 0:
                    pbar.set_postfix(
                        {stage: f"{total / n_events * 1000:.1f}ms" for stage, total in timers.items()}
                    )
                    f_out.flush()

        if log_file is not None:
            log_file.write(f"\nEnd of problems, total events processed: {len(rows)}\n")

    n_events = len(rows)
    print(f"\nDONE. {n_events} events processed.")
    if n_events:
        for stage, total in timers.items():
            print(f"  avg {stage:<8}: {total / n_events * 1000:.2f} ms/evt")
    print(f"[OK] File {path.name} completed successfully. Saved to {etp_path}")
    return rows


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(
    input_path,
    output_dir,
    geometry_name,
    model_traced,
    only_acd_veto=True,
    random_forest_path=None,
    lookup_path=None,
    debug=False,
    three_class=False,
    energy_probability_lookup_path=DEFAULT_ENERGY_LOOKUP_PATH,
    use_energy_dependent_probability=True,
):
    """Classify all events and return a DataFrame with one row per event."""
    megalib_global = init_megalib()  # noqa: F841  (must stay alive while we work)
    geometry = load_geometry(geometry_name)

    files = find_input_files(input_path)
    if not files:
        print(f"No .sim or .sim.gz files found in '{input_path}'. Exiting.")
        return pd.DataFrame()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = EventClassifierPipeline(
        model_traced,
        onlyACDVeto=only_acd_veto,
        random_forest_path=random_forest_path,
        lookup_path=lookup_path,
        three_class=three_class,
        energy_probability_lookup_path=energy_probability_lookup_path,
        use_energy_dependent_probability=use_energy_dependent_probability,
    )

    print("Starting event processing loop...")
    rows = []
    for path in files:
        rows.extend(process_file(path, out_dir, geometry, pipeline, only_acd_veto, debug))

    df = pd.DataFrame(rows)
        
    df.to_csv(out_dir / "my_data.csv", index=False)
    return df


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
        "--energy-probability-lookup",
        type=str,
        default=DEFAULT_ENERGY_LOOKUP_PATH,
        help="CSV lookup table for energy-dependent CO/PA probabilities. "
             f"Default: {DEFAULT_ENERGY_LOOKUP_PATH}",
    )
    parser.add_argument(
        "--no-energy-dependence",
        action="store_true",
        help="Disable energy-dependent CO/PA probability combination for the two-class model.",
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

    use_energy_dependent_probability = (
        not args.three_class and not args.no_energy_dependence
    )

    DEFAULT_RF_PATH = "./RandomForest/vega_model.skops"
    DEFAULT_MODEL_PATH_2C = "./PointNetModels/test_torch_model_params_2c_nhits2.pth"
    DEFAULT_MODEL_PATH_3C = "./PointNetModels/test_torch_model_params_3c_nhits2.pth"

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
        only_acd_veto=args.only_acd_veto, 
        random_forest_path=None if args.pca is not None else rf_path,
        lookup_path=args.pca,
        debug=args.debug,
        three_class=args.three_class,
        energy_probability_lookup_path=args.energy_probability_lookup,
        use_energy_dependent_probability=use_energy_dependent_probability,
    )
