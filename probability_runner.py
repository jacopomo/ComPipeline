#!/usr/bin/env python3
"""Run the classifier on binned simulation files and collect CO/PA probabilities.

Edit the configuration below, then run:

    python3 probability_runner.py

Each input file contributes one row to CSV_OUTPUT. Photoelectric, background,
and unclassified outcomes are excluded from the CO/PA binomial estimate.
"""

import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENERGIES = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

SCRIPT = "EventType_withhisto.py"
GEOMETRY = "Geometry/ComPair_23/ComPair23.geo.setup"
MODEL = "PointNetModels/test_torch_model_params_train_2C_nhits2.pth"

INPUT_TEMPLATE = "/scratch/ComPair/Sim_1MeV_50MeV_forMLtrain_ComPair23/E_HE/E_{energy1}MeV_{energy2}MeV/ComPair23_{energy1}MeV_{energy2}MeV_ADCveto_Comp.p1.inc1.id1.sim.gz"
OUTPUT_TEMPLATE = "output_etp/E_{energy1}MeV_{energy2}MeV/"
CSV_OUTPUT = Path("output_etp/compton_pair_probabilities_HE.csv")
# Additional classifier flags can be added here, for example:
# EXTRA_FLAGS = ["--disable-onlyacd", "-rf", "RandomForest/vega_model.skops"]
EXTRA_FLAGS = []

DRY_RUN = False
STOP_ON_ERROR = True
RESET_CSV = True

# ---------------------------------------------------------------------------


def build_command(energy):
    input_path = INPUT_TEMPLATE.format(energy1=energy, energy2=energy+1)
    output_path = OUTPUT_TEMPLATE.format(energy1=energy, energy2=energy+1)
    energy_bin = f"{energy}-{energy + 1}MeV"

    command = [
        sys.executable,
        SCRIPT,
        "-g", GEOMETRY,
        "-i", input_path,
        "-m", MODEL,
        "-o", output_path,
        "--probability-only",
        "--probability-csv", str(CSV_OUTPUT),
        "--energy-bin", energy_bin,
    ] + EXTRA_FLAGS
    return command, input_path, output_path


def main():
    if RESET_CSV and not DRY_RUN:
        CSV_OUTPUT.unlink(missing_ok=True)

    failures = []
    for energy in ENERGIES:
        command, input_path, output_path = build_command(energy)
        Path(output_path).mkdir(parents=True, exist_ok=True)

        print(f"\n=== Running energy bin = {energy}-{energy + 1} MeV ===")
        print("Command:", " ".join(command))

        if DRY_RUN:
            continue
        if not Path(input_path).exists():
            print(f"[WARNING] Input file not found: {input_path}")

        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"[ERROR] Run failed for energy={energy} (exit code {result.returncode})")
            failures.append(energy)
            if STOP_ON_ERROR:
                break
        else:
            print(f"[OK] Finished energy={energy}")

    print("\n=== Summary ===")
    print(f"Total: {len(ENERGIES)}, Failed: {len(failures)}")
    print(f"CSV: {CSV_OUTPUT}")
    if failures:
        print("Failed energies:", failures)


if __name__ == "__main__":
    main()
