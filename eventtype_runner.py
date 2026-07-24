#!/usr/bin/env python3
"""
Runs EventType_withhisto.py once per energy value in ENERGIES, substituting
the value into both the input .sim.gz filename and the output directory.

Usage:
    python3 run_energies.py

Edit ENERGIES below to add/remove/change energy values.
"""

import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration - edit these as needed
# ---------------------------------------------------------------------------

ENERGIES = [80, 100, 125, 160, 200, 250, 315, 500, 1000, 1600, 3200, 5000, 16000, 25000]  # <-- update this list with your energy values (keV)

SCRIPT = "EventType_withhisto.py"
GEOMETRY = "../Geometry/ComPair_23/ComPair23.geo.setup"
MODEL = "./PointNetModels/test_torch_model_params_train_2C_nhits2.pth"

INPUT_TEMPLATE = "simgz/grant_single_energies/ComPair2_{energy}keV.inc1.id1.sim.gz"
OUTPUT_TEMPLATE = "output_etp/grant_single_energies/nhits2/{energy}keV/"

EXTRA_FLAGS = ["--debug"]

# Set to True to just print the commands without running them (dry run)
DRY_RUN = False

# Stop on first failure? If False, continues to next energy even if one fails.
STOP_ON_ERROR = True

# ---------------------------------------------------------------------------


def build_command(energy):
    input_path = INPUT_TEMPLATE.format(energy=energy)
    output_path = OUTPUT_TEMPLATE.format(energy=energy)

    cmd = [
        sys.executable, SCRIPT,
        "-g", GEOMETRY,
        "-i", input_path,
        "-m", MODEL,
        "-o", output_path,
    ] + EXTRA_FLAGS

    return cmd, input_path, output_path


def main():
    failures = []

    for energy in ENERGIES:
        cmd, input_path, output_path = build_command(energy)

        # Ensure output directory exists
        Path(output_path).mkdir(parents=True, exist_ok=True)

        print(f"\n=== Running energy = {energy} keV ===")
        print("Command:", " ".join(cmd))

        if DRY_RUN:
            continue

        # Check input file exists before running (helpful early warning)
        if not Path(input_path).exists():
            print(f"[WARNING] Input file not found: {input_path}")

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"[ERROR] Run failed for energy={energy} (exit code {result.returncode})")
            failures.append(energy)
            if STOP_ON_ERROR:
                break
        else:
            print(f"[OK] Finished energy={energy}")

    print("\n=== Summary ===")
    print(f"Total: {len(ENERGIES)}, Failed: {len(failures)}")
    if failures:
        print("Failed energies:", failures)


if __name__ == "__main__":
    main()