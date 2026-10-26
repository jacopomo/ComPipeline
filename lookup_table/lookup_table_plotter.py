import pandas as pd
import matplotlib.pyplot as plt

# Path to your CSV file
csv_file = "1M_1keV_50MeV.csv"

# Read the CSV
df = pd.read_csv(csv_file)

# Extract columns
energy = (
    df["mean_deposited_energy"]
    if "mean_deposited_energy" in df.columns
    else df["energy_bin"]
)
compton = df["compton_probability"]
pair = df["pair_probability"]
phot = df["photoelectric_probability"]
compton_err = df["compton_err"]
pair_err = df["pair_err"]
phot_err = df["phot_err"]

# Create plot
plt.figure(figsize=(8, 6))

plt.errorbar(
    energy,
    compton,
    compton_err,
    marker="o",
    label="Compton"
)

plt.errorbar(
    energy,
    pair,
    pair_err,
    marker="o",
    label="Pair"
)

plt.errorbar(
    energy,
    phot,
    phot_err,
    marker="o",
    label="Photoelectric"
)
# Labels and formatting
plt.xlabel("Measured Energy [MeV]")
plt.ylabel("Probability")
plt.title("Different Event Type Probabilities")

plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("compton_pair_probability_1M_1keV_50MeV.png", dpi=300)
plt.show()