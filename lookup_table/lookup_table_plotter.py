import pandas as pd
import matplotlib.pyplot as plt

# Path to your CSV file
csv_file = "compton_pair_probabilities.csv"

# Read the CSV
df = pd.read_csv(csv_file)

# Extract columns
energy = df["energy_bin"]
compton = df["compton_probability"]
pair = df["pair_probability"]
phot = df["photoelectric_probability"]

# Create plot
plt.figure(figsize=(8, 6))

plt.plot(
    energy,
    compton,
    marker="o",
    label="Compton"
)

plt.plot(
    energy,
    pair,
    marker="o",
    label="Pair"
)

plt.plot(
    energy,
    phot,
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
plt.savefig("compton_pair_probability.png", dpi=300)
plt.show()