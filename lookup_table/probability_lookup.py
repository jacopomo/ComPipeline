import re

import numpy as np
import pandas as pd


class ProbabilityLookup:
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        self.energy = np.asarray(
            [self._energy_bin_center(value) for value in df["energy_bin"]],
            dtype=float,
        )
        self.compton_probability = df["compton_probability"].to_numpy(dtype=float)
        self.pair_probability = df["pair_probability"].to_numpy(dtype=float)

        order = np.argsort(self.energy)
        self.energy = self.energy[order]
        self.compton_probability = self.compton_probability[order]
        self.pair_probability = self.pair_probability[order]

    @staticmethod
    def _energy_bin_center(value):
        numbers = [float(number) for number in re.findall(r"\d+(?:\.\d+)?", str(value))]
        if not numbers:
            raise ValueError(f"Could not parse energy bin {value!r}")
        return sum(numbers[:2]) / min(len(numbers), 2)

    def get_probabilities(self, energy):
        """Return (Compton probability, Pair probability) by interpolation.

        Values outside the table range use the nearest endpoint value.
        """

        p_compton = np.interp(energy, self.energy, self.compton_probability)
        p_pair = np.interp(energy, self.energy, self.pair_probability)
        probability_sum = p_compton + p_pair
        if probability_sum <= 0:
            raise ValueError(f"Lookup table has invalid probabilities at energy {energy}")
        return p_compton / probability_sum, p_pair / probability_sum