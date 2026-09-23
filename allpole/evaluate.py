"""Scoring: per-utterance RMSE (Hz) on masked frames, averaged over utterances."""
import csv
from pathlib import Path

import torch

FORMANTS = ["F1", "F2", "F3", "F4"]


def utterance_rmse(formants_hat, utt):
    """RMSE per formant over the scored frames of one utterance. formants_hat: (frames, 4) in Hz."""
    n = min(len(formants_hat), len(utt["formants"]))
    diff = formants_hat[:n].float().cpu() - utt["formants"][:n]
    diff = diff[utt["mask"][:n]]
    return torch.sqrt(torch.nanmean(diff ** 2, dim=0)).tolist()


class Scorer:
    """Collects per-utterance RMSEs, prints the mean and writes results/<name>/<split>.csv."""

    def __init__(self, name, split, out_dir="results"):
        self.path = Path(out_dir) / name / f"{split}.csv"
        self.rows = []

    def add(self, formants_hat, utt):
        self.rows.append([utt["key"]] + utterance_rmse(formants_hat, utt))

    def finish(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["key"] + FORMANTS)
            w.writerows(self.rows)
        means = torch.tensor([r[1:] for r in self.rows]).nanmean(dim=0).tolist()
        print("RMSE (Hz)  " + "  ".join(f"{f}: {m:.0f}" for f, m in zip(FORMANTS, means))
              + f"  mean: {sum(means) / 4:.0f}")
        print(f"Per-utterance results written to {self.path}")
        return means
