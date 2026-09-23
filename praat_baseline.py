"""Praat baseline: Burg formant analysis via parselmouth, sampled on the 10 ms VTR grid.

    python praat_baseline.py --data_root /path/to/data --split test
"""
import argparse
import os

import numpy as np
import parselmouth
import torch
from tqdm import tqdm

from allpole.data import load_split
from allpole.evaluate import Scorer


def praat_formants(x, fs=16000, hop_length=160, win_length=512):
    """x: (1, 1, T) waveform -> (frames, 4) formants in Hz (NaN where Praat finds none)."""
    x = -1.0 * x.flatten().double().numpy()
    formant = parselmouth.Sound(x, fs).to_formant_burg(
        time_step=hop_length / fs, max_number_of_formants=5,
        maximum_formant=5500, window_length=win_length / fs)
    times = np.arange(0, len(x) / fs, hop_length / fs)
    return torch.tensor([[formant.get_value_at_time(i, t) for i in range(1, 5)] for t in times]).float()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", default=os.environ.get("DATA_ROOT"), required="DATA_ROOT" not in os.environ)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    args = parser.parse_args()

    scorer = Scorer("praat", args.split)
    for utt in tqdm(load_split(args.data_root, args.split)):
        scorer.add(praat_formants(utt["waveform"]), utt)
    scorer.finish()
