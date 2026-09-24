"""LP baseline: autocorrelation-method linear prediction per frame, formants from the roots of A(z).

    python lp_baseline.py --data_root /path/to/data --split test
"""
import argparse
import os

from tqdm import tqdm

from allpole.data import load_split
from allpole.dsp import Emphasis, LinearPredictor, get_formants
from allpole.evaluate import Scorer


def lp_formants(x, order=16):
    """x: (1, 1, T) waveform -> (frames, 4) formants in Hz, and the LP polynomial."""
    x = x.double()  # the reported baseline numbers were computed in float64
    x_emph = Emphasis(0.97)(-1.0 * x)
    a, _ = LinearPredictor(order=order).estimate(x_emph[:, 0, :])
    return get_formants(a), a


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", default=os.environ.get("DATA_ROOT"), required="DATA_ROOT" not in os.environ)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--order", type=int, default=18)
    args = parser.parse_args()

    scorer = Scorer("lp_baseline", args.split)
    for utt in tqdm(load_split(args.data_root, args.split)):
        formants, _ = lp_formants(utt["waveform"], args.order)
        scorer.add(formants, utt)
    scorer.finish()
