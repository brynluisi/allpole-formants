"""Run the LP baseline and LP-DDSP on a single 16 kHz wav file and plot the spectral envelopes.

    python demo.py speech.wav [--steps 1500]
"""
import argparse

import matplotlib.pyplot as plt
import soundfile as sf
import torch

from allpole.dsp import envelope_db, get_formants
from lp_baseline import lp_formants
from lp_ddsp import lp_ddsp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wav")
    parser.add_argument("--order", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--out", default="demo.png")
    args = parser.parse_args()

    x, fs = sf.read(args.wav, always_2d=True)
    assert fs == 16000, "expects 16 kHz audio"
    x = torch.from_numpy(x[:, 0]).float().reshape(1, 1, -1)

    torch.manual_seed(0)
    _, a_lp = lp_formants(x, args.order)
    a_ddsp, _ = lp_ddsp(x, args.order, args.steps)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, a, title in zip(axes, [a_lp.float(), a_ddsp], ["LP baseline", "LP-DDSP"]):
        env = envelope_db(a)
        env = env - env.max()
        t = torch.arange(env.shape[1]) * 160 / fs
        ax.imshow(env, origin="lower", aspect="auto", vmin=-60, vmax=0, cmap="viridis",
                  extent=[0, t[-1].item(), 0, fs / 2])
        f = get_formants(a)
        f[f == 0] = float("nan")
        ax.plot(t, f, "w.", markersize=1.5)
        ax.set(title=title, xlabel="Time (s)", ylim=(0, 5500))
    axes[0].set_ylabel("Frequency (Hz)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")
