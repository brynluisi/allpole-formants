"""Run any combination of the included formant trackers on a single 16 kHz wav file and plot them.

    python demo.py speech.wav                                   # LP baseline and LP-DDSP (the default)
    python demo.py speech.wav --models all
    python demo.py speech.wav --models lp_ddsp --steps 300
    python demo.py speech.wav --models lp_baseline praat smelp --smelp_checkpoint checkpoints/smelp.pt

lp_lstm and smelp are trained models: they run only from a checkpoint saved by lp_lstm.py / smelp.py.
Panels show each method's own all-pole envelope where it has one, and the signal's spectrogram
otherwise, with the estimated formant tracks on top.
"""
import argparse
import os

import matplotlib.pyplot as plt
import soundfile as sf
import torch

from allpole.data import log_spectrogram
from allpole.dsp import Emphasis, LinearPredictor, envelope_db, get_formants, lar_to_poly
from allpole.models import FormantDecoder, LarEncoder
from lp_baseline import lp_formants
from lp_ddsp import lp_ddsp
from lp_lstm import lp_features

HOP_LENGTH = 160
MODELS = ["lp_baseline", "praat", "lp_ddsp", "lp_lstm", "smelp"]  # panel order, as in the README table
LABELS = {"lp_baseline": "LP baseline", "praat": "Praat", "lp_ddsp": "LP-DDSP",
          "lp_lstm": "LP-LSTM", "smelp": "SMELP"}
TRAINED = {"lp_lstm": "checkpoints/lp_lstm.pt", "smelp": "checkpoints/smelp.pt"}  # default checkpoints


def load_weights(module, state, model, path, order):
    """load_state_dict with a message that points at the usual cause: an --order mismatch."""
    try:
        module.load_state_dict(state)
    except RuntimeError as err:
        raise SystemExit(f"{path}: these weights do not fit {model} at --order {order}.\n{err}")


def run_lp_baseline(x, args, device):
    formants, a = lp_formants(x, args.order)
    return envelope_db(a.float()), formants


def run_lp_ddsp(x, args, device):
    torch.manual_seed(0)
    a, _ = lp_ddsp(x.to(device), args.order, args.steps)
    return envelope_db(a), get_formants(a)


def run_praat(x, args, device):
    from praat_baseline import praat_formants  # optional dependency, only needed for this method
    return None, praat_formants(x)


def run_lp_lstm(x, args, device):
    """LP coefficients -> BiLSTM -> formants. No envelope of its own: it reads the LP one."""
    model = FormantDecoder(input_size=args.order + 1).to(device)
    load_weights(model, torch.load(args.lp_lstm_checkpoint, map_location=device),
                 "lp_lstm", args.lp_lstm_checkpoint, args.order)
    model.eval()
    lpc = LinearPredictor(order=args.order).to(device)
    emphasis = Emphasis(0.97).to(device)
    with torch.no_grad():
        feats = lp_features(x.to(device), lpc, emphasis)
        return None, torch.exp(model(feats).squeeze(1))


def run_smelp(x, args, device):
    """CNN -> LARs -> (envelope, BiLSTM -> formants)."""
    ckpt = torch.load(args.smelp_checkpoint, map_location=device)
    cnn, lstm = LarEncoder(lpc_order=args.order).to(device), FormantDecoder(input_size=args.order).to(device)
    load_weights(cnn, ckpt["cnn"], "smelp (CNN)", args.smelp_checkpoint, args.order)
    load_weights(lstm, ckpt["lstm"], "smelp (BiLSTM)", args.smelp_checkpoint, args.order)
    cnn.eval(), lstm.eval()  # the encoder is batch-normed: eval() is required, not just tidy
    spec = log_spectrogram(x.flatten()).unsqueeze(0).to(device)
    with torch.no_grad():
        lars = cnn(spec)
        return envelope_db(lar_to_poly(lars)), torch.exp(lstm(lars.permute(2, 0, 1)).squeeze(1))


RUNNERS = {"lp_baseline": run_lp_baseline, "praat": run_praat, "lp_ddsp": run_lp_ddsp,
           "lp_lstm": run_lp_lstm, "smelp": run_smelp}


def select(names):
    """CLI names -> the models to run, deduplicated and in panel order."""
    return MODELS if "all" in names else [m for m in MODELS if m in set(names)]


def check_available(models, args):
    """Fail before any model runs: LP-DDSP alone takes minutes. Report every missing checkpoint at once."""
    missing = [(m, getattr(args, f"{m}_checkpoint")) for m in models
               if m in TRAINED and not os.path.isfile(getattr(args, f"{m}_checkpoint"))]
    if missing:
        raise SystemExit("\n".join(
            [f"{m} is a trained model and needs a checkpoint, but {path} was not found. "
             f"Train it with `python {m}.py`, or pass --{m}_checkpoint /path/to/weights.pt"
             for m, path in missing]))
    if "praat" in models:
        try:
            import parselmouth  # noqa: F401
        except ImportError:
            raise SystemExit("praat needs parselmouth: pip install praat-parselmouth")


def plot(panels, spec, fs, out):
    n_cols = min(len(panels), 3)
    n_rows = -(len(panels) // -n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True, squeeze=False)
    for ax, (model, env, formants) in zip(axes.flatten(), panels):
        background, source = (env, "all-pole envelope") if env is not None else (spec, "signal spectrogram")
        background = background - background.max()
        t = torch.arange(background.shape[1]) * HOP_LENGTH / fs
        ax.imshow(background, origin="lower", aspect="auto", vmin=-60, vmax=0, cmap="viridis",
                  extent=[0, t[-1].item(), 0, fs / 2])
        formants = formants.detach().cpu().clone()
        formants[formants == 0] = float("nan")  # get_formants pads missing roots with 0
        ax.plot(torch.arange(len(formants)) * HOP_LENGTH / fs, formants, "w.", markersize=1.5)
        ax.set(title=f"{LABELS[model]}\n{source}", xlabel="Time (s)", ylim=(0, 5500))
    for ax in axes.flatten()[len(panels):]:
        ax.axis("off")
    for row in axes:
        row[0].set_ylabel("Frequency (Hz)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wav")
    parser.add_argument("--models", nargs="+", choices=MODELS + ["all"], default=["lp_baseline", "lp_ddsp"],
                        metavar="MODEL", help=f"any of: {', '.join(MODELS)}, all")
    parser.add_argument("--order", type=int, default=16, help="LP order, shared by every method (Praat aside)")
    parser.add_argument("--steps", type=int, default=1500, help="LP-DDSP optimisation steps")
    parser.add_argument("--lp_lstm_checkpoint", default=TRAINED["lp_lstm"])
    parser.add_argument("--smelp_checkpoint", default=TRAINED["smelp"])
    parser.add_argument("--out", default="demo.png")
    args = parser.parse_args()

    models = select(args.models)
    check_available(models, args)

    x, fs = sf.read(args.wav, always_2d=True)
    assert fs == 16000, "expects 16 kHz audio"
    x = torch.from_numpy(x[:, 0]).float().reshape(1, 1, -1)  # kept on the CPU; runners move it as needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    panels = []
    for model in models:
        print(f"Running {LABELS[model]} ...", flush=True)
        panels.append((model,) + RUNNERS[model](x, args, device))

    # only the methods without an envelope of their own are drawn over the signal's spectrogram
    spec = log_spectrogram(x.flatten()) if any(env is None for _, env, _ in panels) else None
    plot(panels, spec, fs, args.out)
