"""LP-DDSP: per-utterance optimisation of log-area ratios through a differentiable LP inverse filter.

Loss = L2(e) + 0.5 * L1(e) + 0.1 * mean((LAR[m+1] - LAR[m])^2); use --loss to run the ablations.

    python lp_ddsp.py --data_root /path/to/data --split test
    python lp_ddsp.py --data_root /path/to/data --split train --loss l1+reg
"""
import argparse
import os

import torch
from tqdm import tqdm

from allpole.data import load_split
from allpole.dsp import Emphasis, LinearPredictor, get_formants, lar_to_poly
from allpole.evaluate import Scorer

LOSS_WEIGHTS = {"l2": 1.0, "l1": 0.5, "reg": 0.1}
ABLATIONS = ["l1", "l1+reg", "l2", "l2+reg", "l1+l2", "l1+l2+reg"]


def ddsp_loss(e, lars, terms=("l2", "l1", "reg")):
    """Weighted residual L2 / L1 and LAR temporal smoothness. e: residual, lars: (B, P, frames)."""
    parts = {"l2": torch.mean(e ** 2),
             "l1": torch.mean(torch.abs(e)),
             "reg": torch.mean(torch.abs(lars[:, :, 1:] - lars[:, :, :-1]) ** 2)}
    loss = 0
    for name in ("l2", "l1", "reg"):
        if name in terms:
            loss = loss + LOSS_WEIGHTS[name] * parts[name]
    return loss


def lp_ddsp(x, order=16, steps=1500, lr=0.1, terms=("l2", "l1", "reg")):
    """x: (1, 1, T) waveform -> optimised LP polynomial a (1, order+1, frames) and LARs."""
    lpc = LinearPredictor(order=order).to(x.device)
    x_emph = Emphasis(0.97).to(x.device)(-1.0 * x)
    n_frames = x.size(-1) // lpc.hop_length + 1
    # N(0, 0.2) init, drawn in frame-major memory order (as in the paper runs, for bit-exact reproduction)
    init = torch.empty(1, n_frames, order, device=x.device).transpose(1, 2).normal_()
    lars = torch.nn.Parameter(0.2 * init)
    optim = torch.optim.Adam([lars], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=steps, eta_min=1e-6)
    for _ in range(steps):
        optim.zero_grad()
        a = lar_to_poly(lars)
        loss = ddsp_loss(lpc.inverse_filter(x_emph, a), lars, terms)
        loss.backward()
        optim.step()
        scheduler.step()
    return a.detach(), lars.detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", default=os.environ.get("DATA_ROOT"), required="DATA_ROOT" not in os.environ)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--loss", default="l1+l2+reg", choices=ABLATIONS)
    parser.add_argument("--order", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer = Scorer(f"lp_ddsp_{args.loss}", args.split)
    for idx, utt in enumerate(tqdm(load_split(args.data_root, args.split))):
        torch.manual_seed(idx)  # LAR init depends only on the utterance's position
        a, _ = lp_ddsp(utt["waveform"].to(device), args.order, args.steps, args.lr, args.loss.split("+"))
        scorer.add(get_formants(a), utt)
    scorer.finish()
