"""SMELP: a CNN predicts log-area ratios from the STFT, the LP-DDSP loss keeps them a valid
all-pole model of the signal, and a BiLSTM decodes the LARs into formant tracks.

Loss = L_DDSP (L2 + 0.5 L1 + 0.1 reg) + 0.1 * MSE(a, a_LP)  [L_coeff] + MSE(log F, log F_true)  [L_F]

    python smelp.py --data_root /path/to/data                       # train, save, evaluate on test
    python smelp.py --data_root /path/to/data --checkpoint ckpt.pt  # evaluate only
"""
import argparse
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from allpole.data import load_split
from allpole.dsp import Emphasis, LinearPredictor, lar_to_poly
from allpole.evaluate import Scorer
from allpole.models import FormantDecoder, LarEncoder
from lp_ddsp import ddsp_loss


def smelp_loss(utt, cnn, lstm, lpc, emphasis, device):
    x_emph = emphasis(utt["waveform"].to(device))
    lars = cnn(utt["spectrogram"].to(device))                        # (1, P, frames)
    a = lar_to_poly(lars)
    a_lp, _ = lpc.estimate(x_emph[:, 0, :])
    loss_ddsp = ddsp_loss(lpc.inverse_filter(x_emph, a), lars) + 0.1 * F.mse_loss(a_lp, a)

    T = min(lars.shape[-1], len(utt["formants"]))
    log_pred = lstm(lars[:, :, :T].permute(2, 0, 1)).squeeze(1)     # (T, 4)
    loss_formants = F.mse_loss(log_pred, torch.log(utt["formants"][:T].to(device)))
    return loss_ddsp + loss_formants


def predict(utt, cnn, lstm, device):
    lars = cnn(utt["spectrogram"].to(device))
    T = min(lars.shape[-1], len(utt["formants"]))  # the BiLSTM sees exactly the scored time span
    return torch.exp(lstm(lars[:, :, :T].permute(2, 0, 1)).squeeze(1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", default=os.environ.get("DATA_ROOT"), required="DATA_ROOT" not in os.environ)
    parser.add_argument("--split", default="test", choices=["train", "test"], help="evaluation split")
    parser.add_argument("--order", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--checkpoint", help="load weights from this file and skip training")
    parser.add_argument("--save", default="checkpoints/smelp.pt")
    args = parser.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn = LarEncoder(lpc_order=args.order).to(device)
    lstm = FormantDecoder(input_size=args.order).to(device)
    lpc = LinearPredictor(order=args.order).to(device)
    emphasis = Emphasis(0.97).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        cnn.load_state_dict(ckpt["cnn"])
        lstm.load_state_dict(ckpt["lstm"])
    else:
        utts = load_split(args.data_root, "train", spectrogram=True)
        optimizer = torch.optim.Adam(list(cnn.parameters()) + list(lstm.parameters()), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        cnn.train(), lstm.train()
        for epoch in range(args.epochs):
            total = 0.0
            for utt in tqdm(utts, leave=False):
                optimizer.zero_grad()
                loss = smelp_loss(utt, cnn, lstm, lpc, emphasis, device)
                loss.backward()
                optimizer.step()
                total += loss.item()
            scheduler.step()
            print(f"[epoch {epoch + 1}/{args.epochs}] loss {total / len(utts):.4f}", flush=True)
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        torch.save({"cnn": cnn.state_dict(), "lstm": lstm.state_dict()}, args.save)
        print(f"Saved {args.save}")

    cnn.eval(), lstm.eval()
    scorer = Scorer("smelp", args.split)
    with torch.no_grad():
        for utt in tqdm(load_split(args.data_root, args.split, spectrogram=True)):
            scorer.add(predict(utt, cnn, lstm, device), utt)
    scorer.finish()
