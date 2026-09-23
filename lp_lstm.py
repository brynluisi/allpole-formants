"""LP-LSTM baseline: a BiLSTM maps frame-wise LP coefficients to log formant frequencies.

    python lp_lstm.py --data_root /path/to/data                       # train, save, evaluate on test
    python lp_lstm.py --data_root /path/to/data --checkpoint ckpt.pt  # evaluate only
"""
import argparse
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from allpole.data import load_split
from allpole.dsp import Emphasis, LinearPredictor
from allpole.evaluate import Scorer
from allpole.models import FormantDecoder


def lp_features(x, lpc, emphasis):
    """(1, 1, T) waveform -> LP coefficients as an LSTM sequence (frames, 1, order+1)."""
    a, _ = lpc.estimate(emphasis(x)[:, 0, :])
    return a.permute(2, 0, 1)


def train(model, utts, lpc, emphasis, epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for utt in utts:
            optimizer.zero_grad()
            feats = lp_features(utt["waveform"].to(device), lpc, emphasis)
            T = min(len(feats), len(utt["formants"]))
            mask = utt["mask"][:T].to(device)
            log_pred = model(feats[:T]).squeeze(1)
            loss = F.mse_loss(log_pred[mask], torch.log(utt["formants"][:T].to(device))[mask])
            loss.backward()
            optimizer.step()
            total += loss.item()
        scheduler.step()
        print(f"[epoch {epoch + 1}/{epochs}] loss {total / len(utts):.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", default=os.environ.get("DATA_ROOT"), required="DATA_ROOT" not in os.environ)
    parser.add_argument("--split", default="test", choices=["train", "test"], help="evaluation split")
    parser.add_argument("--order", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--checkpoint", help="load weights from this file and skip training")
    parser.add_argument("--save", default="checkpoints/lp_lstm.pt")
    args = parser.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FormantDecoder(input_size=args.order + 1).to(device)
    lpc = LinearPredictor(order=args.order).to(device)
    emphasis = Emphasis(0.97).to(device)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        train(model, load_split(args.data_root, "train"), lpc, emphasis, args.epochs, device)
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        torch.save(model.state_dict(), args.save)
        print(f"Saved {args.save}")

    model.eval()
    scorer = Scorer("lp_lstm", args.split)
    with torch.no_grad():
        for utt in tqdm(load_split(args.data_root, args.split)):
            feats = lp_features(utt["waveform"].to(device), lpc, emphasis)
            T = min(len(feats), len(utt["formants"]))  # the BiLSTM sees exactly the scored time span
            scorer.add(torch.exp(model(feats[:T]).squeeze(1)), utt)
    scorer.finish()
