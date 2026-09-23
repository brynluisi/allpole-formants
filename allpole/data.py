"""TIMIT audio paired with VTR Formants ground truth.

Expected layout (directory names are matched case-insensitively, so both the
LDC upper-case TIMIT release and lower-case copies work):

    <data_root>/TIMIT/{train,test}/<dr>/<speaker>/<utt>.wav
    <data_root>/VTRFormants/{train,test}/<dr>/<speaker>/<utt>.{fb,phn}
"""
import struct
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Phone classes scored in the evaluation (frames where formants exist).
FORMANT_PHONES = {
    # vowels
    "aa", "ae", "ah", "ao", "aw", "ax", "ax-h", "axr", "ay", "eh", "er", "ey",
    "ih", "ix", "iy", "ow", "oy", "uh", "uw", "ux",
    # nasals, liquids, glides
    "m", "n", "ng", "em", "en", "eng", "nx", "l", "r", "el", "rx", "lx", "w", "y",
    # voiced fricatives and stops
    "v", "dh", "z", "zh", "jh", "b", "d", "g",
}


def read_fb(path):
    """VTR .fb (HTK format): returns (frames, 8) = F1-F4 and B1-B4 in kHz, 10 ms hop."""
    with open(path, "rb") as f:
        n_samples, _ = struct.unpack(">ii", f.read(8))
        samp_size, _ = struct.unpack(">hh", f.read(4))
        n = n_samples * (samp_size // 4)
        data = struct.unpack(f">{n}f", f.read(4 * n))
    return np.array(data, dtype=np.float32).reshape(n_samples, samp_size // 4)


def formant_mask(phn_path, num_frames, hop_length=160):
    """Boolean (num_frames,) mask, True on frames inside phones that have formants."""
    mask = torch.zeros(num_frames, dtype=torch.bool)
    for line in open(phn_path, encoding="ascii"):
        if len(line.split()) != 3:
            continue
        start, end, label = line.split()
        mask[max(0, int(start) // hop_length):min(num_frames, int(end) // hop_length)] = label in FORMANT_PHONES
    return mask


def log_spectrogram(x, n_fft=512, hop_length=160, win_length=512):
    """20 log10 |STFT| of a waveform (T,) -> (n_fft//2+1, frames)."""
    S = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                   window=torch.hann_window(win_length), return_complex=True).abs()
    return 20 * torch.log10(S + 1e-8)


def _subdir(root, name):
    for p in Path(root).iterdir():
        if p.is_dir() and p.name.lower() == name.lower():
            return p
    raise FileNotFoundError(f"No '{name}' directory in {root}")


def _index(root, suffix):
    return {(p.parent.name.lower(), p.stem.lower()): p
            for p in root.rglob("*") if p.suffix.lower() == suffix}


def load_split(data_root, split, hop_length=160, spectrogram=False):
    """Load all TIMIT utterances that have VTR ground truth, sorted by speaker/utterance.

    Each item is a dict:
        key          "<speaker>_<utt>"
        waveform     (1, 1, T) float32
        formants     (frames, 4) ground-truth F1-F4 in Hz
        mask         (frames,) bool, frames that are scored
        spectrogram  (1, 257, stft_frames) log-magnitude (only if spectrogram=True)
    """
    wavs = _index(_subdir(_subdir(data_root, "TIMIT"), split), ".wav")
    fbs = _index(_subdir(_subdir(data_root, "VTRFormants"), split), ".fb")
    keys = sorted(k for k in wavs.keys() & fbs.keys() if fbs[k].with_suffix(".phn").exists())

    utts = []
    for speaker, utt in keys:
        x, fs = sf.read(wavs[speaker, utt])
        assert fs == 16000, f"expected 16 kHz audio, got {fs}"
        x = torch.from_numpy(x).float()
        formants = torch.from_numpy(read_fb(fbs[speaker, utt]))[:, :4] * 1000
        item = dict(
            key=f"{speaker}_{utt}",
            waveform=x.reshape(1, 1, -1),
            formants=formants,
            mask=formant_mask(fbs[speaker, utt].with_suffix(".phn"), len(formants), hop_length),
        )
        if spectrogram:
            item["spectrogram"] = log_spectrogram(x).unsqueeze(0)
        utts.append(item)
    print(f"Loaded {len(utts)} {split} utterances")
    return utts
