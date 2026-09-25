# Smooth Formant Tracking with Differentiable Linear Prediction

Code for the Interspeech 2026 paper by Bryn Luisi and Lauri Juvela (Aalto University).

**LP-DDSP** treats the reflection coefficients of an all-pole model as free parameters in log-area-ratio (LAR) form.
It optimises them by backpropagating through a differentiable LP inverse filter.
The loss combines the L2 and L1 norms of the prediction error with a temporal smoothness penalty on the LARs.
Formants are read from the roots of the resulting polynomial.
**SMELP** puts the same differentiable LP block inside a neural tracker: a CNN predicts LARs from the STFT, and a BiLSTM decodes them into formant tracks.

```
allpole/dsp.py       differentiable LP: STFT-domain filtering, Levinson recursions, LAR -> polynomial, formant picking
allpole/data.py      TIMIT + VTR Formants loader and the scoring mask
allpole/models.py    CNN LAR encoder and BiLSTM formant decoder
allpole/evaluate.py  per-utterance RMSE
allpole/hub.py       downloads lp_lstm/smelp checkpoints from Hugging Face
lp_baseline.py       LP baseline
lp_ddsp.py           LP-DDSP (and its loss ablations)
lp_lstm.py           LP-LSTM baseline
smelp.py             SMELP
praat_baseline.py    Praat baseline (via parselmouth)
stats.py             mean RMSE with 95% confidence intervals, paired Wilcoxon tests
run.py               run any combination of the methods on a single 16 kHz wav file
```

## Setup

You need Python 3.10–3.13 and git. From a terminal:

```bash
git clone https://github.com/brynluisi/allpole-formants.git
cd allpole-formants
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

This creates a self-contained environment in `.venv/` with the pinned package versions the results were produced with.
Run `source .venv/bin/activate` again in each new terminal before using the scripts.

On Linux, the default `torch` wheel from PyPI includes CUDA support.
For a different CUDA version, install PyTorch first by following [pytorch.org](https://pytorch.org/get-started/locally/), then run `pip install -r requirements.txt`.

## Quick start (no dataset needed)

Run any combination of the five methods on a single 16 kHz mono wav file:

```bash
python run.py path/to/speech.wav                                    # LP baseline + LP-DDSP (default)
python run.py path/to/speech.wav --models lp_ddsp                   # a single method
python run.py path/to/speech.wav --models lp_lstm                   # pretrained LP-LSTM
python run.py path/to/speech.wav --models smelp                     # pretrained SMELP
python run.py path/to/speech.wav --models lp_baseline praat smelp   # any combination
python run.py path/to/speech.wav --models all                       # all five
```

Choices: `lp_baseline`, `praat`, `lp_ddsp`, `lp_lstm`, `smelp`, `all`.
This writes `run.png`: each method's spectral envelope (or, for Praat and LP-LSTM, the signal's
spectrogram) with the estimated formant tracks on top. Add `--steps 300` to speed up LP-DDSP's
per-utterance optimisation. `praat` needs `parselmouth`, already in `requirements.txt`.

### Pretrained models (Hugging Face)

`lp_lstm` and `smelp` are trained models, but you don't need to train them yourself: the first time
you ask for either, the paper's checkpoints are downloaded from
[huggingface.co/Aalto-Speech-Synthesis/smelp](https://huggingface.co/Aalto-Speech-Synthesis/smelp) to
`checkpoints/lp_lstm.pt` / `checkpoints/smelp.pt` and reused after that — no account or token needed.
`lp_lstm.py --checkpoint checkpoints/lp_lstm.pt` / `smelp.py --checkpoint checkpoints/smelp.pt` fetch
the same way (see [Training the neural models](#training-the-neural-models)), reproducing the paper's
rows on your own `DATA_ROOT` without training anything.

To fetch the weights yourself instead — e.g. to work offline afterwards:

```bash
mkdir -p checkpoints
curl -L -o checkpoints/lp_lstm.pt https://huggingface.co/Aalto-Speech-Synthesis/smelp/resolve/main/lp_lstm.pt
curl -L -o checkpoints/smelp.pt   https://huggingface.co/Aalto-Speech-Synthesis/smelp/resolve/main/smelp.pt
```

`--lp_lstm_checkpoint` / `--smelp_checkpoint` point at a different file instead, e.g. one you trained
yourself — an explicitly-named checkpoint is never auto-downloaded, so a missing one just errors.

## Data

The experiments use [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1) (LDC, licence required) and the VTR Formants ground truth.
VTR Formants is from Deng et al., "A database of vocal tract resonance trajectories for research in speech processing", ICASSP 2006.
Put both under one directory (directory-name case does not matter):

```
<data_root>/TIMIT/{train,test}/<dr>/<speaker>/<utt>.wav
<data_root>/VTRFormants/{train,test}/<dr>/<speaker>/<utt>.fb   (+ <utt>.phn)
```

This gives 324 train and 192 test utterances.
Tell the scripts where the data is, either once per terminal:

```bash
export DATA_ROOT=/path/to/data_root
```

or with `--data_root /path/to/data_root` on each command.

## Reproducing the results

With the environment activated and `DATA_ROOT` set, run from the repository folder:

```bash
python lp_baseline.py        # LP baseline         (~1 min on CPU)
python praat_baseline.py     # Praat               (~1 min on CPU)
python lp_ddsp.py            # LP-DDSP             (~20 min on CPU, faster on GPU)
python lp_lstm.py            # LP-LSTM: trains, then evaluates
python smelp.py              # SMELP: trains, then evaluates (GPU recommended)
```

Each script evaluates on the test set by default; use `--split train` for the training set.
Each prints the RMSE per formant and writes per-utterance results to `results/<method>/<split>.csv`.
Scoring happens every 10 ms, only on frames inside vowels, nasals, liquids, glides, voiced fricatives and voiced stops.
The metric is the RMSE per utterance, averaged over utterances.

Test-set RMSE (Hz) reported in the paper:

| Method | Script | F1 | F2 | F3 | F4 | Mean |
|---|---|---|---|---|---|---|
| LP baseline | `lp_baseline.py` | 163 | 299 | 389 | 662 | 378 |
| LP-DDSP | `lp_ddsp.py` | 131 | 222 | 268 | 362 | 246 |
| Praat | `praat_baseline.py` | 257 | 350 | 413 | 356 | 344 |
| LP-LSTM | `lp_lstm.py` | 107 | 145 | 183 | 249 | 171 |
| SMELP | `smelp.py` | 100 | 141 | 195 | 230 | 166 |

TV-QCP and KARMA were run with their authors' MATLAB implementations and are not included here.

**Confidence intervals and significance:** compare any set of result files:

```bash
python stats.py results/lp_baseline/test.csv "results/lp_ddsp_l1+l2+reg/test.csv" \
    --reference "results/lp_ddsp_l1+l2+reg/test.csv"
```

This prints the mean ± 95% confidence interval per formant.
With `--reference`, it adds a paired Wilcoxon test of each file against the reference.

**LP-DDSP loss ablation (training set):**

```bash
for loss in l1 l1+reg l2 l2+reg l1+l2 l1+l2+reg; do
    python lp_ddsp.py --split train --loss $loss
done
python stats.py results/lp_ddsp_*/train.csv --reference "results/lp_ddsp_l1+l2+reg/train.csv"
```

### Training the neural models

`lp_lstm.py` and `smelp.py` train on the training split for 100 epochs (`--epochs`), save the
weights (`--save`, default `checkpoints/<model>.pt`), then evaluate on the test split:

```bash
python lp_lstm.py --data_root /path/to/data
python smelp.py   --data_root /path/to/data   # GPU recommended
```

To evaluate a checkpoint instead of training, pass `--checkpoint` (see
[Pretrained models](#pretrained-models-hugging-face) for using the paper's own weights this way):

```bash
python lp_lstm.py --data_root /path/to/data --checkpoint checkpoints/lp_lstm.pt
python smelp.py   --data_root /path/to/data --checkpoint checkpoints/smelp.pt
```

**Notes:**
- LP-DDSP runs 1500 Adam steps per utterance (learning rate 0.1, cosine decay), starting from N(0, 0.2) LARs.
  It is seeded per utterance.
  It uses the GPU when one is available, but results are only bit-reproducible on the same device and software stack.
- The LP baseline defaults to order 18, which is the setting that produced the row above. All other methods use order 16.
- This release gives LP-DDSP 131 / 224 / 271 / 366, mean 248 Hz, on the test set (CPU).
  The 246 Hz row above came from an earlier run whose initialisation differed slightly.
  The gap is well inside the 95% confidence interval (±12 Hz).

## Citation

```bibtex
@inproceedings{luisi26_interspeech,
  title     = {{Smooth Formant Tracking with Differentiable Linear Prediction}},
  author    = {Bryn Luisi and Lauri Juvela},
  year      = {2026},
  booktitle = {{Interspeech 2026}},
  pages     = {1828--1832},
  doi       = {10.21437/Interspeech.2026-1222},
  issn      = {2958-1796},
}
```

## License

MIT, see [LICENSE](LICENSE).
