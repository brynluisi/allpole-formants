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
lp_baseline.py       LP baseline
lp_ddsp.py           LP-DDSP (and its loss ablations)
lp_lstm.py           LP-LSTM baseline
smelp.py             SMELP
praat_baseline.py    Praat baseline (via parselmouth)
stats.py             mean RMSE with 95% confidence intervals, paired Wilcoxon tests
demo.py              LP vs LP-DDSP on any 16 kHz wav file
```

## Setup

```bash
pip install -r requirements.txt
```

To try LP-DDSP on your own recording, without any dataset:

```bash
python demo.py speech.wav   # writes demo.png: LP and LP-DDSP envelopes with formant tracks
```

## Data

The experiments use [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1) (LDC, licence required) and the VTR Formants ground truth.
VTR Formants is from Deng et al., "A database of vocal tract resonance trajectories for research in speech processing", ICASSP 2006.
Arrange them as follows (directory-name case does not matter):

```
$DATA_ROOT/TIMIT/{train,test}/<dr>/<speaker>/<utt>.wav
$DATA_ROOT/VTRFormants/{train,test}/<dr>/<speaker>/<utt>.fb   (+ <utt>.phn)
```

This gives 324 train and 192 test utterances. Every script takes `--data_root`, or reads the `DATA_ROOT` environment variable.

## Reproducing the results

Scoring happens every 10 ms, only on frames inside vowels, nasals, liquids, glides, voiced fricatives and voiced stops.
The metric is the RMSE per utterance, averaged over utterances.
Each run prints the mean and writes per-utterance RMSEs to `results/<method>/<split>.csv`.

| Method | Command | F1 | F2 | F3 | F4 | Mean |
|---|---|---|---|---|---|---|
| LP baseline | `python lp_baseline.py` | 163 | 299 | 389 | 662 | 378 |
| LP-DDSP | `python lp_ddsp.py` | 131 | 222 | 268 | 362 | 246 |
| Praat | `python praat_baseline.py` | 257 | 350 | 413 | 356 | 344 |
| LP-LSTM | `python lp_lstm.py` | 107 | 145 | 183 | 249 | 171 |
| SMELP | `python smelp.py` | 100 | 141 | 195 | 230 | 166 |

The table shows test-set RMSE in Hz as reported in the paper.
TV-QCP and KARMA were run with their authors' MATLAB implementations and are not included here.

**LP-DDSP loss ablation (training set):**

```bash
for loss in l1 l1+reg l2 l2+reg l1+l2 l1+l2+reg; do
    python lp_ddsp.py --split train --loss $loss
done
python stats.py results/lp_ddsp_*/train.csv --reference "results/lp_ddsp_l1+l2+reg/train.csv"
```

**Training:** `lp_lstm.py` and `smelp.py` train on the training split for 100 epochs.
They save a checkpoint to `checkpoints/` and then evaluate.
Pass `--checkpoint <file>` to evaluate a saved model without training. SMELP realistically needs a GPU.

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
@inproceedings{luisi2026smooth,
  title     = {Smooth Formant Tracking with Differentiable Linear Prediction},
  author    = {Luisi, Bryn and Juvela, Lauri},
  booktitle = {Proc. Interspeech},
  year      = {2026}
}
```
