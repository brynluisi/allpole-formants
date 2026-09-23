"""Summarise per-utterance results: mean RMSE with 95% t-confidence intervals, and an optional
paired Wilcoxon signed-rank test (Holm-corrected) of each method against a reference.

    python stats.py results/lp_baseline/test.csv results/lp_ddsp_l1+l2+reg/test.csv
    python stats.py results/*/train.csv --reference results/lp_ddsp_l1+l2+reg/train.csv
"""
import argparse
import csv

import numpy as np
from scipy import stats

FORMANTS = ["F1", "F2", "F3", "F4"]


def load(path):
    return {r["key"]: [float(r[f]) for f in FORMANTS] for r in csv.DictReader(open(path))}


def mean_ci(x):
    x = x[~np.isnan(x)]
    half = stats.t.ppf(0.975, len(x) - 1) * np.std(x, ddof=1) / np.sqrt(len(x))
    return np.mean(x), half


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvs", nargs="+")
    parser.add_argument("--reference", help="CSV to test the others against")
    args = parser.parse_args()

    paths = list(dict.fromkeys(args.csvs + ([args.reference] if args.reference else [])))
    results = {p: load(p) for p in paths}
    keys = sorted(set.intersection(*(set(r) for r in results.values())))
    arrays = {p: np.array([results[p][k] for k in keys]) for p in paths}  # (utterances, 4)
    print(f"{len(keys)} utterances common to all files; cells are mean +/- 95% CI half-width (Hz)\n")
    print(f"{'':40s}" + "".join(f"{c:>14s}" for c in FORMANTS + ["Mean"]))
    for p in paths:
        cols = [arrays[p][:, i] for i in range(4)] + [np.nanmean(arrays[p], axis=1)]
        print(f"{p:40s}" + "".join("{:>8.0f} +/-{:<3.0f}".format(*mean_ci(c)) for c in cols))

    if args.reference:
        others = [p for p in paths if p != args.reference]
        ref = np.nanmean(arrays[args.reference], axis=1)
        pvals = np.array([stats.wilcoxon(np.nanmean(arrays[p], axis=1), ref, nan_policy="omit").pvalue for p in others])
        order = np.argsort(pvals)
        holm = np.minimum(1, np.maximum.accumulate((len(pvals) - np.arange(len(pvals))) * pvals[order]))[np.argsort(order)]
        print(f"\nPaired Wilcoxon on per-utterance mean RMSE vs {args.reference} (Holm-corrected p):")
        for p, q in zip(others, holm):
            print(f"  {p:40s} p = {q:.2e}")
