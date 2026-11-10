#!/usr/bin/env python3
"""Teacher-forced comparison of logits dumps: the gate for a change that moves outputs.

  compare_logits.py IDS REF RUN [RUN ...]
  compare_logits.py IDS --oracle SCORE.npz RUN [RUN ...]

IDS is the comma-separated token file given to `ds41_load_probe --decode`; REF and each
RUN are directories holding the `logits.f32` that `--out DIR` wrote (one vocabulary row
per position). Prints the perplexity of each, and against REF the mean next-token NLL
difference with its standard error over 100-token blocks, the KL divergence from REF
(mean, 99th percentile, maximum) and the share of positions with the same argmax.
With --oracle the reference is the `score.npz` of `reference.py --score` (argmax and
NLL per position, no logits): each run's NLL difference, standard error and argmax
agreement against the CPU oracle.
"""
import sys

import numpy as np

VOCAB = 129280
CHUNK = 250


def load(directory):
    return np.memmap(f"{directory}/logits.f32", np.float32, "r").reshape(-1, VOCAB)


def log_softmax(rows):
    x = rows.astype(np.float64)
    top = x.max(1, keepdims=True)
    return x - (top + np.log(np.exp(x - top).sum(1, keepdims=True)))


def against_oracle(ids, score, runs):
    oracle = np.load(score)
    n = min(len(ids) - 1, len(oracle["nll"]))
    ref_nll, ref_argmax = oracle["nll"][:n].astype(np.float64), oracle["argmax"][:n]
    print(f"{'run':<24}{'ppl':>8}{'top-1':>8}{'dNLL':>9}{'se':>8}{'|dNLL|':>9}{'argmax':>8}")
    print(f"{'oracle':<24}{np.exp(ref_nll.mean()):8.3f}{(ref_argmax == ids[1:n + 1]).mean():8.4f}")
    for directory in runs:
        run = load(directory)
        nll, argmax = [], []
        for s in range(0, n, CHUNK):
            e = min(n, s + CHUNK)
            q = log_softmax(run[s:e])
            nll.append(-q[np.arange(e - s), ids[s + 1:e + 1]])
            argmax.append(q.argmax(1))
        nll, argmax = np.concatenate(nll), np.concatenate(argmax)
        diff = nll - ref_nll
        blocks = diff[: len(diff) // 100 * 100].reshape(-1, 100).mean(1)
        print(f"{directory.rstrip('/').split('/')[-1]:<24}{np.exp(nll.mean()):8.3f}{(argmax == ids[1:n + 1]).mean():8.4f}"
              f"{diff.mean():+9.4f}{blocks.std(ddof=1) / np.sqrt(len(blocks)):8.4f}{np.abs(diff).mean():9.4f}"
              f"{(argmax == ref_argmax).mean():8.4f}")


def main():
    ids = np.array([int(x) for x in open(sys.argv[1]).read().split(",")])
    if sys.argv[2] == "--oracle":
        return against_oracle(ids, sys.argv[3], sys.argv[4:])
    ref_dir, runs = sys.argv[2], sys.argv[3:]
    ref = load(ref_dir)
    n = min(len(ids) - 1, len(ref))

    def nll(logits):
        parts = []
        for s in range(0, n, CHUNK):
            e = min(n, s + CHUNK)
            parts.append(-log_softmax(logits[s:e])[np.arange(e - s), ids[s + 1:e + 1]])
        return np.concatenate(parts)

    ref_nll = nll(ref)
    print(f"{'run':<24}{'ppl':>8}{'dNLL':>9}{'se':>8}{'KL mean':>11}{'KL p99':>10}{'KL max':>10}{'argmax':>8}{'bits':>10}")
    print(f"{ref_dir.rstrip('/').split('/')[-1]:<24}{np.exp(ref_nll.mean()):8.3f}")
    for directory in runs:
        run = load(directory)
        kl, agree, same = [], 0, True
        for s in range(0, n, CHUNK):
            e = min(n, s + CHUNK)
            p, q = log_softmax(ref[s:e]), log_softmax(run[s:e])
            kl.append((np.exp(p) * (p - q)).sum(1))
            agree += (p.argmax(1) == q.argmax(1)).sum()
            same = same and np.array_equal(ref[s:e], run[s:e])
        kl = np.concatenate(kl)
        diff = nll(run) - ref_nll
        blocks = diff[: len(diff) // 100 * 100].reshape(-1, 100).mean(1)
        se = blocks.std(ddof=1) / np.sqrt(len(blocks))
        print(f"{directory.rstrip('/').split('/')[-1]:<24}{np.exp((ref_nll + diff).mean()):8.3f}{diff.mean():+9.4f}{se:8.4f}"
              f"{kl.mean():11.2e}{np.percentile(kl, 99):10.2e}{kl.max():10.2e}{agree / n:8.4f}"
              f"{'identical' if same else 'differ':>10}")


if __name__ == "__main__":
    main()
