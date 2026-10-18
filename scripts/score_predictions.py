"""Recompute the reported error rates and bootstrap intervals from predictions/."""
import argparse, csv, random
from pathlib import Path


def edits(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def load(path):
    rows = {}
    for r in csv.reader(open(path, encoding="utf-8"), delimiter="\t"):
        if len(r) >= 3:
            rows[r[0]] = (r[1], r[2])
    return rows


def tally(rows, keys, word):
    """Per-utterance (edits, reference length), computed once."""
    out = []
    for k in keys:
        ref, hyp = rows[k]
        a, b = (ref.split(), hyp.split()) if word else (ref, hyp)
        out.append((edits(a, b), len(a)))
    return out


def rate(t, idx):
    num = sum(t[i][0] for i in idx)
    den = sum(t[i][1] for i in idx)
    return 100 * num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="predictions", type=Path)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    x, w = load(a.dir / "xlsr_test.tsv"), load(a.dir / "w2vbert_test.tsv")
    keys = [k for k in w if k in x]
    assert all(x[k][0] == w[k][0] for k in keys), "references differ between systems"
    print(f"{len(keys)} paired test utterances, identical references\n")

    n = len(keys)
    for word, name in ((False, "CER"), (True, "WER")):
        tx, tw = tally(x, keys, word), tally(w, keys, word)
        whole = list(range(n))
        rx, rw = rate(tx, whole), rate(tw, whole)
        random.seed(a.seed)
        gains = sorted(
            rate(tx, s) - rate(tw, s)
            for s in ([random.randrange(n) for _ in range(n)] for _ in range(a.resamples))
        )
        lo, hi = gains[int(0.025 * a.resamples)], gains[int(0.975 * a.resamples)]
        wins = sum(1 for g in gains if g <= 0)
        print(f"{name}: XLS-R {rx:.2f}   w2v-BERT {rw:.2f}   gain {rx - rw:.2f}")
        print(f"     95% CI [{lo:.2f}, {hi:.2f}] over {a.resamples} paired resamples; "
              f"XLS-R wins {wins}\n")


if __name__ == "__main__":
    main()
