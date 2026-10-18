#!/usr/bin/env python3
"""Build train/dev/test splits disjoint by sentence, not only by speaker."""
import argparse, json, sys
import pandas as pd


def components(df):
    """Connected components of the clip graph linked by shared speaker or sentence."""
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for _, r in df.iterrows():
        union(("spk", r.client_id), ("sent", r.sentence))
    lab = df.apply(lambda r: find(("spk", r.client_id)), axis=1)
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Common Voice TSV (needs client_id, sentence)")
    ap.add_argument("--output-dir", default="data/splits_grouped")
    ap.add_argument("--dev-ratio", type=float, default=0.10)
    ap.add_argument("--test-ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    df = pd.read_csv(a.input, sep="\t")
    for c in ("client_id", "sentence"):
        if c not in df.columns:
            sys.exit(f"missing required column: {c}  (have: {list(df.columns)})")
    df["sentence"] = df["sentence"].astype(str).str.strip()

    print(f"{len(df)} clips  {df.client_id.nunique()} speakers  "
          f"{df.sentence.nunique()} unique sentences")
    print(f"mean recordings per sentence: {len(df)/df.sentence.nunique():.1f}")

    lab = components(df)
    sizes = lab.value_counts()
    print(f"\nspeaker+sentence connected components: {len(sizes)}")
    print(f"largest holds {sizes.iloc[0]} clips ({100*sizes.iloc[0]/len(df):.1f}% of corpus)")
    if sizes.iloc[0] > 0.5 * len(df):
        print("=> One component dominates: splits CANNOT be disjoint on both axes.\n"
              "   Splitting on SENTENCE, since generalising to unseen text is what\n"
              "   the held-out test set will actually require. Speaker overlap is\n"
              "   reported below as the accepted cost.")

    # Sentence-wise assignment, greedy on clip count to hit the target ratios.
    rng = pd.Series(sorted(df.sentence.unique())).sample(frac=1.0, random_state=a.seed)
    counts = df.sentence.value_counts()
    n = len(df)
    targets = {"test": a.test_ratio * n, "dev": a.dev_ratio * n}
    got = {"test": 0, "dev": 0}
    assign = {}
    for s in rng:
        for split in ("test", "dev"):
            if got[split] < targets[split]:
                assign[s] = split
                got[split] += counts[s]
                break
        else:
            assign[s] = "train"
    df["split"] = df.sentence.map(assign)

    print("\n--- resulting splits ---")
    for split in ("train", "dev", "test"):
        d = df[df.split == split]
        print(f"{split:6s} {len(d):6d} clips  {d.sentence.nunique():5d} sentences  "
              f"{d.client_id.nunique():4d} speakers")

    tr = set(df[df.split == "train"].sentence)
    for split in ("dev", "test"):
        d = df[df.split == split]
        leak = d.sentence.isin(tr).sum()
        print(f"sentence leakage {split}: {leak}/{len(d)} ({100*leak/max(len(d),1):.2f}%)")
    trs = set(df[df.split == "train"].client_id)
    for split in ("dev", "test"):
        d = df[df.split == split]
        ov = d.client_id.isin(trs).sum()
        print(f"speaker overlap  {split}: {ov}/{len(d)} ({100*ov/max(len(d),1):.1f}%)  [accepted]")

    import os
    os.makedirs(a.output_dir, exist_ok=True)
    for split in ("train", "dev", "test"):
        p = f"{a.output_dir}/{split}.tsv"
        df[df.split == split].drop(columns=["split"]).to_csv(p, sep="\t", index=False)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
