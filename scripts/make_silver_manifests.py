#!/usr/bin/env python3
"""Build the silver-labelled spontaneous train and dev manifests.

Clips longer than --max-sec are dropped rather than truncated, and clips whose
text needs symbols outside the vocabulary are dropped."""
import argparse
import csv
import json
from pathlib import Path

SCORE_STRIP = set("?!¿¡.,")


def normalise(text):
    """The official scorer's transformation, and nothing else."""
    return "".join(c for c in str(text).lower() if c not in SCORE_STRIP).strip()


def read_tsv(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="clone of QuechuaBase/asr-puno-quechua")
    ap.add_argument("--vocab", required=True, help="vocab.json of the warm-start checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-sec", type=float, default=30.0)
    a = ap.parse_args()

    repo = Path(a.repo)
    sps = repo / "data/raw/sps-corpus-4.0-2026-06-12-qxp/ss-corpus-qxp.tsv"
    dur = {r["audio_file"]: int(r["duration_ms"]) / 1000
           for r in read_tsv(sps) if r["duration_ms"].isdigit()}

    vocab = json.loads(Path(a.vocab).read_text(encoding="utf-8"))
    emittable = set(vocab) - {"[PAD]", "[UNK]", "|"}

    def prepare(rows):
        kept, long, oov, missing = [], 0, 0, 0
        for r in rows:
            name = Path(r["path"]).stem + ".mp3"
            seconds = dur.get(name)
            if seconds is None:
                missing += 1
                continue
            if seconds > a.max_sec:
                long += 1
                continue
            text = normalise(r["sentence"])
            if not text or set(text) - {" "} - emittable:
                oov += 1
                continue
            kept.append({"path": name, "sentence": text, "sec": round(seconds, 2)})
        return kept, long, oov, missing

    splits = {k: read_tsv(repo / f"data/splits/silver_spontaneous/{k}.tsv")
              for k in ("train", "dev", "test")}

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # Their train and dev become our training data; their test becomes our dev.
    train = []
    for k in ("train", "dev"):
        kept, long, oov, missing = prepare(splits[k])
        print(f"their {k:5s}: {len(splits[k]):5d} -> {len(kept):5d} kept  "
              f"(over {a.max_sec:.0f}s {long}, unemittable {oov}, no audio {missing})")
        train += kept
    dev, long, oov, missing = prepare(splits["test"])
    print(f"their test : {len(splits['test']):5d} -> {len(dev):5d} kept  "
          f"(over {a.max_sec:.0f}s {long}, unemittable {oov}, no audio {missing})")

    def write(name, rows):
        with open(out / name, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("path\tsentence\tsec\n")
            for r in rows:
                fh.write(f"{r['path']}\t{r['sentence']}\t{r['sec']}\n")
        print(f"wrote {out/name}: {len(rows)} clips, {sum(r['sec'] for r in rows)/3600:.2f} h")

    write("silver2_train.tsv", train)
    write("silver2_dev.tsv", dev)

    overlap = {r["path"] for r in train} & {r["path"] for r in dev}
    assert not overlap, f"{len(overlap)} clips in both train and dev"
    print("no train/dev overlap")


if __name__ == "__main__":
    main()
