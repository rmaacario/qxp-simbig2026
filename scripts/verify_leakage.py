"""Recompute the Section 3.1 leakage figures from the organisers' released files."""
import argparse, csv, re, sys, unicodedata
from pathlib import Path

STRIP = set("?!¿¡.,")

NORMALISERS = {
    "exact":                 lambda s: s,
    "nfc":                   lambda s: unicodedata.normalize("NFC", s),
    "lower":                 lambda s: s.lower(),
    "scorer":                lambda s: "".join(c for c in s.lower() if c not in STRIP).strip(),
    "scorer+space":          lambda s: " ".join("".join(c for c in s.lower() if c not in STRIP).split()),
    "nfc+scorer+space":      lambda s: " ".join("".join(
                                 c for c in unicodedata.normalize("NFC", s).lower()
                                 if c not in STRIP).split()),
}

# What the paper states. The script fails if any of these stops being true.
EXPECTED = {
    "validated_scripted":   (2299, 2298),
    "splits_cv26/scripted": (909, 909),
    "ours":                 (2275, 0),
}


def read(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))


def sentences(rows):
    key = "sentence" if "sentence" in rows[0] else list(rows[0])[-1]
    return [str(r[key]) for r in rows]


def check_split(label, d, fails):
    train, dev, test = (sentences(read(d / f"{s}.tsv")) for s in ("train", "dev", "test"))
    n_clips, n_leak = EXPECTED[label]
    if len(test) != n_clips:
        fails.append(f"{label}: expected {n_clips} test clips, found {len(test)}")
    for name, f in NORMALISERS.items():
        tr = {f(s) for s in train}
        hit = sum(1 for s in test if f(s) in tr)
        dev_hit = sum(1 for s in dev if f(s) in tr)
        flag = "" if hit == n_leak else "   <-- MISMATCH"
        print(f"  {label:24s} {name:18s} test {hit:5d}/{len(test)} "
              f"({100*hit/len(test):6.2f}%)  dev {dev_hit:5d}/{len(dev)}{flag}")
        if hit != n_leak:
            fails.append(f"{label}/{name}: expected {n_leak} leaking test clips, found {hit}")


def check_make_split(repo, fails):
    src = (repo / "data" / "make_split.py").read_text(encoding="utf-8")
    m = re.search(r"required_cols\s*=\s*\{([^}]*)\}", src)
    cols = {c.strip().strip("\"'") for c in m.group(1).split(",")} if m else set()
    print(f"\n  make_split.py required_cols = {sorted(cols)}")
    if "sentence" in cols:
        fails.append("make_split.py now requires the sentence field")
    if cols != {"client_id", "gender", "duration_ms"}:
        fails.append(f"make_split.py required_cols changed: {sorted(cols)}")


def check_baseline(repo, fails):
    """The 1.19 WER row, and whether its own references appear in training."""
    rows = list(csv.DictReader(open(repo / "results" / "summary.csv", encoding="utf-8")))
    row = next(r for r in rows if r["model"] == "ft_cpt_validated" and r["subset"] == "test")
    print(f"\n  summary.csv: ft_cpt_validated/test  WER {row['wer']}  n={row['n_utterances']}")
    if row["wer"] != "1.19":
        fails.append(f"baseline WER is now {row['wer']}, not 1.19")

    ref_file = repo / "results/ft_cpt_validated/test/ref.word-checkpoint_best.pt-test.txt"
    f = NORMALISERS["nfc+scorer+space"]
    refs = [f(re.sub(r"\s*\((?:None-)?\d+\)\s*$", "", l.strip()).replace("<unk>", " "))
            for l in ref_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    train = {f(s) for s in sentences(read(repo / "data/splits/validated_scripted/train.tsv"))}
    hit = sum(1 for r in refs if r in train)
    print(f"  baseline's own scored references present in train: {hit}/{len(refs)} "
          f"({100*hit/len(refs):.2f}%)")
    if hit / len(refs) < 0.99:
        fails.append(f"baseline reference overlap dropped to {100*hit/len(refs):.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--ours", required=True, type=Path)
    a = ap.parse_args()
    fails = []
    print("Leakage verification -- recomputed from the organisers' released files\n")
    check_split("validated_scripted", a.repo / "data/splits/validated_scripted", fails)
    check_split("splits_cv26/scripted", a.repo / "data/splits_cv26/scripted", fails)
    check_split("ours", a.ours, fails)
    check_make_split(a.repo, fails)
    check_baseline(a.repo, fails)
    print()
    if fails:
        print("FAIL:"); [print("  -", x) for x in fails]; sys.exit(1)
    print("PASS: every leakage number in the paper reproduces from the released files.")


if __name__ == "__main__":
    main()
