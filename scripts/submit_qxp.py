"""Transcribe a folder of audio with a checkpoint and write the submission zip."""
import argparse, csv, json, os, re, sys, time, zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_qxp import normalise, read_audio, cer, wer

AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus"}


def find_audio(root):
    files = [p for p in Path(root).rglob("*") if p.suffix.lower() in AUDIO_EXT]
    if not files:
        sys.exit(f"no audio under {root} (looked for {sorted(AUDIO_EXT)})")
    names = [p.name for p in files]
    if len(set(names)) != len(names):
        sys.exit("duplicate basenames under --audio; submission keys would collide")
    return sorted(files, key=lambda p: p.name)


def load_refs(path):
    refs = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            refs[Path(r["path"]).name] = normalise(r["sentence"])
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="required unless --dry-run")
    ap.add_argument("--audio", required=True, help="folder of clips to transcribe")
    ap.add_argument("--vocab", default=None, help="defaults to <ckpt>/vocab.json")
    ap.add_argument("--out", default="qxp", help="basename for the tsv and zip")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-sec", type=float, default=20.0)
    ap.add_argument("--header", action="store_true",
                    help="write a 'filename\\ttranscription' header row")
    ap.add_argument("--refs", default=None,
                    help="optional tsv with path/sentence columns; scores the output")
    ap.add_argument("--dry-run", action="store_true",
                    help="skip the model; emit empty transcriptions. Checks the "
                         "manifest, the zip and the filenames without a GPU.")
    a = ap.parse_args()

    files = find_audio(a.audio)
    if a.dry_run:
        done = {p.name: "" for p in files}
        write_output(a, files, done)
        return
    if not a.ckpt:
        sys.exit("--ckpt is required unless --dry-run")

    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2BertForCTC, AutoConfig

    vocab_path = a.vocab or os.path.join(a.ckpt, "vocab.json")
    vocab = json.load(open(vocab_path, encoding="utf-8"))
    inv = {v: k for k, v in vocab.items()}
    blank = vocab["[PAD]"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = AutoConfig.from_pretrained(a.ckpt)
    is_bert = "bert" in cfg.model_type
    Model = Wav2Vec2BertForCTC if is_bert else Wav2Vec2ForCTC
    fe = None
    if is_bert:
        from transformers import AutoFeatureExtractor
        fe = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    model = Model.from_pretrained(a.ckpt).to(dev).eval()

    part = Path(f"{a.out}.partial.jsonl")
    done = {}
    if part.exists():
        for line in part.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                done[d["file"]] = d["hyp"]
    todo = [p for p in files if p.name not in done]
    print(f"{len(files)} clips  |  {len(done)} already done  |  {len(todo)} to go", flush=True)
    print(f"{dev}  |  {cfg.model_type}  |  vocab {len(vocab)} from {vocab_path}", flush=True)

    # Longest first: the batch that would OOM fails in the first minute, not the last.
    order = sorted(todo, key=lambda p: p.stat().st_size, reverse=True)
    t0 = time.time()
    with part.open("a", encoding="utf-8") as ph:
        for i in range(0, len(order), a.batch):
            chunk = order[i:i + a.batch]
            ys = [read_audio(str(p.parent), p.name, a.max_sec) for p in chunk]
            if fe is not None:
                ys = [fe(y.numpy(), sampling_rate=16000,
                         return_tensors="pt").input_features[0] for y in ys]
            xl = torch.tensor([len(y) for y in ys])
            if ys[0].dim() == 2:
                X = torch.zeros(len(ys), int(max(xl)), ys[0].shape[1])
            else:
                X = torch.zeros(len(ys), int(max(xl)))
            for k, y in enumerate(ys):
                X[k, : len(y)] = y
            am = (torch.arange(X.shape[1])[None, :] < xl[:, None]).long()
            with torch.no_grad():
                lg = model(X.to(dev), attention_mask=am.to(dev)).logits.float()
            out_len = model._get_feat_extract_output_lengths(xl).tolist()
            for k, seq in enumerate(lg.argmax(-1).cpu().tolist()):
                prev, chars = None, []
                for t in seq[: out_len[k]]:
                    if t != prev and t != blank:
                        chars.append(inv.get(t, ""))
                    prev = t
                hyp = re.sub(r"\s+", " ", "".join(chars).replace("|", " ")).strip()
                done[chunk[k].name] = hyp
                ph.write(json.dumps({"file": chunk[k].name, "hyp": hyp},
                                    ensure_ascii=False) + "\n")
            ph.flush()
            os.fsync(ph.fileno())
            if (i // a.batch) % 25 == 0:
                n = min(i + a.batch, len(order))
                print(f"  {n}/{len(order)}  {(time.time()-t0)/60:.1f}m", flush=True)

    write_output(a, files, done)


def write_output(a, files, done):
    tsv = Path(f"{a.out}.tsv")
    with tsv.open("w", encoding="utf-8", newline="\n") as fh:
        if a.header:
            fh.write("filename\ttranscription\n")
        for p in files:
            fh.write(f"{p.name}\t{done[p.name]}\n")

    zp = Path(f"{a.out}.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(tsv, arcname=tsv.name)

    empty = sum(1 for p in files if not done[p.name])
    print(f"wrote {tsv} ({len(files)} rows, {empty} empty) and {zp}")

    if not a.refs:
        return
    refs = load_refs(a.refs)
    ce = cn = we = wn = 0
    n = 0
    for p in files:
        ref = refs.get(p.name)
        if ref is None:
            continue
        x, y = cer(ref, done[p.name]); ce += x; cn += y
        x, y = wer(ref, done[p.name]); we += x; wn += y
        n += 1
    if not n:
        print("no filenames in --refs matched the audio folder; nothing scored")
        return
    print(f"[SCORED] CER {100*ce/cn:.2f}  WER {100*we/wn:.2f}  n={n}")
    if n != len(files):
        print(f"  note: {len(files)-n} clips had no reference and were skipped")


if __name__ == "__main__":
    main()
