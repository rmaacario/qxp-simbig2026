"""Score a saved checkpoint on a split."""
import argparse, csv, json, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_qxp import normalise, read_audio, cer, wer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", default="splits_grouped")
    ap.add_argument("--split", default="test")
    ap.add_argument("--clips", required=True)
    ap.add_argument("--vocab", default="vocab_qxp.json")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-sec", type=float, default=20.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2BertForCTC, AutoConfig

    vocab = json.load(open(a.vocab, encoding="utf-8"))
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

    rows = []
    with open(f"{a.splits}/{a.split}.tsv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            t = normalise(r["sentence"])
            if t:
                rows.append((r["path"], t))
    print(f"{len(rows)} clips from {a.split}  |  {dev}  |  {cfg.model_type}", flush=True)

    ce = cn = we = wn = 0
    preds = []
    t0 = time.time()
    for i in range(0, len(rows), a.batch):
        chunk = rows[i:i + a.batch]
        ys = [read_audio(a.clips, p, a.max_sec) for p, _ in chunk]
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
            ref = chunk[k][1]
            preds.append((chunk[k][0], ref, hyp))
            x, y = cer(ref, hyp); ce += x; cn += y
            x, y = wer(ref, hyp); we += x; wn += y
        if (i // a.batch) % 25 == 0:
            done = min(i + a.batch, len(rows))
            print(f"  {done}/{len(rows)}  running CER {100*ce/max(cn,1):.2f}  "
                  f"WER {100*we/max(wn,1):.2f}  {(time.time()-t0)/60:.1f}m", flush=True)

    m = {"split": a.split, "ckpt": a.ckpt, "n": len(preds),
         "cer": 100 * ce / cn, "wer": 100 * we / wn}
    print(f"\n[{a.split.upper()}] CER {m['cer']:.2f}  WER {m['wer']:.2f}  n={m['n']}")
    out = a.out or f"eval_{a.split}.json"
    json.dump(m, open(out, "w"), indent=1)
    with open(out.replace(".json", ".tsv"), "w", encoding="utf-8") as fh:
        for p, ref, hyp in preds:
            fh.write(f"{p}\t{ref}\t{hyp}\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
