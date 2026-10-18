"""Fine-tune XLS-R 300M or w2v-BERT 2.0 for Puno Quechua CTC ASR."""

import argparse
import json
import os
import random
import re
import time

# The scorer the organisers use, reproduced exactly (eval/omnilingual/compute_wer.py):
#   "".join(c for c in text.lower() if c not in "?!¿¡.,").strip()
# Training targets go through this and nothing else, so that what the model
# learns to emit is what the references are scored as.
SCORE_STRIP = set("?!¿¡.,")


def normalise(text):
    return "".join(c for c in str(text).lower() if c not in SCORE_STRIP).strip()


def set_all_seeds(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # MPS lacks deterministic kernels


def build_vocab(texts, path):
    chars = sorted({c for t in texts for c in t if c != " "})
    vocab = {"[PAD]": 0, "[UNK]": 1, "|": 2}
    for c in chars:
        vocab.setdefault(c, len(vocab))
    json.dump(vocab, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"vocab: {len(vocab)} symbols -> {path}")
    return vocab


def cer(ref, hyp):
    r, h = list(ref), list(hyp)
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1], max(len(r), 1)


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1], max(len(r), 1)


_RS = {}


def read_audio(clips_dir, path, max_sec):
    """Common Voice ships MP3 at 32 kHz; wav2vec2 wants 16 kHz mono float.

    Decoding goes through soundfile rather than torchaudio.load: as of
    torchaudio 2.11 that call delegates to TorchCodec and raises without it,
    while libsndfile reads these MP3s directly. Resamplers are cached per
    source rate — building one per clip dominates data loading otherwise.
    """
    import os
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    y, sr = sf.read(os.path.join(clips_dir, path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    y = torch.from_numpy(np.ascontiguousarray(y))
    if sr != 16000:
        if sr not in _RS:
            _RS[sr] = torchaudio.transforms.Resample(sr, 16000)
        y = _RS[sr](y)
    if len(y) > max_sec * 16000:
        y = y[: int(max_sec * 16000)]
    return y


class Utt:
    """Module-level so it pickles under the spawn start method."""

    def __init__(self, rows, clips_dir, vocab, max_sec, augment=False,
                 feature_extractor=None):
        self.rows, self.clips_dir = rows, clips_dir
        self.vocab, self.max_sec, self.augment = vocab, max_sec, augment
        # w2v-BERT consumes 80-bin mel filterbanks, not raw audio. Passing a
        # feature extractor switches this dataset to emit those instead.
        self.fe = feature_extractor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import numpy as np
        import torch
        r = self.rows[i]
        y = read_audio(self.clips_dir, r["path"], self.max_sec)
        if self.augment:
            # Speed perturbation only. Pitch shift would distort the vowel
            # qualities carrying the trivocalic/loanword contrast.
            f = random.choice([0.9, 1.0, 1.1])
            if f != 1.0:
                n = int(len(y) / f)
                y = torch.from_numpy(np.interp(
                    np.linspace(0, len(y) - 1, n), np.arange(len(y)),
                    y.numpy()).astype("float32"))
        # Spaces become the word-separator symbol, never dropped: without ⟨|⟩
        # the model emits an unbroken character string, CER inflates by every
        # missing space and WER pins at 100%.
        ids = [self.vocab["|"] if c == " "
               else self.vocab.get(c, self.vocab["[UNK]"]) for c in r["text"]]
        if self.fe is not None:
            y = self.fe(y.numpy(), sampling_rate=16000,
                        return_tensors="pt").input_features[0]
        return y, torch.tensor(ids), r["path"]


def collate(b):
    """Pad a batch. Raw-waveform models (XLS-R) get 1-D tensors; w2v-BERT gets
    2-D mel frames, so the padded tensor needs the feature axis."""
    import torch
    xs, ys, ids = zip(*b)
    xl = torch.tensor([len(x) for x in xs])
    yl = torch.tensor([len(y) for y in ys])
    if xs[0].dim() == 2:                       # (frames, mel_bins)
        X = torch.zeros(len(xs), int(max(xl)), xs[0].shape[1])
    else:                                      # (samples,)
        X = torch.zeros(len(xs), int(max(xl)))
    for i, x in enumerate(xs):
        X[i, : len(x)] = x
    return X, torch.cat(ys), xl, yl, ids



def save_ckpt(path, model, opt, sched, scaler, step, epoch, best, args):
    """Checkpoint holding everything needed to resume mid-epoch.

    Written to a temp path and renamed so an interrupted save cannot leave a
    corrupt file. Holds optimiser,
    scheduler, AMP scaler, step, epoch, best-so-far and RNG state — weights
    alone would silently restart the schedule and the data order.

    **Optimiser state is dropped above ~400M parameters.** AdamW keeps two
    fp32 tensors per parameter, so a 581M model's checkpoint is ~7 GB and the
    atomic rename needs 14 GB at peak — which overran Kaggle's 19.5 GB working
    directory and killed a w2v-BERT run. Without it the file is ~2.3 GB.
    Resume then restarts Adam's moments, costing a few steps of momentum;
    that is a far better trade than not being able to resume at all.

    Refuses to write at all below 3 GB free, because a half-written checkpoint
    plus a full disk is how you lose both the run and the ability to restart it.
    """
    import os, random, shutil, numpy as np, torch
    free_gb = shutil.disk_usage(os.path.dirname(path) or ".").free / 1e9
    if free_gb < 3.0:
        print(f"  !! only {free_gb:.1f} GB free — skipping checkpoint", flush=True)
        return
    n_par = sum(p.numel() for p in model.parameters())
    light = n_par > 400e6
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "opt": None if light else opt.state_dict(),
        "sched": sched.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step, "epoch": epoch, "best": best,
        "args": vars(args),
        "rng": {"python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state()},
    }, tmp)
    os.replace(tmp, path)


def load_ckpt(path, model, opt, sched, scaler):
    import os, random, numpy as np, torch
    if not os.path.exists(path):
        return 0, 0, 1e9
    try:
        c = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  checkpoint at {path} unreadable ({e}); starting fresh", flush=True)
        return 0, 0, 1e9
    model.load_state_dict(c["model"])
    if c.get("opt") is not None:
        opt.load_state_dict(c["opt"])
    else:
        print("  (checkpoint has no optimiser state — Adam moments restart)", flush=True)
    sched.load_state_dict(c["sched"])
    scaler.load_state_dict(c["scaler"])
    random.setstate(c["rng"]["python"])
    np.random.set_state(c["rng"]["numpy"])
    torch.set_rng_state(c["rng"]["torch"].cpu().to(torch.uint8))
    print(f"  RESUMED from {path}: epoch {c['epoch']}, step {c['step']}, "
          f"best dev CER {c['best']:.2f}", flush=True)
    return c["step"], c["epoch"], c["best"]



def hf_push(local_dir, repo, token, note=""):
    """Mirror a finished checkpoint to the Hugging Face Hub.

    Survives the container. Kaggle
    writes /kaggle/working to a version only when a commit *completes*; a run
    that hangs or is cancelled takes every local checkpoint with it, which is
    exactly how 10 GPU-hours were lost. Pushing out of the box means the run
    can die at any moment without costing the weights.

    Failures here are logged and swallowed: a network problem must never kill
    a training run that is otherwise fine.
    """
    if not repo or not token:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo, private=True, exist_ok=True)
        api.upload_folder(folder_path=local_dir, repo_id=repo,
                          commit_message=note or "checkpoint")
        print(f"  pushed {local_dir} -> hf.co/{repo} ({note})", flush=True)
    except Exception as e:
        print(f"  HF push failed (continuing): {type(e).__name__}: {e}", flush=True)


def hf_pull(repo, token, dest):
    """Restore a checkpoint pushed by a previous, dead session."""
    if not repo or not token:
        return False
    try:
        from huggingface_hub import snapshot_download
        import os
        if os.path.exists(os.path.join(dest, "config.json")):
            return False                      # local copy already present
        snapshot_download(repo_id=repo, token=token, local_dir=dest)
        print(f"  restored {repo} -> {dest}", flush=True)
        return True
    except Exception as e:
        print(f"  no HF checkpoint to restore ({type(e).__name__})", flush=True)
        return False



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits_grouped")
    ap.add_argument("--clips", default="data/raw/cv-corpus-26.0-2026-06-12/qxp/clips")
    ap.add_argument("--vocab", default="vocab_qxp.json")
    ap.add_argument("--model", default="facebook/wav2vec2-xls-r-300m")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)   # baseline paper, XLS-R-300M
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--max-sec", type=float, default=20.0)
    ap.add_argument("--speed-perturb", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap rows per split")
    ap.add_argument("--out", default="runs_qxp")
    ap.add_argument("--save-every", type=int, default=100,
                    help="checkpoint every N optimiser steps; 0 disables")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--hf-repo", default="",
                    help="HF repo that best/ is mirrored to")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--abort-step", type=int, default=0,
                    help="run a dev eval at this step and stop if it is bad")
    ap.add_argument("--abort-cer", type=float, default=100.0,
                    help="with --abort-step: abort if dev CER exceeds this")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="stop after N optimiser steps; 0 = run all epochs")
    ap.add_argument("--mask-time-prob", type=float, default=None,
                    help="default: 0.0 for w2v-BERT (official recipe), 0.05 for XLS-R")
    args = ap.parse_args()

    set_all_seeds(args.seed)
    import csv
    import numpy as np
    import torch
    import torchaudio
    import transformers

    IS_BERT = "w2v-bert" in args.model.lower() or "bert" in args.model.lower()
    if IS_BERT:
        from transformers import (Wav2Vec2BertForCTC as CTCModel,
                                  AutoFeatureExtractor)
        feat_ex = AutoFeatureExtractor.from_pretrained(args.model)
    else:
        from transformers import Wav2Vec2ForCTC as CTCModel
        feat_ex = None

    dev = ("cuda" if torch.cuda.is_available() else
           "mps" if torch.backends.mps.is_available() else "cpu")

    def load_split(name):
        rows = []
        with open(f"{args.splits}/{name}.tsv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
                t = normalise(r["sentence"])
                if t:
                    rows.append({"path": r["path"], "text": t,
                                 "client_id": r.get("client_id", "")})
        return rows[: args.limit] if args.limit else rows

    tr, dv, te = load_split("train"), load_split("dev"), load_split("test")
    print(f"train {len(tr)}  dev {len(dv)}  test {len(te)}  |  {dev}", flush=True)

    trs = {r["text"] for r in tr}
    for n, s in (("dev", dv), ("test", te)):
        leak = sum(1 for r in s if r["text"] in trs)
        print(f"  sentence leakage {n}: {leak}/{len(s)} ({100*leak/max(len(s),1):.2f}%)")

    vocab = (json.load(open(args.vocab, encoding="utf-8"))
             if os.path.exists(args.vocab)
             else build_vocab([r["text"] for r in tr], args.vocab))
    inv = {v: k for k, v in vocab.items()}
    blank = vocab["[PAD]"]

    class LengthBucketed(torch.utils.data.Sampler):
        """Batch clips of similar length. Padding is charged at the longest
        item, so mixing a 3 s clip with a 20 s one wastes most of the compute."""
        def __init__(self, rows, batch, seed):
            import soundfile  # noqa: F401  (duration via header only)
            self.rows, self.batch, self.seed, self.epoch = rows, batch, seed, 0
            self.order = list(range(len(rows)))

        def set_lengths(self, lengths):
            self.order = sorted(range(len(lengths)), key=lambda i: lengths[i])

        def __iter__(self):
            batches = [self.order[i:i + self.batch]
                       for i in range(0, len(self.order), self.batch)]
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(batches)
            self.epoch += 1
            for b in batches:
                yield b

        def __len__(self):
            return (len(self.order) + self.batch - 1) // self.batch

    # Architecture-specific settings, from HuggingFace's own recipes rather than
    # reused across families. Getting this wrong is not a hyperparameter nicety:
    # w2v-BERT without `add_adapter` emits roughly twice as many CTC frames per
    # utterance, and a 581M model then spends its whole budget learning a
    # longer alignment instead of the language. Observed as loss stalling at
    # ~2.4 and dev CER 70.9 after an epoch, where XLS-R was at 0.35 and 6.15.
    #   w2v-BERT: huggingface/blog/fine-tune-w2v2-bert.md
    #   XLS-R:    huggingface/blog/fine-tune-xlsr-wav2vec2.md
    mtp = args.mask_time_prob
    kw = dict(vocab_size=len(vocab), ignore_mismatched_sizes=True,
              ctc_loss_reduction="mean", pad_token_id=blank,
              attention_dropout=0.0, hidden_dropout=0.0,
              feat_proj_dropout=0.0, layerdrop=0.0)
    if IS_BERT:
        kw["add_adapter"] = True                      # stride-2 conv adapter
        kw["mask_time_prob"] = 0.0 if mtp is None else mtp
    else:
        kw["mask_time_prob"] = 0.05 if mtp is None else mtp
    model = CTCModel.from_pretrained(args.model, **kw)
    print(f"  arch settings: add_adapter={kw.get('add_adapter', False)} "
          f"mask_time_prob={kw['mask_time_prob']}", flush=True)
    # The CNN feature encoder stays frozen: it encodes generic acoustics learnt
    # from far more speech than Quechua has, and 30 h is not enough to improve
    # it. Everything above it trains.
    if hasattr(model, "freeze_feature_encoder") and not IS_BERT:
        model.freeze_feature_encoder()
    model.gradient_checkpointing_enable()
    model.to(dev)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable {n_tr/1e6:.0f}M of "
          f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M", flush=True)

    dl = torch.utils.data.DataLoader(
        Utt(tr, args.clips, vocab, args.max_sec,
            augment=args.speed_perturb, feature_extractor=feat_ex), batch_size=args.batch,
        shuffle=True, collate_fn=collate, num_workers=args.workers, drop_last=True)
    steps = (len(dl) // args.accum) * args.epochs
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = transformers.get_linear_schedule_with_warmup(
        opt, int(args.warmup * steps), steps)
    print(f"  {steps} optimiser steps over {args.epochs} epochs", flush=True)

    @torch.no_grad()
    def evaluate(rows, tag):
        model.eval()
        ce = cn = we = wn = 0
        loader = torch.utils.data.DataLoader(
            Utt(rows, args.clips, vocab, args.max_sec, feature_extractor=feat_ex),
            batch_size=args.batch, collate_fn=collate, num_workers=0)
        preds = []
        nb = len(loader)
        for bi, (X, _, xl, _, ids) in enumerate(loader):
            if bi % 25 == 0:
                print(f"    [{tag}] batch {bi}/{nb}", flush=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                lg = model(X.to(dev),
                           attention_mask=(torch.arange(X.shape[1])[None, :]
                                           < xl[:, None]).long().to(dev)).logits
            lg = lg.float()
            out_len = model._get_feat_extract_output_lengths(xl).tolist()
            for k, seq in enumerate(lg.argmax(-1).cpu().tolist()):
                # CTC collapse: drop repeats, then drop blanks.
                prev, chars = None, []
                for t in seq[: out_len[k]]:
                    if t != prev and t != blank:
                        chars.append(inv.get(t, ""))
                    prev = t
                hyp = "".join(chars).replace("|", " ").strip()
                hyp = re.sub(r"\s+", " ", hyp)
                preds.append((ids[k], hyp))
        ref = {r["path"]: r["text"] for r in rows}
        for pid, hyp in preds:
            a, b = cer(ref[pid], hyp); ce += a; cn += b
            a, b = wer(ref[pid], hyp); we += a; wn += b
        model.train()
        return {"tag": tag, "cer": 100 * ce / cn, "wer": 100 * we / wn,
                "n": len(preds)}, preds

    os.makedirs(args.out, exist_ok=True)
    run = f"{args.out}/seed{args.seed}"
    os.makedirs(run, exist_ok=True)
    model.train()
    # The resume checkpoint is ephemeral and not part of the output; on Kaggle
    # it goes to /kaggle/temp, which is outside the working-directory quota.
    CKPT_DIR = "/kaggle/temp" if os.path.isdir("/kaggle/temp") else run
    os.makedirs(CKPT_DIR, exist_ok=True)
    CKPT = f"{CKPT_DIR}/{os.path.basename(args.out)}_seed{args.seed}_last.pt"
    print(f"  resume checkpoint -> {CKPT}", flush=True)
    import shutil as _sh
    for _d in ("/kaggle/working", CKPT_DIR):
        if os.path.isdir(_d):
            print(f"  free on {_d}: {_sh.disk_usage(_d).free/1e9:.1f} GB", flush=True)
    if args.hf_repo and not os.path.exists(CKPT):
        hf_pull(args.hf_repo, args.hf_token, f"{run}/best")
    step, start_ep, best = load_ckpt(CKPT, model, opt, sched, scaler)
    model.to(dev)
    if step:
        print(f"  skipping {step} completed steps", flush=True)
    t0 = time.time()
    for ep in range(start_ep, args.epochs):
        run_loss = k = 0
        # Steps already completed in a previous session are skipped without
        # a forward pass, so a resumed run does not redo work or re-consume
        # the schedule.
        seen = 0
        for i, (X, Y, xl, yl, _) in enumerate(dl):
            if seen < step - ep * (len(dl) // args.accum) * args.accum:
                seen += 1
                continue
            am = (torch.arange(X.shape[1])[None, :] < xl[:, None]).long()
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                out = model(X.to(dev), attention_mask=am.to(dev))
            # CTC loss stays in fp32 regardless: log_softmax over a long
            # sequence underflows in half precision and the loss goes to inf.
            lg = out.logits.float().log_softmax(-1).transpose(0, 1)
            loss = torch.nn.functional.ctc_loss(
                lg, Y.to(dev),
                model._get_feat_extract_output_lengths(xl).to(dev), yl.to(dev),
                blank=blank, reduction="mean", zero_infinity=True)
            scaler.scale(loss / args.accum).backward()
            run_loss += loss.item(); k += 1
            if (i + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    print(f"  ep{ep} step {step}/{steps} loss {run_loss/max(k,1):.3f} "
                          f"{(time.time()-t0)/60:.1f}m", flush=True)
                    run_loss = k = 0
                if args.save_every and step % args.save_every == 0:
                    save_ckpt(CKPT, model, opt, sched, scaler, step, ep, best, args)
                if args.abort_step and step == args.abort_step:
                    am_, _ = evaluate(dv, "abort-check")
                    print(f"  [ABORT CHECK] step {step} dev CER {am_['cer']:.2f} "
                          f"(threshold {args.abort_cer})", flush=True)
                    if am_["cer"] > args.abort_cer:
                        raise SystemExit(
                            f"ABORTED at step {step}: dev CER {am_['cer']:.2f} > "
                            f"{args.abort_cer}. Not spending the remaining hours.")
                    print("  [ABORT CHECK] passed — continuing", flush=True)
                if args.max_steps and step >= args.max_steps:
                    print(f"  reached --max-steps {args.max_steps}; stopping",
                          flush=True)
                    break
        if args.max_steps and step >= args.max_steps:
            break
        save_ckpt(CKPT, model, opt, sched, scaler, step, ep + 1, best, args)
        m, _ = evaluate(dv, "dev")
        print(f"[epoch {ep}] dev CER {m['cer']:.2f}  WER {m['wer']:.2f}", flush=True)
        if m["cer"] < best:
            best = m["cer"]
            save_ckpt(CKPT, model, opt, sched, scaler, step, ep + 1, best, args)
            model.save_pretrained(f"{run}/best")
            json.dump(vocab, open(f"{run}/best/vocab.json", "w"), ensure_ascii=False)
            json.dump({"epoch": ep, "step": step, "dev": m},
                      open(f"{run}/best/dev_metrics.json", "w"), indent=1)
            hf_push(f"{run}/best", args.hf_repo, args.hf_token,
                    f"epoch {ep} dev CER {m['cer']:.2f} WER {m['wer']:.2f}")

    if args.max_steps:
        m, _ = evaluate(dv, "dev")
        print(f"\n[PROBE] {step} steps  dev CER {m['cer']:.2f}  "
              f"WER {m['wer']:.2f}", flush=True)
        print("[PROBE] unfixed run for comparison: "
              "loss 2.42 @ step 1200, dev CER 70.88 after epoch 1", flush=True)
        os.makedirs(f"{run}/best", exist_ok=True)
        model.save_pretrained(f"{run}/best")
        json.dump(vocab, open(f"{run}/best/vocab.json", "w"), ensure_ascii=False)
        json.dump({"probe": True, "steps": step, "dev": m, "args": vars(args)},
                  open(f"{run}/probe_result.json", "w"), indent=1)
        hf_push(f"{run}/best", args.hf_repo, args.hf_token,
                f"probe {step} steps dev CER {m['cer']:.2f}")
        return

    m, preds = evaluate(te, "test")
    print(f"\n[TEST] CER {m['cer']:.2f}  WER {m['wer']:.2f}  n={m['n']}", flush=True)
    with open(f"{run}/predictions_test.tsv", "w", encoding="utf-8") as fh:
        for pid, hyp in preds:
            fh.write(f"{pid}\t{hyp}\n")
    json.dump({"test": m, "args": vars(args),
               "versions": {"transformers": transformers.__version__,
                            "torch": torch.__version__}},
              open(f"{run}/results.json", "w"), indent=1)
    print(f"wrote {run}/results.json")
    if os.path.exists(CKPT):
        os.remove(CKPT)
        print(f"removed {CKPT} (run complete)")


if __name__ == "__main__":
    main()
