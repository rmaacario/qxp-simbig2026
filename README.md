# USP at SIMBig 2026 Shared Task 1 — Puno Quechua ASR

Code, manifests and per-utterance predictions for the paper *USP at SIMBig 2026
Task 1: Sentence Leakage, Encoder Choice, and an Orthographic Burden in Puno
Quechua ASR*.

Every error rate in the paper can be recomputed from `predictions/` without a
GPU. Reproducing the models themselves requires one.

## Sentence leakage

`scripts/verify_leakage.py` recomputes every figure in Section 3.1 from the
organisers' own published files, under six matching rules, and exits non-zero
if any of them no longer holds.

```bash
git clone https://github.com/QuechuaBase/asr-puno-quechua
python scripts/verify_leakage.py --repo asr-puno-quechua --ours splits/
```

## The sentence-disjoint split

`splits/` holds the split the paper reports all results on: 18,168 train /
2,284 dev / 2,275 test clips over 1,650 / 207 / 210 disjoint prompts. No test
or development sentence occurs in the training text. Rebuild it with:

```bash
python scripts/make_split_grouped.py \
    --input <cv-corpus>/qxp/validated.tsv --output-dir splits/
```

`scripts/make_silver_manifests.py` builds the spontaneous manifests used by the
submitted system: 6,442 training and 244 development clips, after dropping
clips over 20 s and clips whose text needs symbols outside the 36-symbol head.

## Predictions

`predictions/` holds reference and hypothesis for all 2,275 test clips, one
file per system, alongside the metrics each produced.

| file | system | CER | WER |
|---|---|---|---|
| `xlsr_test.tsv` | XLS-R 300M | 4.43 | 28.89 |
| `w2vbert_test.tsv` | w2v-BERT 2.0 + adapter | 3.12 | 20.11 |
| `w2vbert_noadapter_test.json` | w2v-BERT 2.0, `add_adapter=False` | 69.65 | 100.00 |

## Training and inference

```bash
python scripts/train_qxp.py --model facebook/w2v-bert-2.0 \
    --splits splits/ --clips <audio> --vocab vocab_qxp.json --seed 1
python scripts/eval_qxp.py --ckpt <run>/best --split test --clips <audio>
python scripts/submit_qxp.py --ckpt <run>/best --audio <test audio> \
    --out qxp --max-sec 30
```

`--max-sec 30` at inference: eleven spontaneous test clips run past 20 s, and
the loader truncates audio without truncating the reference.

Seeds are set once per run across Python, NumPy and Torch. Deterministic
kernels are not requested, so a rerun reproduces the reported figures closely
rather than bitwise.

## Notebooks

Each notebook runs on Kaggle and produced one set of reported numbers.

| notebook | produced |
|---|---|
| `qxp_kaggle_resumable.ipynb` | XLS-R 300M baseline, 4.43 / 28.89 |
| `qxp_probe.ipynb` | the `add_adapter` ablation, 69.65 / 100.00 |
| `qxp_full_run.ipynb` | w2v-BERT 2.0 + adapter, 3.12 / 20.11 |
| `qxp_vs2.ipynb` | the submitted system |
| `qxp_submit_vs2.ipynb` | the submitted transcriptions |

## Data

Audio and prompts come from Mozilla Common Voice (Puno Quechua, qxp). The
blind test audio is not redistributed here.
