# USP at SIMBig 2026 Shared Task 1 — Puno Quechua ASR

Code, manifests and per-utterance predictions for the paper *USP at SIMBig 2026
Task 1: Sentence Leakage, Encoder Choice, and Orthographic Inconsistency in Puno
Quechua ASR*.

Every error rate in the paper can be recomputed from `predictions/` without a
GPU. Reproducing the models themselves requires one.

## The language

Puno Quechua (ISO 639-3: **qxp**) is a Cusco-Collao variety of Southern Quechua
(QIIC), spoken by roughly **474,000** people in the Puno region of the Peruvian
altiplano. Its share of the regional population has fallen steadily — 87.23% in
1940, 50% in 1993, 38.01% in 2007 — and formal education is conducted almost
entirely in Spanish, so most speakers are not literate in their own language.
The first dedicated speech resources for it appeared in 2026.

It is agglutinative, verb-final, and phonologically demanding for recognition:
26 consonant phonemes including a three-way plain / aspirated / ejective stop
contrast alongside a uvular series, over three vowels /a i u/, with Aymara
influence in the phonology, suffix inventory and lexicon. One misplaced
apostrophe turns one word into another.

**Why a variety-specific system.** Quechua ASR work has tended to pool varieties
under "Southern Quechua" or the macrolanguage code `que`, conflating varieties
that are not mutually intelligible. The pooling is not merely nominal — the same
lexical item differs across neighboring varieties:

| variety | form |
|---|---|
| `quz` Cusco | askha |
| `quy` Ayacucho-Chanka | achka |
| `qve` Eastern Apurímac | aska |
| **`qxp` Puno** | **ashkha** |

The Puno form alone carries both the postalveolar and the aspirated stop. A
system trained or scored against another variety's orthography is penalized for
spellings its own variety never uses.

## The shared task

SIMBig 2026 Shared Task 1 asks for ASR on Puno Quechua using the Mozilla Common
Voice collection: 22,727 validated scripted clips (30.49 h) from 77 speakers,
plus 1,222 transcribed (5.81 h) and 6,083 untranscribed (29.60 h) spontaneous
clips. The blind test is 659 clips — 438 scripted and 221 spontaneous.

Scoring is **literal**: lowercase, strip `? ! ¿ ¡ . ,` and nothing else. No
normalization is available to participants, which is what makes orthographic
variation in the prompts a cost rather than a preprocessing choice.

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

## Sources

Figures for the language and the collection are from Huaman et al., *Quechua
Speech Datasets in Common Voice: The Case of Puno Quechua* (arXiv:2510.13871),
and Huaman et al., *Building Community-Centred NLP Resources for Puno Quechua*
(AmericasNLP 2026), which also supplies the cross-variety lexical comparison.
Corpus counts quoted here were recounted from the released manifests.

Model weights: https://huggingface.co/rmacario/qxp-vs2
