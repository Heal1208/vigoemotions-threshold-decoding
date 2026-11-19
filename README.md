# Is one threshold enough? Decision thresholds for multi-label Vietnamese emotion recognition

Course project for Advanced Topics in Information Technology at Hanoi University (class 62FIT4ATI-CLC02, group 02,
project P02).

| Member | Student ID |
|---|---|
| Nguyễn Đức Hiếu | 2301140034 |
| Nguyễn Minh Hiền | 2301140032 |
| Trần Thị Kim Oanh | 2301140071 |

## What we did

[ViGoEmotions](https://github.com/ricardo-tran/ViGoEmotions) has 20,664 Vietnamese social-media comments. Each one
carries one or more of 27 emotions, or `neutral`, so a model has to make 28 yes/no decisions per comment. The labels
are imbalanced, and that raises a question most multi-label code skips: is 0.5 the right cut-off for every sigmoid
output?

The project has two parts.

1. **Midterm.** Dataset analysis and three baselines on the official split: TF-IDF with one-vs-rest logistic
   regression, a BiLSTM and PhoBERT-base-v2.
2. **Final (RQ1).** We freeze the midterm PhoBERT and compare three ways to turn its probabilities into labels:
   `fixed` (0.5 for every label), `global` (one threshold tuned on validation) and `per_label` (one threshold per
   label tuned on validation). Thresholds never see the test split.

## Results

Test split, 28 labels:

| Model and decoding | Macro-F1 | Micro-F1 | Hamming loss |
|---|---|---|---|
| TF-IDF + OvR logistic regression, τ = 0.5 | 0.5365 | 0.5546 | 0.0663 |
| BiLSTM, τ = 0.5 | 0.4892 | 0.5156 | 0.0618 |
| PhoBERT-base-v2, `fixed` (τ = 0.5) | 0.5671 | 0.6051 | **0.0481** |
| PhoBERT-base-v2, `global` (τ* = 0.31) | 0.5958 | **0.6161** | 0.0550 |
| PhoBERT-base-v2, `per_label` (τ_c* from 0.14 to 0.64) | **0.6010** | 0.6131 | 0.0575 |

What we found:

- 0.5 is too high. Tuned on validation, 26 of the 28 per-label thresholds land below 0.5, and the single global one
  lands at 0.31.
- Both tuned rules beat the fixed one on test in all 1,000 paired bootstrap resamples. Per-label thresholds add only
  0.005 over the global one, and that difference is not reliable (95 % CI −0.001 to +0.011).
- The rare labels gain most from per-label thresholds (+0.046 mean F1 in the tail tier) and also over-fit most: their
  validation-to-test gap is the largest.
- The gains have a price. The tuned rules predict about 2.2 labels per comment where the gold average is 1.9, false
  positives more than double, and comments with a single emotion get worse.

All numbers come from one training run with seed 42.

## What is in this repository

```
├── notebooks/
│   ├── 01_Midterm_Baselines.ipynb              dataset analysis, three models, midterm questions, error analysis
│   └── 02_Final_RQ1_Threshold_Decoding.ipynb   RQ1: fixed, global and per-label thresholds
├── configs/            hyperparameters of the three models and the RQ1 settings
├── splits/             train / val / test comment ids and SHA-256 checksums of the CSV files
├── results/
│   ├── tables/         result tables as CSV
│   ├── figures/        figures of notebook 02
│   ├── rq1_thresholds.json   the frozen thresholds of the three strategies
│   ├── rq1_summary.json      headline numbers of RQ1
│   └── environment.md        GPU, library versions, seeds, run times
└── requirements.txt
```

The notebooks are committed with their outputs, so you can read every table and plot without running anything.

### What is not here, and why

| Left out | Reason |
|---|---|
| `data/` (the dataset) | The licence forbids redistributing it. The notebooks download it for you. |
| `runs/` (model weights) | Too large for GitHub (the PhoBERT checkpoint alone is 540 MB). |
| `results/predictions/` and `results/tables/rq1_audit_*.csv` | They pair comment ids with gold labels, which would rebuild the dataset's labels. Running the notebooks recreates them locally. |

These paths are listed in `.gitignore`.

## How to run

You need Python 3 and, for notebook 01, a GPU. A free Colab T4 is enough.

**Google Colab**

```python
!git clone https://github.com/Heal1208/vigoemotions-threshold-decoding
%cd vigoemotions-threshold-decoding
!pip install -q -r requirements.txt
```

Then open `notebooks/01_Midterm_Baselines.ipynb`, set the runtime to T4 GPU and run all. After that, run
`notebooks/02_Final_RQ1_Threshold_Decoding.ipynb`, which needs no GPU.

**Locally**

```bash
git clone https://github.com/Heal1208/vigoemotions-threshold-decoding
cd vigoemotions-threshold-decoding
pip install -r requirements.txt
jupyter notebook notebooks/
```

Run notebook 01 before notebook 02. Notebook 02 only reads the PhoBERT probabilities that notebook 01 writes to
`results/predictions/midterm_phobert_{val,test}.csv`, and those files are not in the repository.

Notes:

- Every path is relative to the repository root. The first cell of each notebook finds the root by itself, whether
  you start Jupyter in the root or in `notebooks/`.
- If `data/` is empty, the first cell downloads `train.csv`, `val.csv`, `test.csv` and `label_dict.json` from the
  authors' repository and checks that the ids match `splits/`. To check the files byte for byte:
  `cd data && sha256sum -c ../splits/SHA256SUMS.txt`.
- Notebook 01 saves each trained model in `runs/<model>/`. On the next run it loads a saved model with the same
  configuration instead of training again; set `RETRAIN = True` to force training. A full run from scratch took about
  35 minutes on a Colab T4 (PhoBERT alone about 28 minutes for two learning rates).
- The notebooks print comment ids, never comment text, unless you set `SHOW_COMMENT_TEXT = True`. Leave it off in
  anything you share.

## Method in brief

- **Split:** the official one (16,531 / 2,066 / 2,067), unchanged.
- **Text pipeline** (same input for all models): Unicode NFC, emoji turned into English names with `emoji.demojize`,
  letters repeated three or more times cut to one and symbols cut to two, word segmentation with `underthesea`.
  TF-IDF and the BiLSTM lowercase the text; PhoBERT keeps case. Teencode is left as it is.
- **Selection:** hyperparameters and epochs chosen on validation Macro-F1 at τ = 0.5. The test split is scored once
  per model and once per decoding rule.
- **Thresholds:** searched on 0.05, 0.06, …, 0.95 on validation, written to `results/rq1_thresholds.json`, then
  applied to test.
- **Tiers:** the 28 labels ranked by training count; head = top 9, mid = next 10, tail = last 9. Fixed before any
  evaluation.
- **Checkpoint:** `vinai/phobert-base-v2`. Exact settings are in `configs/`, library versions in
  `results/environment.md`.

## Where each result comes from

| File | Produced by |
|---|---|
| `results/tables/midterm_main_metrics.csv`, `midterm_per_label_f1.csv` | notebook 01, Section 5 |
| `results/tables/rq1_table_main_metrics.csv`, `rq1_table_val_test_gap.csv` | notebook 02, Section 5 |
| `results/tables/rq1_table_bootstrap_ci.csv` | notebook 02, Section 5.1 |
| `results/tables/rq1_table_per_label_f1.csv`, `figures/rq1_per_label_f1.png` | notebook 02, Section 6 |
| `results/tables/rq1_table_tiers.csv`, `figures/rq1_tiers.png` | notebook 02, Section 7 |
| `results/tables/rq1_table_cardinality.csv`, `figures/rq1_cardinality.png` | notebook 02, Section 8 |
| `results/tables/rq1_table_threshold_profile.csv`, `figures/rq1_threshold_profile.png`, `figures/rq1_f1_vs_threshold_per_label.png` | notebook 02, Section 9 |
| `results/tables/rq1_table_error_transitions.csv`, `rq1_table_confusions.csv`, `rq1_table_pair_recall.csv` | notebook 02, Section 10 |

The dataset analysis, the midterm plots and the confusion matrices are drawn inside notebook 01.

## Data and licence

The data come from the authors' repository, folder `corpus/`. Their licence says the dataset may be used only for
non-profit research and education, must not be used in commercial systems and must not be redistributed, while
summaries and analyses may be published if the original data cannot be rebuilt from them. This repository follows
those terms: it holds code, comment ids, aggregate results and figures, and no comment text or per-comment labels.

If you use the dataset, cite the authors:

> H. Q. Tran, N. T. Pham, S. T. Luu, and K. V. Nguyen, "ViGoEmotions: A benchmark dataset for fine-grained emotion
> detection on Vietnamese texts," in *Proc. 19th Conf. Eur. Chapter Assoc. Comput. Linguistics (EACL)*, vol. 1,
> 2026, pp. 2805–2831, doi: 10.18653/v1/2026.eacl-long.129.
