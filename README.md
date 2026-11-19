# ViGoEmotions: multi-label Vietnamese emotion recognition and threshold decoding

Advanced Topics in Information Technology, class 62FIT4ATI-CLC02, Group 02, Project P02.

| Member | Student ID |
|---|---|
| Nguyễn Đức Hiếu (submitter) | 2301140034 |
| Nguyễn Minh Hiền | 2301140032 |
| Trần Thị Kim Oanh | 2301140071 |

GitHub repository: <https://github.com/Heal1208/vigoemotions-threshold-decoding>

## What is in this project

The midterm compares three baselines on ViGoEmotions (20,664 Vietnamese social-media comments, 27 emotions plus
`neutral`, 28 labels). The final part answers the required research question, RQ1: is one uniform decision threshold
of 0.5 the right choice for every label when the labels are imbalanced? We keep the midterm PhoBERT model fixed and
compare three ways of turning its probabilities into labels.

Test results (Macro-F1 over 28 labels):

| | Macro-F1 | Micro-F1 | Hamming loss |
|---|---|---|---|
| TF-IDF + OvR logistic regression, τ = 0.5 | 0.5365 | 0.5546 | 0.0663 |
| BiLSTM, τ = 0.5 | 0.4892 | 0.5156 | 0.0618 |
| PhoBERT-base-v2, τ = 0.5 (`fixed`) | 0.5671 | 0.6051 | 0.0481 |
| PhoBERT-base-v2, one threshold tuned on validation (`global`, τ* = 0.31) | 0.5958 | 0.6161 | 0.0550 |
| PhoBERT-base-v2, one threshold per label tuned on validation (`per_label`) | 0.6010 | 0.6131 | 0.0575 |

Sources: `results/tables/midterm_main_metrics.csv` and `results/tables/rq1_table_main_metrics.csv`.

## Folder layout

```
code/
├── README.md               this file
├── requirements.txt
├── notebooks/
│   ├── 01_Midterm_Baselines.ipynb             dataset analysis, 3 models, midterm questions, error analysis
│   └── 02_Final_RQ1_Threshold_Decoding.ipynb  RQ1: fixed / global / per-label thresholds
├── configs/                hyperparameters of every experiment (JSON)
├── splits/                 train / val / test comment ids + SHA-256 of the three CSV files
├── src/                    empty: all code lives in the notebooks
└── results/
    ├── tables/             every result table as CSV
    ├── figures/            figures of notebook 02
    ├── predictions/        per-comment probabilities and predictions (CSV, plus one .npz)
    ├── rq1_thresholds.json frozen thresholds of the three strategies
    ├── rq1_summary.json    headline numbers of RQ1
    └── environment.md      hardware, library versions, seeds
```

## Data

- Source: the authors' repository <https://github.com/ricardo-tran/ViGoEmotions>, folder `corpus/`
  (`train.csv`, `val.csv`, `test.csv`, `label_dict.json`). The dataset card is
  <https://huggingface.co/datasets/uitnlp/vigoemotions>.
- Licence: non-commercial academic research and education only. Raw comment text must not be redistributed, so the
  submission contains no dataset file and no comment text, only comment ids, labels and model outputs.
- You do not have to download anything by hand. If `code/data/` is missing or empty, the first cell of each notebook
  downloads the four files from the GitHub repository into `code/data/` and checks that the ids match `splits/`.
- Split: the official split, unchanged (16,531 / 2,066 / 2,067). `splits/*_ids.txt` lists the ids in file order.
  `splits/SHA256SUMS.txt` holds the checksums of the CSV files as downloaded from GitHub; check them with
  `cd code/data && sha256sum -c ../splits/SHA256SUMS.txt`.

## How to run

All paths are relative to this `code/` folder. The notebooks find it on their own.

**Google Colab**

1. Upload the unzipped submission folder `62FIT4ATI-CLC02_G02_P02/` anywhere in *My Drive*.
2. Open `code/notebooks/01_Midterm_Baselines.ipynb` in Colab, choose Runtime → Change runtime type → T4 GPU, then
   Runtime → Run all. The first cell mounts Drive and looks for `62FIT4ATI-CLC02_G02_P02/code` up to four folders
   deep.
3. Open `code/notebooks/02_Final_RQ1_Threshold_Decoding.ipynb` and run all. A CPU runtime is enough.

From the GitHub repository instead, run `!git clone https://github.com/Heal1208/vigoemotions-threshold-decoding`
and `%cd vigoemotions-threshold-decoding` in the first Colab cell; the notebooks then find the repository root on
their own.

**Local Jupyter**

```bash
cd 62FIT4ATI-CLC02_G02_P02/code
pip install -r requirements.txt
cd notebooks
jupyter notebook        # run 01_Midterm_Baselines.ipynb, then 02_Final_RQ1_Threshold_Decoding.ipynb
```

Run the notebooks in this order. Notebook 02 reads the PhoBERT probabilities that notebook 01 writes to
`results/predictions/midterm_phobert_{val,test}.csv`. Those two files are already in the submission, so notebook 02
also runs on its own.

Notebook 01 trains the three models and saves weights and probabilities in `code/runs/<model>/`. If a run with the
same configuration is already there, it loads it instead of training again (set `RETRAIN = True` to force training).
`code/runs/` and `code/data/` are not part of the submission because they hold model weights and raw data.
A full run from scratch took about 35 minutes on a Colab T4.

## Which notebook produces which result

Table and figure numbers refer to the final report (`report/62FIT4ATI-CLC02_G02_P02_report.docx`).

| Report item | Produced by | File |
|---|---|---|
| Table 2 (dataset statistics) | 01, Sections 2.1 and 2.4 | printed in the notebook |
| Table 3 (training configuration) | 01, Section 1 and Section 4 | `configs/tfidf_lr.json`, `configs/bilstm.json`, `configs/phobert.json` |
| Table 4 (baseline recap) | 01, Section 5 | `results/tables/midterm_main_metrics.csv` |
| Figure 1 (label frequency by tier) | 01, Section 2.3 | printed in the notebook |
| Figure 2 (label correlation) | 01, Section 2.5 | printed in the notebook |
| Figure 3 (uniform threshold sweep) | 01, Section 6, Q5 | printed in the notebook |
| Figure 8 (gold × predicted labels) | 01, Section 7.5 | printed in the notebook |
| Table 5 (RQ1 main results) | 02, Section 5 | `results/tables/rq1_table_main_metrics.csv` |
| Table 6 (validation-to-test gap, bootstrap) | 02, Sections 5 and 5.1 | `results/tables/rq1_table_val_test_gap.csv`, `rq1_table_bootstrap_ci.csv` |
| Table 7 (frequency tiers) | 02, Section 7 | `results/tables/rq1_table_tiers.csv` |
| Table 8 (errors fixed and added) | 02, Section 10.1 | `results/tables/rq1_table_error_transitions.csv` |
| Tables 9 and 10 (error audit) | 02, Section 10.2 | `results/tables/rq1_audit_fn.csv`, `rq1_audit_fp.csv` |
| Figure 4 (threshold profile) | 02, Section 9 | `results/figures/rq1_threshold_profile.png` |
| Figure 5 (per-label F1 by strategy) | 02, Section 6 | `results/figures/rq1_per_label_f1.png` |
| Figure 6 (tiers) | 02, Section 7 | `results/figures/rq1_tiers.png` |
| Figure 7 (label cardinality) | 02, Section 8 | `results/figures/rq1_cardinality.png` |

Other files in `results/`:

- `predictions/midterm_{tfidf_lr,bilstm,phobert}_{val,test}.csv`: one row per comment with `sample_id`,
  `true_label_vector`, `probability_vector` and `pred_fixed` (τ = 0.5).
- `predictions/rq1_predictions_{val,test}.csv` and `rq1_predictions.npz`: `sample_id`, `true_label_vector`,
  `probability_vector`, `pred_fixed`, `pred_global`, `pred_per_label`, `label_cardinality`.
- `tables/midterm_per_label_f1.csv`, `tables/rq1_table_per_label_f1.csv`, `rq1_table_cardinality.csv`,
  `rq1_table_threshold_profile.csv`, `rq1_table_confusions.csv`, `rq1_table_pair_recall.csv`,
  `rq1_audit_high_cardinality.csv`.
- `figures/rq1_f1_vs_threshold_per_label.png`: validation F1 against the threshold for each label.

Vectors are JSON lists of 28 values in the label order of `label_dict.json`.

The public GitHub repository leaves out `results/predictions/` and the three `results/tables/rq1_audit_*.csv`
files. They pair comment ids with gold labels, and the dataset licence forbids redistributing the dataset. They are
part of the course submission, and running the two notebooks recreates them.

## Reproducibility notes

- Seed 42 for Python, NumPy, PyTorch and CUDA; deterministic cuDNN. Details in `results/environment.md`.
- Model selection and every threshold use the validation split only. The test split is scored once per model and
  once per decoding strategy.
- Text pipeline: Unicode NFC, emoji to English names (`emoji.demojize`), letters repeated 3+ times cut to one and
  symbols cut to two, word segmentation with `underthesea.word_tokenize`. TF-IDF and BiLSTM lowercase the text;
  PhoBERT keeps case. Teencode is not normalised.
- Pretrained checkpoint: `vinai/phobert-base-v2`.
- Comment text is never printed or saved unless `SHOW_COMMENT_TEXT = True` in a notebook. Keep it `False`.
