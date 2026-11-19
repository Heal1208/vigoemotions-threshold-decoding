# Environment

Everything below is taken from the printed output of the two notebooks in `notebooks/`.

## Hardware

| Notebook | Where it ran | Device |
|---|---|---|
| `01_Midterm_Baselines.ipynb` | Google Colab | NVIDIA Tesla T4 (CUDA, mixed precision on) |
| `02_Final_RQ1_Threshold_Decoding.ipynb` | Google Colab | CPU is enough; it only reads saved probabilities |

## Software

| Package | Version |
|---|---|
| torch | 2.14.1+cu130 |
| transformers | 5.22.1 |
| scikit-learn | 1.6.1 |
| scipy | 1.16.3 |
| numpy | 2.4.6 |
| pandas | 2.2.3 |
| underthesea | 9.5.0 |
| emoji | 2.16.0 |

The notebooks do not print the Python or matplotlib version, so we do not list them here.
Python 3 is whatever Colab provided on the day of the run.

## Pretrained checkpoint

`vinai/phobert-base-v2`, downloaded from the Hugging Face Hub by `transformers`.

## Seeds

Seed 42 everywhere:

- Python `random`, NumPy, PyTorch and CUDA (`seed_everything` in notebook 01), with
  `torch.backends.cudnn.deterministic = True` and `benchmark = False`;
- the paired bootstrap (1,000 resamples) in notebook 01 (Q4) and notebook 02 (Section 5.1);
- the random choice of the 20 false negatives and 20 false positives in the RQ1 audit.

Each model was trained once with this seed. We did not run extra seeds.

## Wall-clock time (Colab T4)

| Step | Time |
|---|---|
| Text normalisation + word segmentation (all splits) | 46 s |
| TF-IDF + OvR LR, selected configuration / whole grid of 8 | 0.6 min / 3.5 min |
| BiLSTM, 17 epochs (best epoch 14) | 2.2 min |
| PhoBERT-base-v2, selected run (lr 3e-5) / both learning rates | 13.7 min / 27.9 min |

One PhoBERT epoch took about 82 s on the T4 (517 batches of 32).
