# TinySepsis

**Calibrated, low-resource early warning for sepsis with a compact (<200K-parameter) temporal model.**

Research prototype. **Not a medical device. Not for clinical use.** Trained and evaluated entirely on the
open-access [PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/)
dataset — no PhysioNet account, CITI training, or data use agreement required.

See [`paper/main.pdf`](paper/main.pdf) for the full writeup (methods, results, limitations, ethics).

## Headline result

Trained on Hospital A (PhysioNet Challenge 2019) and evaluated, without any further tuning, on Hospital B
(a different institution never seen during development). Retrained under 5 random seeds each
(`scripts/run_multiseed.py`) to check the finding isn't a lucky initialization:

| Model | Internal AUROC | External AUROC | Gap (mean ± std, n=5) |
|---|---|---|---|
| XGBoost | 0.805 ± 0.003 | 0.686 ± 0.003 | 0.119 ± 0.003 |
| Logistic Regression | 0.770 ± 0.000 | 0.644 ± 0.000 | 0.126 ± 0.000 |
| LightGBM | 0.786 ± 0.002 | 0.670 ± 0.006 | 0.115 ± 0.007 |
| **TinySepsis (ours)** | 0.708 ± 0.010 | 0.650 ± 0.024 | **0.058 ± 0.030** |

The tabular baselines win internally in every seed but lose 0.115–0.126 AUROC on average on the external
hospital; TinySepsis loses less than half that (0.058), and its degradation was smaller than all 15
baseline-seed runs (Welch's t p ≤ 0.011, Mann-Whitney p = 0.004 against each baseline, Bonferroni-corrected
— see `results/tables/multiseed_stats.json`, reproducible via `scripts/multiseed_stats.py`). This is the
same internal-vs-external gap pattern documented for the Epic Sepsis Model in real-world deployment (Wong
et al., *JAMA Internal Medicine*, 2021) — reproduced here in miniature, under full experimental control.

**This is not the whole story.** At each model's own conformal-calibrated alarm threshold (the threshold
this project actually argues for, not the fixed-sensitivity threshold used for the table above), TinySepsis
is the *only* learned model whose clinical utility turns negative on the external hospital — its calibrated
false-alarm rate roughly triples there (19.8% vs. a 10% target) while XGBoost's and LightGBM's do not. AUROC
ranking transfers across hospitals better for TinySepsis; the calibrated decision threshold transfers worse.
Full accounting, including this caveat and the single-hospital-pair limitation, is in `paper/main.pdf`
(Sections 9, "Discussion", "Limitations").

## What this is

- A fully reproducible pipeline: raw `.psv` files → missingness-aware feature engineering → patient-level
  splits (train/val/test on Hospital A, Hospital B held out entirely as external validation) → model
  training → calibration (temperature scaling + isotonic regression) → conformal false-alarm-rate control
  → ONNX export → a local FastAPI research demo.
- A small GRU model (`TinySepsisModel`, <200K params) compared against qSOFA/NEWS2 clinical-score
  baselines, logistic regression, XGBoost, and LightGBM.
- Cross-hospital external validation (Hospital A → Hospital B) as a stand-in for the kind of validation
  whose absence contributed to the well-documented failure of the Epic Sepsis Model in external deployment
  (Wong et al., *JAMA Internal Medicine*, 2021).

## Requirements

- Windows/Linux/macOS, Python 3.11+ (developed on 3.13).
- ~2 GB disk for the raw+processed dataset, ~5 GB for the Python environment.
- A CUDA GPU is optional (tested on an 8 GB VRAM RTX 5060); CPU-only training works with smaller
  `--batch-size`/`--seq-len`.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate   # or `source .venv/bin/activate` on Linux/macOS
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` points at the CUDA 12.8 PyTorch wheel index; on CPU-only machines, remove the
`--extra-index-url` line first or install `torch` from PyPI directly.

## Reproduce everything, in order

```bash
python scripts/download_data.py        # ~255 MB, open access, no login
python scripts/ingest.py                # .psv -> data/processed/raw_long.parquet
python scripts/build_features.py        # -> data/processed/enriched.parquet + norm_stats.json
python scripts/train_baselines.py       # qSOFA, NEWS2, LogReg, XGBoost, LightGBM
python scripts/train_model.py           # TinySepsis GRU, primary 6h-horizon task
python scripts/calibrate_and_conformal.py
python scripts/export_onnx.py
python scripts/run_ablations.py         # missingness / sequence-length ablations
python scripts/evaluate.py              # consolidated tables + figures
python scripts/utility_at_conformal_threshold.py  # utility at each model's own conformal tau, not just fixed-sensitivity
python scripts/run_multiseed.py         # 5-seed retrain of TinySepsis + LogReg/XGBoost/LightGBM (~1-1.5h)
python scripts/multiseed_stats.py       # Welch/Mann-Whitney/Cohen's d on the multi-seed gaps
python scripts/generate_paper_tables.py # refresh paper/tables/*.tex from results
```

Then, to rebuild the PDF:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Run the tests

```bash
pytest tests/ -v
```

Tests cover feature engineering (missingness masks, forward-fill, time-since-last-measurement), label
construction and censoring (no post-suspicion leakage), patient-level splitting (no cross-split leakage),
metrics (AUROC/AUPRC/Brier/ECE/decision-curve/lead-time), calibration, conformal risk control, the utility
function, the model architecture, the PyTorch `Dataset`, and the demo API — all without requiring the full
dataset to be downloaded.

## Run the local demo

```bash
uvicorn tinysepsis.demo.app:app --port 8420
```

`POST /predict` with hourly observations returns a calibrated risk probability, a conformal alarm decision,
and the top contributing (most recently observed, highest-magnitude) feature channels. The response always
includes a "not for clinical use" disclaimer. No data leaves the machine.

## Repository layout

```
src/tinysepsis/
  data/       ingestion, feature engineering, labels, splits, normalization, PyTorch Dataset
  models/     TinySepsis GRU, clinical-score baselines
  eval/       metrics, calibration, conformal risk control, clinical utility function
  demo/       FastAPI research demo
scripts/      standalone, ordered pipeline scripts (see above)
tests/        pytest suite
paper/        LaTeX source, references.bib, generated tables/figures, main.pdf
results/      generated: predictions, tables, figures, calibration artifacts, checkpoints (gitignored)
data/         generated: raw + processed parquet (gitignored; regenerate via scripts/download_data.py)
```

## Data & licensing

- Code: MIT License (see `LICENSE`).
- Dataset: PhysioNet/CinC Challenge 2019, distributed under its own open license by PhysioNet; not
  redistributed in this repository. `scripts/download_data.py` fetches it directly from PhysioNet's public,
  unsigned S3 mirror (`s3://physionet-open/challenge-2019/1.0.0/`).

## Citation

If this pipeline or model is useful, please cite the accompanying paper (`paper/main.tex`) and the original
Challenge dataset:

> Reyna MA, Josef CS, Jeter R, Shashikumar SP, Westover MB, Nemati S, Clifford GD, Sharma A. Early Prediction
> of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge. *Critical Care Medicine*
> 48(2):210-217, 2019.

## Disclaimer

TinySepsis is a research artifact produced for methodological study of calibration, false-alarm control, and
low-resource deployability in clinical early-warning systems. It has not been prospectively validated, is not
FDA-cleared or CE-marked, and must not be used to make or influence real clinical decisions.
