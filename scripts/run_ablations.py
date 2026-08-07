"""Run the ablation suite: missingness representation, sequence length,
calibration. Each variant reuses train_model.py's training loop with a
different --tag so predictions land in separate files, then summarizes
val/test AUROC/AUPRC across variants into one table.
"""
import json
import subprocess
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from tinysepsis.eval.metrics import auroc, auprc  # noqa: E402

PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
PRED_DIR = ROOT / "results" / "predictions"
TABLE_DIR = ROOT / "results" / "tables"

VARIANTS = [
    {"tag": "ablation_full_seq24", "ablation": "full", "seq_len": 24, "epochs": 10},
    {"tag": "ablation_no_missingness_seq24", "ablation": "no_missingness", "seq_len": 24, "epochs": 10},
    {"tag": "ablation_full_seq12", "ablation": "full", "seq_len": 12, "epochs": 10},
]


def run_variant(v):
    cmd = [
        PY, str(ROOT / "scripts" / "train_model.py"),
        "--tag", v["tag"], "--ablation", v["ablation"],
        "--seq-len", str(v["seq_len"]), "--epochs", str(v["epochs"]),
        "--patience", "3",
    ]
    print(f"Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    print(result.stdout[-3000:], flush=True)
    if result.returncode != 0:
        print(result.stderr[-3000:], flush=True)
        raise RuntimeError(f"variant {v['tag']} failed")


def main():
    rows = []
    for v in VARIANTS:
        run_variant(v)
        for split in ["val", "test", "external_test"]:
            path = PRED_DIR / f"{v['tag']}__{split}.parquet"
            if not path.exists():
                continue
            pred = pl.read_parquet(path)
            y, p = pred["y_true"].to_numpy(), pred["y_prob"].to_numpy()
            rows.append({
                "variant": v["tag"], "ablation": v["ablation"], "seq_len": v["seq_len"],
                "split": split, "auroc": auroc(y, p), "auprc": auprc(y, p), "n": len(y),
            })

    df = pl.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.write_csv(TABLE_DIR / "ablations.csv")
    print(df, flush=True)


if __name__ == "__main__":
    main()
