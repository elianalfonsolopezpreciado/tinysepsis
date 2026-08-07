"""Regenerate paper/tables/*.tex from results/tables/*.csv and results/calibration/*.json
so the LaTeX paper always reflects the latest experiment run.
"""
import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = ROOT / "results" / "tables"
CAL_DIR = ROOT / "results" / "calibration"
CKPT_DIR = ROOT / "results" / "checkpoints"
OUT_DIR = ROOT / "paper" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    "qsofa": "qSOFA (lite)",
    "news2": "NEWS2 (lite)",
    "logreg": "Logistic Regression",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "tinysepsis": r"\textbf{TinySepsis (ours)}",
}
MODEL_ORDER = ["qsofa", "news2", "logreg", "xgboost", "lightgbm", "tinysepsis"]


def fmt(x, decimals=3):
    if x is None:
        return "--"
    try:
        if isinstance(x, float) and (x != x):  # NaN
            return "--"
        return f"{x:.{decimals}f}"
    except (TypeError, ValueError):
        return "--"


def write_results_table(df: pl.DataFrame, split: str, out_path: Path):
    rows = []
    for model in MODEL_ORDER:
        sub = df.filter((pl.col("model") == model) & (pl.col("split") == split))
        if sub.height == 0:
            rows.append((MODEL_LABELS[model], "--", "--", "--", "--", "--", "--"))
            continue
        r = sub.to_dicts()[0]
        rows.append((
            MODEL_LABELS[model],
            fmt(r.get("auroc")), fmt(r.get("auprc")), fmt(r.get("brier")),
            fmt(r.get("ece")), fmt(r.get("alarms_per_1000h"), 1),
            fmt(r.get("normalized_utility")),
        ))

    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & AUROC & AUPRC & Brier & ECE & Alarms/1000h & Norm.\ Utility \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


def write_efficiency_table():
    latency_path = CKPT_DIR / "latency_benchmark.json"
    if not latency_path.exists():
        print("skip efficiency table: no latency benchmark yet")
        return
    d = json.loads(latency_path.read_text())
    lines = [
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        rf"Parameters & {d['n_parameters']:,} \\",
        rf"ONNX model size (MB) & {d['onnx_file_size_mb']:.2f} \\",
        rf"PyTorch CPU latency (ms/sample) & {d['pytorch_cpu_latency_ms']:.3f} \\",
        rf"ONNX Runtime CPU latency (ms/sample) & {d['onnxruntime_cpu_latency_ms']:.3f} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (OUT_DIR / "efficiency.tex").write_text("\n".join(lines) + "\n")
    print("wrote efficiency.tex")


def write_calibration_table():
    metrics_path = CAL_DIR / "calibration_metrics.json"
    conformal_path = CAL_DIR / "conformal_results.json"
    if not metrics_path.exists() or not conformal_path.exists():
        print("skip calibration table: no calibration results yet")
        return
    metrics = json.loads(metrics_path.read_text())
    conformal = json.loads(conformal_path.read_text())

    split_labels = {"val": "Validation", "test": "Test (internal)", "external_test": "External (hosp.\\ B)"}
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Split & Raw ECE & Temp.-scaled ECE & Isotonic ECE & Conformal $\tau$ ($\alpha{=}0.10$) \\",
        r"\midrule",
    ]
    for split, label in split_labels.items():
        m = metrics.get(split, {})
        c = conformal.get(split, {})
        lines.append(
            f"{label} & {fmt(m.get('raw', {}).get('ece'))} & {fmt(m.get('temperature', {}).get('ece'))} "
            f"& {fmt(m.get('isotonic', {}).get('ece'))} & {fmt(c.get('tau'))} " r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT_DIR / "calibration.tex").write_text("\n".join(lines) + "\n")
    print("wrote calibration.tex")


def write_ablations_table():
    path = TABLE_DIR / "ablations.csv"
    if not path.exists():
        print("skip ablations table: no ablation results yet")
        return
    df = pl.read_csv(path)
    label_map = {
        "ablation_full_seq24": ("Full (mask+tslm+delta)", 24),
        "ablation_no_missingness_seq24": ("No missingness encoding", 24),
        "ablation_full_seq12": ("Full, short sequence", 12),
    }
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Variant & Seq.\ len. & Test AUROC & External AUROC \\",
        r"\midrule",
    ]
    for variant, (label, seqlen) in label_map.items():
        test_row = df.filter((pl.col("variant") == variant) & (pl.col("split") == "test"))
        ext_row = df.filter((pl.col("variant") == variant) & (pl.col("split") == "external_test"))
        test_auroc = fmt(test_row["auroc"][0]) if test_row.height else "--"
        ext_auroc = fmt(ext_row["auroc"][0]) if ext_row.height else "--"
        lines.append(f"{label} & {seqlen} & {test_auroc} & {ext_auroc} " r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT_DIR / "ablations.tex").write_text("\n".join(lines) + "\n")
    print("wrote ablations.tex")


def main():
    results_path = TABLE_DIR / "main_results.parquet"
    if results_path.exists():
        df = pl.read_parquet(results_path)
        write_results_table(df, "test", OUT_DIR / "main_results.tex")
        write_results_table(df, "external_test", OUT_DIR / "external_results.tex")
    else:
        print("skip main/external tables: no main_results.parquet yet")

    write_efficiency_table()
    write_calibration_table()
    write_ablations_table()


if __name__ == "__main__":
    main()
