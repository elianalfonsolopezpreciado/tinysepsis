"""Ensemble evaluation using ONLY predictions already on disk (5-seed GRU,
attention, and CDE runs from today's model-improvement pass) -- no
retraining. Tests whether simple probability-averaging across seeds and/or
architectures improves external AUROC or the cross-institution gap, since
this is a genuinely different lever from anything tried in
regulatory/model_improvement_roadmap.md so far (hyperparameters,
architecture swap, longer training all individually failed or traded off).

Significance is via patient-level bootstrap (resample patients with
replacement, 1000 resamples) rather than a multi-seed comparison, since an
ensemble of 5 seeds is a single deterministic aggregate, not 5 independent
draws -- there is nothing else to re-seed.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tinysepsis.eval.metrics import auroc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "results" / "predictions"
TABLE_DIR = ROOT / "results" / "tables"
SEEDS = [0, 1, 2, 3, 4]
N_BOOTSTRAP = 300
RNG_SEED = 0


def load_probs(tag_fn, split):
    frames = []
    for seed in SEEDS:
        tag = tag_fn(seed)
        path = PRED_DIR / f"{tag}__{split}.parquet"
        df = pl.read_parquet(path).sort(["patient_id", "ICULOS"])
        frames.append(df)
    key = frames[0].select(["patient_id", "ICULOS", "y_true"])
    probs = np.stack([f["y_prob"].to_numpy() for f in frames], axis=1)  # (n_rows, n_seeds)
    return key, probs


def bootstrap_auroc_diff(y_true, prob_a, prob_b, patient_ids, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    """Row-level bootstrap (not patient-clustered): resample rows directly
    with replacement, compute AUROC(a) - AUROC(b) each time. A patient-
    level cluster bootstrap would be more rigorous (this project's other
    multi-seed comparisons use real independent seeds, not bootstrap, for
    exactly that reason) but is prohibitively slow at this row count
    (~117K-758K rows); row-level resampling ignores within-patient
    correlation and therefore likely UNDERSTATES the true confidence
    interval width -- a significant result here should be read as
    suggestive, not as strong evidence as the seed-based tests elsewhere
    in this project."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            diffs[i] = np.nan
            continue
        diffs[i] = auroc(yt, prob_a[idx]) - auroc(yt, prob_b[idx])
    return diffs[~np.isnan(diffs)]


def evaluate_variant(name, key, probs_dict, splits_probs):
    """splits_probs: dict split_name -> (key, ensemble_prob, single_ref_prob)"""
    row = {"variant": name}
    for split, (k, ens_p, ref_p) in splits_probs.items():
        y = k["y_true"].to_numpy()
        row[f"{split}_ensemble_auroc"] = auroc(y, ens_p)
        row[f"{split}_single_seed0_auroc"] = auroc(y, ref_p)
    return row


def main():
    results = []
    boot_results = {}

    for split in ["test", "external_test"]:
        gru_key, gru_probs = load_probs(lambda s: f"tinysepsis_seed{s}", split)
        attn_key, attn_probs = load_probs(lambda s: f"tinysepsis_attn_seed{s}", split)
        cde_key, cde_probs = load_probs(lambda s: f"cde_seed{s}", split)
        mh_key, mh_probs = load_probs(lambda s: f"tinysepsis_multihorizon_seed{s}", split)

        assert gru_key.equals(attn_key) and gru_key.equals(cde_key) and gru_key.equals(mh_key), "row key mismatch across architectures"
        y_true = gru_key["y_true"].to_numpy()
        patient_ids = gru_key["patient_id"].to_numpy()

        variants = {
            "gru_seed0_single": gru_probs[:, 0],
            "gru_5seed_ensemble": gru_probs.mean(axis=1),
            "gru_attn_10model_ensemble": np.concatenate([gru_probs, attn_probs], axis=1).mean(axis=1),
            "gru_attn_cde_15model_ensemble": np.concatenate([gru_probs, attn_probs, cde_probs], axis=1).mean(axis=1),
            "gru_attn_cde_multihorizon_20model_ensemble": np.concatenate([gru_probs, attn_probs, cde_probs, mh_probs], axis=1).mean(axis=1),
        }

        row = {"split": split}
        for name, p in variants.items():
            row[f"{name}_auroc"] = auroc(y_true, p)
        results.append(row)
        print(f"{split}: " + ", ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "split"), flush=True)

        # Bootstrap: does the 5-seed GRU ensemble beat a single GRU seed?
        diffs = bootstrap_auroc_diff(y_true, variants["gru_5seed_ensemble"], variants["gru_seed0_single"], patient_ids)
        boot_results[f"{split}_5seed_ensemble_vs_single"] = {
            "mean_diff": float(diffs.mean()), "ci_2.5": float(np.percentile(diffs, 2.5)),
            "ci_97.5": float(np.percentile(diffs, 97.5)), "pct_positive": float((diffs > 0).mean()),
        }
        # Does the cross-architecture 15-model ensemble beat the GRU-only 5-seed ensemble?
        diffs2 = bootstrap_auroc_diff(y_true, variants["gru_attn_cde_15model_ensemble"], variants["gru_5seed_ensemble"], patient_ids)
        boot_results[f"{split}_15model_vs_5seed_gru"] = {
            "mean_diff": float(diffs2.mean()), "ci_2.5": float(np.percentile(diffs2, 2.5)),
            "ci_97.5": float(np.percentile(diffs2, 97.5)), "pct_positive": float((diffs2 > 0).mean()),
        }
        # Does adding the multi-horizon seeds to the 15-model ensemble help further?
        diffs3 = bootstrap_auroc_diff(y_true, variants["gru_attn_cde_multihorizon_20model_ensemble"], variants["gru_attn_cde_15model_ensemble"], patient_ids)
        boot_results[f"{split}_20model_vs_15model"] = {
            "mean_diff": float(diffs3.mean()), "ci_2.5": float(np.percentile(diffs3, 2.5)),
            "ci_97.5": float(np.percentile(diffs3, 97.5)), "pct_positive": float((diffs3 > 0).mean()),
        }

    gap_rows = {}
    for r in results:
        gap_rows[r["split"]] = r
    gap_summary = []
    for name in ["gru_seed0_single_auroc", "gru_5seed_ensemble_auroc", "gru_attn_10model_ensemble_auroc",
                 "gru_attn_cde_15model_ensemble_auroc", "gru_attn_cde_multihorizon_20model_ensemble_auroc"]:
        test_v = gap_rows["test"][name]
        ext_v = gap_rows["external_test"][name]
        gap_summary.append({"variant": name.replace("_auroc", ""), "test_auroc": test_v, "external_auroc": ext_v, "gap": test_v - ext_v})

    out_df = pl.DataFrame(gap_summary)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out_df.write_csv(TABLE_DIR / "ensemble_comparison.csv")
    print(out_df, flush=True)

    with open(TABLE_DIR / "ensemble_bootstrap_stats.json", "w") as f:
        json.dump(boot_results, f, indent=2)
    print(json.dumps(boot_results, indent=2), flush=True)
    print("Ensemble evaluation complete.", flush=True)


if __name__ == "__main__":
    main()
