"""Train TinySepsisCDEModel (Neural CDE encoder, Priority 2 of
regulatory/model_improvement_roadmap.md) on the identical data pipeline,
splits, and label definition as scripts/train_model.py, so its numbers
are directly comparable to the GRU's in results/tables.

No AMP/fp16 here: torchcde's ODE integration accumulates numerical error
across solver steps in a way a single GRU forward pass does not, and
fp16 risk is not worth the speed here for an exploratory architecture
comparison -- trains in fp32 throughout.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tinysepsis.data.dataset import TinySepsisDataset  # noqa: E402
from tinysepsis.models.tiny_sepsis_cde import TinySepsisCDEModel  # noqa: E402
from tinysepsis.eval.metrics import auroc, auprc  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "enriched.parquet"
CKPT_DIR = ROOT / "results" / "checkpoints"
PRED_DIR = ROOT / "results" / "predictions"


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            static = batch["static"].to(device)
            logits = model(seq, static)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch["label"].numpy())
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    return y_true, y_prob


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label-col", default="label_6h")
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--mlp-hidden", type=int, default=64)
    p.add_argument("--solver", default="rk4")
    p.add_argument("--solver-step-size", type=float, default=1.0)
    # Profiled empirically: batch=32 left the GPU at ~13% utilization (135 rows/s,
    # 0.14GB VRAM) -- fixed-step rk4 over T=24 is dominated by per-batch Python/kernel-
    # launch overhead, not compute, so small batches waste the GPU rather than protect
    # it. batch=512 hits 2959 rows/s (~22x) at 1.85GB, well under the 8GB budget.
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-train-patients", type=int, default=None)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--tag", default="tinysepsis_cde")
    p.add_argument("--ablation", default="full", choices=["full", "no_missingness", "no_dynamics"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, seed: {args.seed}", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print("Loading datasets...", flush=True)
    train_ds = TinySepsisDataset(DATA_PATH, "train", seq_len=args.seq_len, label_col=args.label_col,
                                  max_patients=args.max_train_patients, ablation=args.ablation)
    val_ds = TinySepsisDataset(DATA_PATH, "val", seq_len=args.seq_len, label_col=args.label_col, ablation=args.ablation)
    test_ds = TinySepsisDataset(DATA_PATH, "test", seq_len=args.seq_len, label_col=args.label_col, ablation=args.ablation)
    ext_ds = TinySepsisDataset(DATA_PATH, "external_test", seq_len=args.seq_len, label_col=args.label_col, ablation=args.ablation)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} external={len(ext_ds)}", flush=True)

    num_dynamic = len(train_ds.patients[next(iter(train_ds.patients))]["feats"][0])
    num_static = train_ds.patients[next(iter(train_ds.patients))]["static"].shape[1]

    train_labels = np.concatenate([train_ds.patients[pid]["labels"] for pid in train_ds.patients])
    pos_rate = train_labels.mean()
    pos_weight = torch.tensor([(1 - pos_rate) / max(pos_rate, 1e-6)], device=device)
    print(f"Train positive rate: {pos_rate:.4f}, pos_weight={pos_weight.item():.2f}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
    ext_loader = DataLoader(ext_ds, batch_size=128, shuffle=False, num_workers=0)

    model = TinySepsisCDEModel(
        num_dynamic_features=num_dynamic,
        num_static_features=num_static,
        hidden_size=args.hidden_size,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout,
        solver=args.solver,
        solver_options={"step_size": args.solver_step_size},
    ).to(device)
    n_params = model.num_parameters()
    print(f"Model parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auroc = -1.0
    epochs_no_improve = 0
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CKPT_DIR / f"{args.tag}_best.pt"

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            seq = batch["seq"].to(device)
            static = batch["static"].to(device)
            label = batch["label"].to(device)

            logits = model(seq, static)
            loss = criterion(logits, label) / args.grad_accum
            loss.backward()
            if (step + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * args.grad_accum

        scheduler.step()
        y_true, y_prob = evaluate(model, val_loader, device)
        val_auroc = auroc(y_true, y_prob)
        val_auprc = auprc(y_true, y_prob)
        elapsed = time.time() - t0
        print(f"epoch {epoch+1}/{args.epochs} loss={running_loss/len(train_loader):.4f} "
              f"val_auroc={val_auroc:.4f} val_auprc={val_auprc:.4f} ({elapsed:.0f}s)", flush=True)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(), "args": vars(args),
                        "num_dynamic": num_dynamic, "num_static": num_static}, best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch+1}", flush=True)
                break

    print(f"Best val AUROC: {best_val_auroc:.4f} -> {best_path}", flush=True)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for name, loader, ds in [("val", val_loader, val_ds), ("test", test_loader, test_ds),
                              ("external_test", ext_loader, ext_ds)]:
        y_true, y_prob = evaluate(model, loader, device)
        pids = [ds.index[i][0] for i in range(len(ds))]
        row_idxs = [ds.index[i][1] for i in range(len(ds))]
        iculos = [int(ds.patients[pid]["iculos"][row_idx]) for pid, row_idx in zip(pids, row_idxs)]
        out = pl.DataFrame({"patient_id": pids, "ICULOS": iculos, "y_true": y_true, "y_prob": y_prob})
        out.write_parquet(PRED_DIR / f"{args.tag}__{name}.parquet")
        print(f"{name}: AUROC={auroc(y_true, y_prob):.4f} AUPRC={auprc(y_true, y_prob):.4f} "
              f"-> saved {len(out)} predictions", flush=True)

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None
    with open(CKPT_DIR / f"{args.tag}_meta.json", "w") as f:
        json.dump({"n_params": n_params, "best_val_auroc": best_val_auroc,
                    "peak_vram_gb": peak_vram_gb, "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()
