"""Multi-horizon auxiliary-task training: predict 4h, 6h, and 8h sepsis
risk jointly from one shared GRU encoder (TinySepsisModel(output_dim=3)),
instead of only the primary 6h task train_model.py trains. This is a
genuinely untried lever from today's model-improvement pass, and a cheap
one: label_4h/label_6h/label_8h are already computed by
tinysepsis.data.labels.add_early_warning_labels for every row in
enriched.parquet, so this needs zero new data, only a wider output head
and a combined loss. Rationale: the three horizons share almost all of
their signal (they differ only in how close to t_susp the positive window
starts), so training on all three is a soft form of the kind of auxiliary-
task regularization that sometimes reduces overfitting to any single
horizon's idiosyncratic label noise -- an open question this script tests
empirically, not a guaranteed win.

Primary evaluation is still on the 6h task only (logits[:, 1]), so results
are directly comparable to every other number in results/tables/multiseed_*.
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
from tinysepsis.models.tiny_sepsis import TinySepsisModel  # noqa: E402
from tinysepsis.eval.metrics import auroc, auprc  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "enriched.parquet"
CKPT_DIR = ROOT / "results" / "checkpoints"
PRED_DIR = ROOT / "results" / "predictions"
HORIZONS = ["label_4h", "label_6h", "label_8h"]  # output order; index 1 (6h) is the primary task


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            static = batch["static"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(seq, static)  # (B, 3)
            probs_6h = torch.sigmoid(logits[:, 1].float()).cpu().numpy()
            all_probs.append(probs_6h)
            all_labels.append(batch["label"].numpy())  # label_6h, the primary task
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    return y_true, y_prob


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-train-patients", type=int, default=None)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--tag", default="tinysepsis_multihorizon")
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
    ds_kwargs = dict(seq_len=args.seq_len, label_col="label_6h", ablation=args.ablation,
                      extra_label_cols=["label_4h", "label_8h"])
    train_ds = TinySepsisDataset(DATA_PATH, "train", max_patients=args.max_train_patients, **ds_kwargs)
    val_ds = TinySepsisDataset(DATA_PATH, "val", **ds_kwargs)
    test_ds = TinySepsisDataset(DATA_PATH, "test", **ds_kwargs)
    ext_ds = TinySepsisDataset(DATA_PATH, "external_test", **ds_kwargs)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} external={len(ext_ds)}", flush=True)

    num_dynamic = len(train_ds.patients[next(iter(train_ds.patients))]["feats"][0])
    num_static = train_ds.patients[next(iter(train_ds.patients))]["static"].shape[1]

    def target_tensor(batch):
        # order must match HORIZONS: [4h, 6h, 8h]
        return torch.stack([batch["extra_labels"][:, 0], batch["label"], batch["extra_labels"][:, 1]], dim=1)

    pos_weights = []
    for col in HORIZONS:
        rate = np.concatenate([
            pl.scan_parquet(DATA_PATH).filter((pl.col("split") == "train")).select(col).collect()[col].to_numpy()
        ]).mean()
        pos_weights.append((1 - rate) / max(rate, 1e-6))
    pos_weight = torch.tensor(pos_weights, dtype=torch.float32, device=device)
    print(f"Per-horizon positive rates -> pos_weight: {dict(zip(HORIZONS, pos_weights))}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    ext_loader = DataLoader(ext_ds, batch_size=256, shuffle=False, num_workers=0)

    model = TinySepsisModel(
        num_dynamic_features=num_dynamic, num_static_features=num_static,
        hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout,
        output_dim=3,
    ).to(device)
    n_params = model.num_parameters()
    print(f"Model parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

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
            target = target_tensor(batch).to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(seq, static)
                loss = criterion(logits, target) / args.grad_accum

            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * args.grad_accum

        scheduler.step()
        y_true, y_prob = evaluate(model, val_loader, device)
        val_auroc = auroc(y_true, y_prob)
        val_auprc = auprc(y_true, y_prob)
        elapsed = time.time() - t0
        print(f"epoch {epoch+1}/{args.epochs} loss={running_loss/len(train_loader):.4f} "
              f"val_auroc_6h={val_auroc:.4f} val_auprc_6h={val_auprc:.4f} ({elapsed:.0f}s)", flush=True)

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

    print(f"Best val AUROC (6h): {best_val_auroc:.4f} -> {best_path}", flush=True)

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
        json.dump({"n_params": n_params, "best_val_auroc_6h": best_val_auroc,
                    "peak_vram_gb": peak_vram_gb, "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()
