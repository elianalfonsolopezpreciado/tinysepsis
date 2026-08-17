"""Train TinySepsisDANNModel: unsupervised domain-adversarial adaptation
using Hospital A's labeled train split (source) and Hospital B's UNLABELED
external_test split (target, features only -- never its sepsis labels).
See src/tinysepsis/models/tiny_sepsis_dann.py's module docstring for the
full methodological caveat: evaluating the result on Hospital B afterward
is legitimate transductive domain adaptation, not the blind holdout every
other number in this project reports, because Hospital B's *features* were
seen during training here.

Lambda (the gradient-reversal strength) follows the standard schedule from
Ganin & Lempitsky (2015): starts at 0 (pure supervised warm-up on the task
loss) and ramps to lambda_max as training progresses, since an adversarial
signal applied too early/strongly destabilizes a not-yet-useful encoder.
"""
import argparse
import itertools
import json
import math
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
from tinysepsis.models.tiny_sepsis_dann import TinySepsisDANNModel  # noqa: E402
from tinysepsis.eval.metrics import auroc, auprc  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "enriched.parquet"
CKPT_DIR = ROOT / "results" / "checkpoints"
PRED_DIR = ROOT / "results" / "predictions"


def evaluate_task(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            static = batch["static"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                task_logits, _ = model(seq, static, lambd=0.0)
            probs = torch.sigmoid(task_logits.float()).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch["label"].numpy())
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    return y_true, y_prob


def lambda_schedule(progress: float, gamma: float = 10.0, lambd_max: float = 1.0) -> float:
    return lambd_max * (2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0)


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
    p.add_argument("--lambda-max", type=float, default=1.0)
    p.add_argument("--max-train-patients", type=int, default=None)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--tag", default="tinysepsis_dann")
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

    print("Loading datasets (source=Hospital A train, target=Hospital B external_test, unlabeled)...", flush=True)
    source_ds = TinySepsisDataset(DATA_PATH, "train", seq_len=args.seq_len, max_patients=args.max_train_patients)
    target_ds = TinySepsisDataset(DATA_PATH, "external_test", seq_len=args.seq_len)
    val_ds = TinySepsisDataset(DATA_PATH, "val", seq_len=args.seq_len)
    test_ds = TinySepsisDataset(DATA_PATH, "test", seq_len=args.seq_len)
    print(f"source(train)={len(source_ds)} target(external, unlabeled)={len(target_ds)} "
          f"val={len(val_ds)} test={len(test_ds)}", flush=True)

    num_dynamic = len(source_ds.patients[next(iter(source_ds.patients))]["feats"][0])
    num_static = source_ds.patients[next(iter(source_ds.patients))]["static"].shape[1]

    train_labels = np.concatenate([source_ds.patients[pid]["labels"] for pid in source_ds.patients])
    pos_rate = train_labels.mean()
    pos_weight = torch.tensor([(1 - pos_rate) / max(pos_rate, 1e-6)], device=device)
    print(f"Source positive rate: {pos_rate:.4f}, pos_weight={pos_weight.item():.2f}", flush=True)

    # pin_memory disabled: with two loaders (source+target) materializing
    # page-locked host buffers concurrently on top of this project's larger
    # datasets (target=external_test has 758K rows fully in memory already),
    # pinned-allocation pressure can throw a CUDA OOM before any GPU compute
    # even starts -- not worth the modest speedup here.
    source_loader = DataLoader(source_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    target_loader = DataLoader(target_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    ext_loader_eval = DataLoader(target_ds, batch_size=256, shuffle=False, num_workers=0)

    model = TinySepsisDANNModel(
        num_dynamic_features=num_dynamic, num_static_features=num_static,
        hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout,
    ).to(device)
    n_params = model.num_parameters()
    print(f"Model parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    task_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    domain_criterion = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    steps_per_epoch = len(source_loader)
    total_steps = steps_per_epoch * args.epochs
    global_step = 0

    best_val_auroc = -1.0
    epochs_no_improve = 0
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CKPT_DIR / f"{args.tag}_best.pt"

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_task_loss, running_domain_loss, running_domain_correct, running_domain_n = 0.0, 0.0, 0, 0
        optimizer.zero_grad()
        target_iter = itertools.cycle(target_loader)

        for step, src_batch in enumerate(source_loader):
            progress = global_step / max(total_steps, 1)
            lambd = lambda_schedule(progress, lambd_max=args.lambda_max)
            global_step += 1

            tgt_batch = next(target_iter)
            src_seq, src_static, src_label = src_batch["seq"].to(device), src_batch["static"].to(device), src_batch["label"].to(device)
            tgt_seq, tgt_static = tgt_batch["seq"].to(device), tgt_batch["static"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                task_logits, src_domain_logits = model(src_seq, src_static, lambd=lambd)
                _, tgt_domain_logits = model(tgt_seq, tgt_static, lambd=lambd)

                task_loss = task_criterion(task_logits, src_label)
                domain_labels = torch.cat([
                    torch.zeros(src_domain_logits.shape[0], device=device),
                    torch.ones(tgt_domain_logits.shape[0], device=device),
                ])
                domain_logits_all = torch.cat([src_domain_logits, tgt_domain_logits])
                domain_loss = domain_criterion(domain_logits_all, domain_labels)
                loss = (task_loss + domain_loss) / args.grad_accum

            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_task_loss += task_loss.item()
            running_domain_loss += domain_loss.item()
            with torch.no_grad():
                domain_pred = (torch.sigmoid(domain_logits_all.float()) > 0.5).float()
                running_domain_correct += (domain_pred == domain_labels).sum().item()
                running_domain_n += domain_labels.shape[0]

        scheduler.step()
        y_true, y_prob = evaluate_task(model, val_loader, device)
        val_auroc = auroc(y_true, y_prob)
        val_auprc = auprc(y_true, y_prob)
        domain_acc = running_domain_correct / max(running_domain_n, 1)
        elapsed = time.time() - t0
        print(f"epoch {epoch+1}/{args.epochs} lambd={lambd:.3f} task_loss={running_task_loss/steps_per_epoch:.4f} "
              f"domain_loss={running_domain_loss/steps_per_epoch:.4f} domain_acc={domain_acc:.3f} "
              f"(0.5=fully invariant, 1.0=fully separable) val_auroc={val_auroc:.4f} val_auprc={val_auprc:.4f} ({elapsed:.0f}s)",
              flush=True)

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
                              ("external_test", ext_loader_eval, target_ds)]:
        y_true, y_prob = evaluate_task(model, loader, device)
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
