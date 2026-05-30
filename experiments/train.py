"""Minimal training loop for experiments.

Usage (quick smoke/trial):

```bash
.venv/bin/python experiments/train.py --processed data/processed --epochs 1 --batch-size 8
```
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Local imports are inside main() to allow this script to be run without installing the project dependencies for quick smoke tests.
def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--processed", type=str, default="data/processed")
    p.add_argument("--signals-key", type=str, default="ecg_train.npy")
    p.add_argument("--labels-key", type=str, default="tabular_train_y.npy")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--augment", action="store_true", help="Enable augmentations during training")
    p.add_argument("--random-crop-len", type=int, default=2000)
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--scale-min", type=float, default=1.0)
    p.add_argument("--scale-max", type=float, default=1.0)
    p.add_argument("--val-frac", type=float, default=0.2, help="Fraction of data for validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--loss", type=str, default="bce", choices=["bce", "weighted_bce", "focal"], help="Training loss")
    p.add_argument("--focal-gamma", type=float, default=2.0, help="Gamma for focal loss")
    p.add_argument("--dropout-p", type=float, default=0.0, help="Dropout probability in the classifier head")
    p.add_argument("--mc-samples", type=int, default=0, help="Number of MC-dropout validation samples (0 disables uncertainty estimation)")
    p.add_argument("--save-uncertainty", action="store_true", help="Save validation uncertainty outputs to npz")
    p.add_argument("--uncertainty-dir", type=str, default="experiments/checkpoints", help="Directory for uncertainty outputs")
    return p.parse_args()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1 - prob) * (1 - targets)
        loss = ((1 - pt) ** self.gamma) * bce
        return loss.mean()


def compute_pos_weight(labels):
    labels = np.asarray(labels).astype(int)
    pos = max(1, int(labels.sum()))
    neg = max(1, int(len(labels) - pos))
    return torch.tensor([neg / pos], dtype=torch.float32)


def enable_mc_dropout(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
        else:
            module.eval()

# The main training loop is defined in main() and includes data loading, model building, training, and checkpoint saving.
def main():
    args = build_args()
    device = torch.device(args.device)

    # local imports
    from experiments.data import ECGDataset, collate_fn
    from experiments.models.resnet1d import build_resnet1d
    # optional augmentations: dataset will handle per-worker RNG
    transform = None
    augment_params = None
    if args.augment:
        augment_params = {
            'random_crop_len': args.random_crop_len,
            'noise_std': args.noise_std,
            'scale_min': args.scale_min,
            'scale_max': args.scale_max,
        }

    ds = ECGDataset(
        args.processed,
        signals_key=args.signals_key,
        labels_key=args.labels_key,
        transform=transform,
        augment_params=augment_params,
        base_seed=args.seed,
    )
    # split into train/val
    n = len(ds)
    val_n = max(1, int(n * args.val_frac))
    train_n = n - val_n
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    if val_n <= 0:
        train_ds = ds
        val_ds = None
    else:
        train_ds, val_ds = random_split(ds, [train_n, val_n], generator=generator)

    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    vdl = None
    if val_ds is not None:
        vdl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # build model
    model = build_resnet1d(in_channels=1, num_classes=1, dropout_p=args.dropout_p)
    model = model.to(device)
    pos_weight = compute_pos_weight(ds.y) if args.loss in {"weighted_bce", "focal"} else None
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)

    if args.loss == "weighted_bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif args.loss == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma, pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    try:
        from torch.utils.tensorboard.writer import SummaryWriter
        writer = SummaryWriter(log_dir='experiments/runs')
    except Exception:
        writer = None

    print(f"Starting training on {device} for {args.epochs} epochs, {len(dl)} batches per epoch")
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for xb, yb in dl:
            xb = xb.to(device).float()
            yb = yb.to(device).float().unsqueeze(1)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs} loss={running_loss/len(dl):.4f} time={(time.time()-t0):.1f}s")

        # validation
        if vdl is not None:
            model.eval()
            ys = []
            yps = []
            with torch.no_grad():
                for xb, yb in vdl:
                    xb = xb.to(device).float()
                    out = model(xb)
                    prob = torch.sigmoid(out).cpu().numpy().ravel()
                    ys.append(yb.numpy().ravel())
                    yps.append(prob)
            import numpy as _np
            ys = _np.concatenate(ys)
            yps = _np.concatenate(yps)
            # compute metrics
            try:
                from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, accuracy_score
                ypred = (yps >= 0.5).astype(int)
                roc = float(roc_auc_score(ys, yps)) if len(_np.unique(ys)) > 1 else float('nan')
                ap = float(average_precision_score(ys, yps)) if len(_np.unique(ys)) > 1 else float('nan')
                brier = float(brier_score_loss(ys, yps))
                acc = float(accuracy_score(ys, ypred))
                print(f"VAL metrics: acc={acc:.3f} roc_auc={roc:.3f} ap={ap:.3f} brier={brier:.4f}")
                # save metrics
                os.makedirs('experiments/checkpoints', exist_ok=True)
                with open('experiments/checkpoints/last_metrics.json', 'w') as fh:
                    json.dump({"epoch": epoch+1, "acc": acc, "roc_auc": roc, "ap": ap, "brier": brier}, fh)
                # determine metric to use (prefer ROC if available)
                try:
                    metric_val = roc if (not (_np.isnan(roc))) else ap
                except Exception:
                    metric_val = ap
                # persist best checkpoint if improved
                best_path = 'experiments/checkpoints/best_by_metric.txt'
                prev_best = None
                if os.path.exists(best_path):
                    try:
                        with open(best_path, 'r') as fh:
                            prev_best = float(fh.read().strip())
                    except Exception:
                        prev_best = None
                if prev_best is None or (not _np.isnan(metric_val) and metric_val > prev_best):
                    best_ckpt = {"model_state_dict": model.state_dict(), "args": vars(args)}
                    torch.save(best_ckpt, f"experiments/checkpoints/best_metric_epoch{epoch+1}.pt")
                    with open(best_path, 'w') as fh:
                        fh.write(str(metric_val))
                    print(f"New best metric {metric_val:.4f} -> saved best checkpoint")
                if writer is not None:
                    writer.add_scalar('val/acc', acc, epoch+1)
                    writer.add_scalar('val/roc_auc', 0.0 if _np.isnan(roc) else roc, epoch+1)
                    writer.add_scalar('val/ap', ap, epoch+1)
                    writer.add_scalar('val/brier', brier, epoch+1)

                if args.mc_samples and args.mc_samples > 0:
                    enable_mc_dropout(model)
                    mc_probs = []
                    with torch.no_grad():
                        for _ in range(args.mc_samples):
                            sample_probs = []
                            for xb, _ in vdl:
                                xb = xb.to(device).float()
                                logits = model(xb)
                                sample_probs.append(torch.sigmoid(logits).cpu().numpy().ravel())
                            mc_probs.append(np.concatenate(sample_probs))
                    mc_probs = np.stack(mc_probs, axis=0)
                    mc_mean = mc_probs.mean(axis=0)
                    mc_std = mc_probs.std(axis=0)
                    mc_entropy = -(mc_mean * np.log(mc_mean + 1e-8) + (1 - mc_mean) * np.log(1 - mc_mean + 1e-8))
                    if args.save_uncertainty:
                        os.makedirs(args.uncertainty_dir, exist_ok=True)
                        np.savez(
                            os.path.join(args.uncertainty_dir, f"val_uncertainty_epoch{epoch+1}.npz"),
                            y_true=ys,
                            mc_probs=mc_probs,
                            mc_mean=mc_mean,
                            mc_std=mc_std,
                            mc_entropy=mc_entropy,
                        )
                        with open(os.path.join(args.uncertainty_dir, f"val_uncertainty_epoch{epoch+1}.json"), "w") as fh:
                            json.dump(
                                {
                                    "epoch": epoch + 1,
                                    "mc_samples": args.mc_samples,
                                    "mean_std": float(mc_std.mean()),
                                    "mean_entropy": float(mc_entropy.mean()),
                                },
                                fh,
                            )
                    if writer is not None:
                        writer.add_scalar('val/mc_std_mean', float(mc_std.mean()), epoch+1)
                        writer.add_scalar('val/mc_entropy_mean', float(mc_entropy.mean()), epoch+1)
            except Exception as e:
                print(f"[WARN] Could not compute sklearn metrics: {e}")

    # save a small checkpoint
    ckpt = {"model_state_dict": model.state_dict(), "args": vars(args)}
    os.makedirs("experiments/checkpoints", exist_ok=True)
    torch.save(ckpt, "experiments/checkpoints/last.pt")
    print("Saved checkpoint to experiments/checkpoints/last.pt")


if __name__ == '__main__':
    main()
