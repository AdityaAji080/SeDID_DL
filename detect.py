"""
Step 3 — detect diffusion-generated images from SeDID latent pairs.

Two detectors, as in the SeDID paper:
  SeDIDStat : score = -‖z_t − z'_t‖₂ ; one threshold, fitted on the training split.
  SeDIDNNs  : ResNet-18 trained on |z_t − z'_t| (a C×h×w "error map").

Data is split once, stratified, into train / val / test. SeDIDNNs picks its best epoch on
val; both detectors are reported on the untouched test split.

Usage:
    python detect.py --real_latents latents/real --fake_latents latents/fake --step 2 \
                     --output_dir outputs/sd15_step2
or point at the four directories explicitly:
    python detect.py --zt_real_dir ... --zt_fake_dir ... --ztprime_real_dir ... --ztprime_fake_dir ...

Labels: real = 0, fake (diffusion-generated) = 1.
"""

import argparse
import csv
import gc
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score, confusion_matrix,
)
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LatentSeDIDDataset(Dataset):

    LAT_EXTS = {".pt", ".pth"}

    def __init__(self, zt_real_dir, zt_fake_dir, ztprime_real_dir, ztprime_fake_dir, abs_diff=True):
        self.abs_diff = abs_diff

        zt_real_paths  = sorted(p for p in Path(zt_real_dir).iterdir()  if p.suffix.lower() in self.LAT_EXTS)
        zt_fake_paths  = sorted(p for p in Path(zt_fake_dir).iterdir()  if p.suffix.lower() in self.LAT_EXTS)
        zt_all_paths   = zt_real_paths + zt_fake_paths
        ztprime_dirs   = [ztprime_real_dir] * len(zt_real_paths) + [ztprime_fake_dir] * len(zt_fake_paths)
        all_labels     = [0] * len(zt_real_paths) + [1] * len(zt_fake_paths)

        self.zt_paths, self.ztprime_paths, self.labels = [], [], []
        skipped = 0
        for zt_path, ztprime_dir, label in zip(zt_all_paths, ztprime_dirs, all_labels):
            ztprime_path = Path(ztprime_dir) / (zt_path.stem + zt_path.suffix)
            if not ztprime_path.exists() or os.path.getsize(zt_path) == 0 or os.path.getsize(ztprime_path) == 0:
                skipped += 1
                continue
            self.zt_paths.append(zt_path)
            self.ztprime_paths.append(ztprime_path)
            self.labels.append(label)

        print(f"  Real: {self.labels.count(0)}, Fake: {self.labels.count(1)}, "
              f"Valid: {len(self.zt_paths)} (skipped {skipped} corrupt/missing pairs)")

    def __len__(self):
        return len(self.zt_paths)

    def __getitem__(self, idx):
        zt      = torch.load(self.zt_paths[idx],      map_location="cpu", weights_only=True).float()
        ztprime = torch.load(self.ztprime_paths[idx], map_location="cpu", weights_only=True).float()
        label   = self.labels[idx]

        if zt.shape != ztprime.shape:
            raise ValueError(f"Shape mismatch: {tuple(zt.shape)} vs {tuple(ztprime.shape)}")

        if zt.dim() == 4 and zt.shape[0] == 1:
            zt      = zt.squeeze(0)
            ztprime = ztprime.squeeze(0)

        diff  = torch.abs(zt - ztprime) if self.abs_diff else zt - ztprime
        error = diff.pow(2).sum().sqrt()

        return error, diff, label


def get_resnet18(in_channels: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc    = nn.Linear(model.fc.in_features, 2)
    return model

def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    fpr_arr, tpr_arr, _ = roc_curve(y_true, scores)
    idx = max(0, np.searchsorted(fpr_arr, target_fpr, side="right") - 1)
    return float(tpr_arr[idx])


def _report(y_true, scores, y_pred, loss=None, threshold=None) -> dict:
    return dict(
        auc          = roc_auc_score(y_true, scores),
        acc          = accuracy_score(y_true, y_pred),
        precision    = precision_score(y_true, y_pred, zero_division=0),
        recall       = recall_score(y_true, y_pred, zero_division=0),
        f1           = f1_score(y_true, y_pred, zero_division=0),
        tpr_at_fpr1  = tpr_at_fpr(y_true, scores, 0.01),
        tpr_at_fpr01 = tpr_at_fpr(y_true, scores, 0.001),
        confusion_matrix = confusion_matrix(y_true, y_pred),
        n_real       = int((y_true == 0).sum()),
        n_fake       = int((y_true == 1).sum()),
        **({} if loss      is None else {"val_loss":  loss}),
        **({} if threshold is None else {"threshold": threshold}),
    )


def print_metrics(m: dict, title: str = ""):
    print(f"\n{'─'*52}")
    if title:
        print(f"  {title}")
        print(f"{'─'*52}")
    print(f"  AUC          : {m['auc']:.4f}")
    print(f"  ACC          : {m['acc']*100:.2f}%")
    print(f"  Precision    : {m['precision']:.4f}")
    print(f"  Recall       : {m['recall']:.4f}")
    print(f"  F1           : {m['f1']:.4f}")
    print(f"  TPR@FPR=1%   : {m['tpr_at_fpr1']:.4f}")
    print(f"  TPR@FPR=0.1% : {m['tpr_at_fpr01']:.4f}")
    if "val_loss"  in m: print(f"  Val Loss     : {m['val_loss']:.4f}")
    if "threshold" in m: print(f"  Threshold    : {m['threshold']:.6f}")
    cm = m["confusion_matrix"]
    print(f"  Confusion Matrix  (real=0, fake=1):")
    print(f"    TN={cm[0,0]:>6}  FP={cm[0,1]:>6}")
    print(f"    FN={cm[1,0]:>6}  TP={cm[1,1]:>6}")
    print(f"  Samples      : real={m['n_real']}, fake={m['n_fake']}")
    print(f"{'─'*52}\n")


def _metrics_to_flat_row(m: dict, variant: str, split: str, epoch: int | None = None) -> dict:
    cm = m["confusion_matrix"]
    return {
        "variant":      variant,
        "split":        split,
        "epoch":        epoch if epoch is not None else "",
        "auc":          round(m["auc"],          6),
        "acc":          round(m["acc"],          6),
        "precision":    round(m["precision"],    6),
        "recall":       round(m["recall"],       6),
        "f1":           round(m["f1"],           6),
        "tpr_at_fpr1":  round(m["tpr_at_fpr1"],  6),
        "tpr_at_fpr01": round(m["tpr_at_fpr01"], 6),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        "n_real":       m["n_real"],
        "n_fake":       m["n_fake"],
        "val_loss":     round(m["val_loss"],  6) if "val_loss"  in m else "",
        "threshold":    round(m["threshold"], 6) if "threshold" in m else "",
    }


CSV_FIELDNAMES = [
    "variant", "split", "epoch",
    "auc", "acc", "precision", "recall", "f1",
    "tpr_at_fpr1", "tpr_at_fpr01",
    "tn", "fp", "fn", "tp",
    "n_real", "n_fake",
    "val_loss", "threshold",
]


def append_to_csv(csv_path: str, row: dict) -> None:
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── SeDIDStat ─────────────────────────────────────────────────────────────────

def _collect_errors(loader: DataLoader, desc: str):
    errors, labels = [], []
    for err, _, lab in tqdm(loader, desc=desc):
        errors.extend(err.numpy())
        labels.extend(lab.numpy())
    return np.array(errors), np.array(labels)


def run_stat(fit_loader: DataLoader, test_loader: DataLoader, csv_path: str) -> dict:
    """Pick the accuracy-maximising threshold on fit_loader, report on test_loader.
    Lower reconstruction error ⇒ more likely generated, so score = -error."""
    fit_err, fit_lab = _collect_errors(fit_loader, "[SeDIDStat] fitting threshold")
    fit_scores = -fit_err
    _, _, thresholds = roc_curve(fit_lab, fit_scores)
    accs = [accuracy_score(fit_lab, (fit_scores >= t).astype(int)) for t in thresholds]
    threshold = float(thresholds[int(np.argmax(accs))])

    test_err, test_lab = _collect_errors(test_loader, "[SeDIDStat] evaluating")
    test_scores = -test_err
    m = _report(test_lab, test_scores, (test_scores >= threshold).astype(int), threshold=threshold)
    append_to_csv(csv_path, _metrics_to_flat_row(m, variant="SeDIDStat", split="test"))
    return m


# ── SeDIDNNs ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_nns(model, loader, criterion) -> dict:
    model.eval()
    all_labels, all_probs, all_preds = [], [], []
    total_loss = 0.0
    for _, diffs, labels in loader:
        diffs      = diffs.to(device, non_blocking=True)
        labels_dev = labels.to(device, non_blocking=True)
        logits     = model(diffs)
        total_loss += criterion(logits, labels_dev).item()
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        all_preds.extend(logits.argmax(1).cpu().numpy())
    return _report(np.array(all_labels), np.array(all_probs), np.array(all_preds),
                   loss=total_loss / max(len(loader), 1))


def run_nns(args, train_dataset, val_loader, test_loader, csv_path: str) -> dict:
    workers = args.num_workers
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True, persistent_workers=workers > 0)

    in_channels = train_dataset[0][1].shape[0]
    model       = get_resnet18(in_channels).to(device)
    optimizer   = optim.SGD(model.parameters(), lr=args.init_lr, momentum=0.9, weight_decay=5e-4)
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
    criterion   = nn.CrossEntropyLoss()
    ckpt_path   = os.path.join(args.output_dir, "sedid_nns_best.pth")

    print(f"\nSeDIDNNs | in_channels={in_channels} | "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    best_auc = -1.0
    for epoch in range(args.n_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.n_epochs}")
        for batch_idx, (_, diffs, labels) in enumerate(pbar):
            diffs      = diffs.to(device, non_blocking=True)
            labels_dev = labels.to(device, non_blocking=True)

            logits = model(diffs)
            loss   = criterion(logits, labels_dev) / args.accumulation_steps
            loss.backward()

            if (batch_idx + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.accumulation_steps
            correct    += logits.argmax(1).eq(labels_dev).sum().item()
            total      += labels_dev.size(0)
            pbar.set_postfix(loss=f"{loss.item()*args.accumulation_steps:.4f}",
                             acc=f"{100.*correct/total:.2f}%")

        if len(train_loader) % args.accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        val_m = evaluate_nns(model, val_loader, criterion)
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.n_epochs} | Train Acc: {100.*correct/total:.2f}% | "
              f"Train Loss: {total_loss/len(train_loader):.4f}")
        print_metrics(val_m, title="Val")
        append_to_csv(csv_path, _metrics_to_flat_row(val_m, variant="SeDIDNNs", split="val", epoch=epoch + 1))

        if val_m["auc"] > best_auc:
            best_auc = val_m["auc"]
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_metrics": val_m, "args": vars(args)}, ckpt_path)
            print(f"  → Saved best model (val AUC={best_auc:.4f})")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final, unbiased number: best-on-val checkpoint evaluated once on test.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_m = evaluate_nns(model, test_loader, criterion)
    append_to_csv(csv_path, _metrics_to_flat_row(test_m, variant="SeDIDNNs", split="test",
                                                 epoch=ckpt["epoch"] + 1))
    print(f"\nTraining complete. Best val AUC: {best_auc:.4f} (epoch {ckpt['epoch'] + 1})")
    return test_m


# ── CLI / main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Latent-space SeDID detectors on precomputed (z_t, z'_t) pairs.")
    p.add_argument("--real_latents", type=str, help="invert.py --out_dir for real images.")
    p.add_argument("--fake_latents", type=str, help="invert.py --out_dir for generated images.")
    p.add_argument("--step",         type=int, help="Which SeDID step k to use (the stepK subfolder).")
    p.add_argument("--zt_real_dir",      type=str)
    p.add_argument("--zt_fake_dir",      type=str)
    p.add_argument("--ztprime_real_dir", type=str)
    p.add_argument("--ztprime_fake_dir", type=str)
    p.add_argument("--output_dir",         type=str, default="outputs/sedid")
    p.add_argument("--variant",            type=str, default="both", choices=["stat", "nns", "both"])
    p.add_argument("--abs_diff",           type=lambda x: x.lower() in ("1", "true", "yes"), default=True)
    p.add_argument("--val_ratio",          type=float, default=0.1)
    p.add_argument("--test_ratio",         type=float, default=0.1)
    p.add_argument("--batch_size",         type=int,   default=128)
    p.add_argument("--accumulation_steps", type=int,   default=1)
    p.add_argument("--n_epochs",           type=int,   default=20)
    p.add_argument("--init_lr",            type=float, default=0.001)
    p.add_argument("--num_workers",        type=int,   default=4)
    p.add_argument("--seed",               type=int,   default=42)
    args = p.parse_args()

    if args.real_latents or args.fake_latents:
        if not (args.real_latents and args.fake_latents and args.step is not None):
            p.error("--real_latents, --fake_latents and --step must be given together.")
        s = f"step{args.step}"
        args.zt_real_dir      = os.path.join(args.real_latents, "zt", s)
        args.ztprime_real_dir = os.path.join(args.real_latents, "zt_prime", s)
        args.zt_fake_dir      = os.path.join(args.fake_latents, "zt", s)
        args.ztprime_fake_dir = os.path.join(args.fake_latents, "zt_prime", s)
    if not all([args.zt_real_dir, args.zt_fake_dir, args.ztprime_real_dir, args.ztprime_fake_dir]):
        p.error("Give either --real_latents/--fake_latents/--step or all four --zt*_dir paths.")
    return args


def stratified_split(labels, val_ratio, test_ratio, seed):
    """Split indices into train / val / test, keeping the real:fake ratio in each part."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train, val, test = [], [], []
    for cls in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == cls))
        n_test, n_val = int(round(len(idx) * test_ratio)), int(round(len(idx) * val_ratio))
        test += idx[:n_test].tolist()
        val += idx[n_test:n_test + n_val].tolist()
        train += idx[n_test + n_val:].tolist()
    return sorted(train), sorted(val), sorted(test)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "results.csv")

    print("Loading latent pairs …")
    dataset = LatentSeDIDDataset(args.zt_real_dir, args.zt_fake_dir,
                                 args.ztprime_real_dir, args.ztprime_fake_dir, abs_diff=args.abs_diff)
    tr, va, te = stratified_split(dataset.labels, args.val_ratio, args.test_ratio, args.seed)
    train_ds, val_ds, test_ds = Subset(dataset, tr), Subset(dataset, va), Subset(dataset, te)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    loader = lambda ds: DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, pin_memory=True)

    if args.variant in ("stat", "both"):
        print("\n" + "═" * 52)
        stat_m = run_stat(loader(train_ds), loader(test_ds), csv_path)
        print_metrics(stat_m, title="SeDIDStat — test")
        torch.save(stat_m, os.path.join(args.output_dir, "sedid_stat_metrics.pt"))

    if args.variant in ("nns", "both"):
        print("\n" + "═" * 52)
        nns_m = run_nns(args, train_ds, loader(val_ds), loader(test_ds), csv_path)
        print_metrics(nns_m, title="SeDIDNNs — test (best-on-val checkpoint)")


if __name__ == "__main__":
    main()
