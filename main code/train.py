# train.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import SmallCNN
from cached_dataset import CachedNPZDataset


def compute_pos_weight_from_y(y: torch.Tensor) -> torch.Tensor:
    """
    pos_weight = (#neg / #pos) for BCEWithLogitsLoss
    """
    y_cpu = y.detach().cpu()
    pos = float((y_cpu == 1).sum().item())
    neg = float((y_cpu == 0).sum().item())
    pos = max(pos, 1.0)
    return torch.tensor([neg / pos], dtype=torch.float32)


@torch.no_grad()
def eval_metrics(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    tp = fp = tn = fn = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        total += y.numel()
        correct += (preds == y).sum().item()

        tp += ((preds == 1) & (y == 1)).sum().item()
        fp += ((preds == 1) & (y == 0)).sum().item()
        tn += ((preds == 0) & (y == 0)).sum().item()
        fn += ((preds == 0) & (y == 1)).sum().item()

    acc = correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return acc, precision, recall


def count_pos_neg(ds: CachedNPZDataset):
    """
    Count positives/negatives in a CachedNPZDataset.
    This iterates through the dataset once, which is fine for small/medium caches.
    If your cache becomes huge, we can optimize this by precomputing counts per file.
    """
    pos = 0
    neg = 0
    for _x, y in ds:
        if float(y.item()) >= 0.5:
            pos += 1
        else:
            neg += 1
    return pos, neg


def main():
    # -----------------------
    # Settings you can tweak
    # -----------------------
    cache_dir = "cache"
    nbins = 512

    batch_size = 64
    epochs = 12
    lr = 1e-3
    val_target_frac = 0.3
    seed = 42

    preload = False  # True loads all cached X/y into RAM (faster, uses more memory)

    # -----------------------
    # Repro + device
    # -----------------------
    np_rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -----------------------
    # Load cached dataset (NO downloading)
    # -----------------------
    full_ds = CachedNPZDataset(cache_dir=cache_dir, preload=False)

    targets = full_ds.get_targets()
    if len(targets) < 2:
        raise ValueError(
            f"Need at least 2 targets in cache to split train/val. Found: {len(targets)}"
        )

    np_rng.shuffle(targets)
    n_val_targets = max(1, int(len(targets) * val_target_frac))
    n_val_targets = min(n_val_targets, len(targets) - 1)

    val_targets = targets[:n_val_targets]
    train_targets = targets[n_val_targets:]

    print("Train targets:", len(train_targets))
    print("Val targets:", len(val_targets))

    train_ds = CachedNPZDataset(cache_dir=cache_dir, allowed_targets=train_targets, preload=preload)
    val_ds = CachedNPZDataset(cache_dir=cache_dir, allowed_targets=val_targets, preload=preload)

    # Optional: counts (can be slow if cache is very large)
    train_pos, train_neg = count_pos_neg(train_ds)
    val_pos, val_neg = count_pos_neg(val_ds)
    print("Train pos/neg:", train_pos, train_neg)
    print("Val   pos/neg:", val_pos, val_neg)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # -----------------------
    # Model
    # -----------------------
    model = SmallCNN(nbins).to(device)

    # -----------------------
    # Loss (pos_weight from TRAIN only)
    # -----------------------
    # Build y tensor once for pos_weight (faster than recounting each batch)
    y_train = torch.tensor([float(y.item()) for _x, y in train_ds], dtype=torch.float32)
    pos_weight = compute_pos_weight_from_y(y_train).to(device)

    print("pos_weight:", float(pos_weight.item()))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -----------------------
    # Checkpointing
    # -----------------------
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_smallcnn.pt"

    best_val_recall = -1.0

    # -----------------------
    # Train
    # -----------------------
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            batches += 1

        train_loss = running_loss / max(batches, 1)

        train_acc, train_prec, train_rec = eval_metrics(model, train_loader, device)
        val_acc, val_prec, val_rec = eval_metrics(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"loss {train_loss:.4f} | "
            f"train acc {train_acc:.3f} P {train_prec:.3f} R {train_rec:.3f} | "
            f"val acc {val_acc:.3f} P {val_prec:.3f} R {val_rec:.3f}"
        )

        if val_rec > best_val_recall:
            best_val_recall = val_rec
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "nbins": nbins,
                    "epoch": epoch,
                    "val_recall": val_rec,
                    "pos_weight": float(pos_weight.item()),
                    "train_targets": train_targets,
                    "val_targets": val_targets,
                    "cache_dir": cache_dir,
                },
                best_path,
            )
            print(f"  Saved best -> {best_path} (val recall {val_rec:.3f})")

    print("Training finished.")
    print("Best checkpoint:", best_path)


if __name__ == "__main__":
    main()