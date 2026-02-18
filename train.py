"""
train.py — Duality AI Offroad Semantic Segmentation Hackathon
Backbone: DINOv2 ViT-S/14 (facebook/dinov2_vits14)
Head:     Lightweight MLP segmentation decoder
Dataset:  Duality AI desert environment synthetic data

Usage:
    conda activate EDU
    python train.py
"""

import os
import time
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── CONFIG ──────────────────────────────────────────────────────────────────

CONFIG = {
    # Paths (adjust if your dataset is in a different location)
    "train_rgb_dir":   "dataset/Train/rgb",
    "train_seg_dir":   "dataset/Train/seg",
    "val_rgb_dir":     "dataset/Val/rgb",
    "val_seg_dir":     "dataset/Val/seg",
    "output_dir":      "train_stats",
    "model_save_path": "segmentation_head.pth",

    # Training hyperparameters
    "num_epochs":     30,
    "batch_size":     4,
    "lr":             1e-4,
    "weight_decay":   1e-4,
    "img_size":       448,      # DINOv2 patch=14, so must be divisible by 14
    "num_workers":    2,
    "device":         "cuda" if torch.cuda.is_available() else "cpu",
    "seed":           42,

    # Class mapping: raw pixel values → contiguous class indices
    # Classes: Trees(100), LushBushes(200), DryGrass(300), DryBushes(500),
    #          GroundClutter(550), Flowers(600), Logs(700), Rocks(800),
    #          Landscape(7100), Sky(10000)
    "class_map": {
        100:   0,   # Trees
        200:   1,   # Lush Bushes
        300:   2,   # Dry Grass
        500:   3,   # Dry Bushes
        550:   4,   # Ground Clutter
        600:   5,   # Flowers
        700:   6,   # Logs
        800:   7,   # Rocks
        7100:  8,   # Landscape
        10000: 9,   # Sky
    },
    "class_names": [
        "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
        "Ground Clutter", "Flowers", "Logs", "Rocks", "Landscape", "Sky"
    ],
    "num_classes": 10,
    "ignore_index": 255,
}

# ─── DATASET ─────────────────────────────────────────────────────────────────

class OffRoadDataset(Dataset):
    """Loads paired RGB and segmentation mask images."""

    def __init__(self, rgb_dir, seg_dir, img_size, class_map, augment=False):
        self.rgb_paths = sorted(Path(rgb_dir).glob("*.png")) + \
                         sorted(Path(rgb_dir).glob("*.jpg")) + \
                         sorted(Path(rgb_dir).glob("*.jpeg"))
        self.seg_dir   = Path(seg_dir)
        self.img_size  = img_size
        self.class_map = class_map
        self.augment   = augment

        self.rgb_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.rgb_paths)

    def _map_mask(self, mask_np):
        """Convert raw pixel values to contiguous class indices."""
        out = np.full(mask_np.shape, CONFIG["ignore_index"], dtype=np.int64)
        for raw, idx in self.class_map.items():
            out[mask_np == raw] = idx
        return out

    def _augment(self, img, mask):
        """Joint augmentation for image + mask."""
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)
        # Random vertical flip
        if torch.rand(1) > 0.5:
            img  = TF.vflip(img)
            mask = TF.vflip(mask)
        # Random color jitter (image only)
        img = T.ColorJitter(brightness=0.3, contrast=0.3,
                            saturation=0.3, hue=0.05)(img)
        return img, mask

    def __getitem__(self, idx):
        rgb_path = self.rgb_paths[idx]
        seg_path = self.seg_dir / rgb_path.name

        img  = Image.open(rgb_path).convert("RGB")
        seg  = Image.open(seg_path)

        # Resize mask with NEAREST to preserve class values
        seg  = seg.resize((self.img_size, self.img_size), Image.NEAREST)

        # Convert mask — handle both single-channel and multi-channel PNGs
        seg_np = np.array(seg)
        if seg_np.ndim == 3:
            # Some pipelines encode class ID in the red channel
            seg_np = seg_np[..., 0].astype(np.int32) * 100
        else:
            seg_np = seg_np.astype(np.int32)

        mask_mapped = self._map_mask(seg_np)
        mask_tensor = torch.from_numpy(mask_mapped).long()

        if self.augment:
            img, mask_tensor = self._augment(img, mask_tensor)

        img_tensor = self.rgb_transform(img)
        return img_tensor, mask_tensor


# ─── MODEL ───────────────────────────────────────────────────────────────────

class DINOv2SegmentationModel(nn.Module):
    """DINOv2 backbone + lightweight MLP decoder head."""

    def __init__(self, num_classes, backbone_name="dinov2_vits14"):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", backbone_name, pretrained=True
        )
        embed_dim = self.backbone.embed_dim  # 384 for vits14

        # Freeze backbone for first few epochs (fine-tune later)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Segmentation head: patch tokens → class logits
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning after warmup."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        B, C, H, W = x.shape
        patch_size  = self.backbone.patch_size   # 14
        h_patches   = H // patch_size
        w_patches   = W // patch_size

        # Extract patch tokens (exclude CLS)
        features = self.backbone.forward_features(x)
        patch_tokens = features["x_norm_patchtokens"]  # (B, N, D)

        # Classify each patch
        B, N, D = patch_tokens.shape
        logits   = self.head(patch_tokens.reshape(B * N, D))  # (B*N, C)
        logits   = logits.reshape(B, N, -1)                   # (B, N, C)
        logits   = logits.permute(0, 2, 1)                    # (B, C, N)
        logits   = logits.reshape(B, -1, h_patches, w_patches) # (B, C, h, w)

        # Upsample to original resolution
        logits = nn.functional.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )
        return logits


# ─── METRICS ─────────────────────────────────────────────────────────────────

def compute_iou(preds, targets, num_classes, ignore_index=255):
    """Compute mean IoU across all classes (ignoring ignore_index)."""
    ious = []
    preds   = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    for cls in range(num_classes):
        pred_mask   = (preds == cls)
        target_mask = (targets == cls) & (targets != ignore_index)
        intersection = (pred_mask & target_mask).sum()
        union        = (pred_mask | target_mask).sum()
        if union == 0:
            continue
        ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0


def compute_dice(preds, targets, num_classes, ignore_index=255):
    """Compute mean Dice score across all classes."""
    dices = []
    preds   = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    for cls in range(num_classes):
        pred_mask   = (preds == cls)
        target_mask = (targets == cls) & (targets != ignore_index)
        tp = (pred_mask & target_mask).sum()
        fp = pred_mask.sum() - tp
        fn = target_mask.sum() - tp
        denom = 2 * tp + fp + fn
        if denom == 0:
            continue
        dices.append((2 * tp) / denom)
    return float(np.mean(dices)) if dices else 0.0


def compute_accuracy(preds, targets, ignore_index=255):
    """Pixel accuracy excluding ignored pixels."""
    valid   = (targets != ignore_index)
    correct = (preds == targets) & valid
    return float(correct.sum() / valid.sum()) if valid.sum() > 0 else 0.0


# ─── TRAINING LOOP ───────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits = model(imgs)
            loss   = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss   = criterion(logits, masks)
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.append(preds)
        all_targets.append(masks)
    all_preds   = torch.cat(all_preds,   dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    avg_loss = total_loss / len(loader)
    iou      = compute_iou(all_preds, all_targets, num_classes)
    dice     = compute_dice(all_preds, all_targets, num_classes)
    acc      = compute_accuracy(all_preds, all_targets)
    return avg_loss, iou, dice, acc


# ─── PLOTTING ────────────────────────────────────────────────────────────────

def save_curves(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # Training vs Val loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss", color="#E74C3C")
    plt.plot(epochs, history["val_loss"],   label="Val Loss",   color="#3498DB")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training & Validation Loss")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved training curves to '{path}'")

    # IoU
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["val_iou"], label="Val IoU", color="#2ECC71", marker="o", markersize=3)
    plt.xlabel("Epoch"); plt.ylabel("Mean IoU"); plt.title("Validation IoU over Epochs")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "iou_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved IoU curves to '{path}'")

    # Dice
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["val_dice"], label="Val Dice", color="#9B59B6", marker="o", markersize=3)
    plt.xlabel("Epoch"); plt.ylabel("Mean Dice"); plt.title("Validation Dice Score over Epochs")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "dice_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved Dice curves to '{path}'")

    # Combined
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, history["train_loss"], color="#E74C3C", label="Train")
    axes[0].plot(epochs, history["val_loss"],   color="#3498DB", label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_iou"], color="#2ECC71", marker="o", markersize=3)
    axes[1].set_title("Val IoU"); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["val_dice"], color="#9B59B6", marker="o", markersize=3)
    axes[2].set_title("Val Dice"); axes[2].grid(True, alpha=0.3)

    plt.suptitle("Training Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "all_metrics_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved combined metrics curves to '{path}'")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train DINOv2 segmentation model")
    parser.add_argument("--epochs",     type=int,   default=CONFIG["num_epochs"])
    parser.add_argument("--batch_size", type=int,   default=CONFIG["batch_size"])
    parser.add_argument("--lr",         type=float, default=CONFIG["lr"])
    parser.add_argument("--img_size",   type=int,   default=CONFIG["img_size"])
    parser.add_argument("--unfreeze_epoch", type=int, default=5,
                        help="Epoch to unfreeze backbone for fine-tuning")
    args = parser.parse_args()

    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    device = CONFIG["device"]
    print(f"Using device: {device}")

    # Datasets
    train_dataset = OffRoadDataset(
        CONFIG["train_rgb_dir"], CONFIG["train_seg_dir"],
        args.img_size, CONFIG["class_map"], augment=True
    )
    val_dataset = OffRoadDataset(
        CONFIG["val_rgb_dir"], CONFIG["val_seg_dir"],
        args.img_size, CONFIG["class_map"], augment=False
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=CONFIG["num_workers"],
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=CONFIG["num_workers"],
                              pin_memory=True)

    # Model
    model = DINOv2SegmentationModel(CONFIG["num_classes"]).to(device)
    print(f"Backbone embedding dim: {model.backbone.embed_dim}")
    print(f"Patch tokens shape: {torch.Size([args.batch_size, (args.img_size // 14) ** 2, model.backbone.embed_dim])}")

    # Loss — ignore unrecognized pixels
    criterion = nn.CrossEntropyLoss(ignore_index=CONFIG["ignore_index"])

    # Optimizer — only head params initially
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=CONFIG["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    history = {k: [] for k in ["train_loss", "val_loss", "val_iou", "val_dice", "val_acc"]}
    best_iou = 0.0

    print("\nStarting training...")
    print("=" * 70)
    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone at specified epoch for fine-tuning
        if epoch == args.unfreeze_epoch:
            print(f"\n[Epoch {epoch}] Unfreezing backbone for fine-tuning...")
            model.unfreeze_backbone()
            # Re-create optimizer with all params
            optimizer = optim.AdamW(model.parameters(),
                                    lr=args.lr * 0.1,
                                    weight_decay=CONFIG["weight_decay"])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch + 1
            )

        t0         = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_iou, val_dice, val_acc = evaluate(
            model, val_loader, criterion, device, CONFIG["num_classes"]
        )
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_dice"].append(val_dice)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch:02d}/{args.epochs}] | "
              f"time={elapsed:.0f}s | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_iou={val_iou:.4f} | "
              f"val_dice={val_dice:.4f} | "
              f"val_acc={val_acc:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "config":      CONFIG,
                "val_iou":     val_iou,
            }, CONFIG["model_save_path"])
            print(f"  ✓ New best IoU={best_iou:.4f} — model saved.")

    print("\nSaving training curves...")
    save_curves(history, CONFIG["output_dir"])

    # Save evaluation metrics text file
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    metrics_path = os.path.join(CONFIG["output_dir"], "evaluation_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Final Evaluation Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Final Val Loss:     {history['val_loss'][-1]:.4f}\n")
        f.write(f"Final Val IoU:      {history['val_iou'][-1]:.4f}\n")
        f.write(f"Final Val Dice:     {history['val_dice'][-1]:.4f}\n")
        f.write(f"Final Val Accuracy: {history['val_acc'][-1]:.4f}\n")
        f.write(f"Best Val IoU:       {best_iou:.4f}\n")
        f.write("\nFull history (IoU per epoch):\n")
        for i, iou in enumerate(history["val_iou"], 1):
            f.write(f"  Epoch {i:02d}: {iou:.4f}\n")
    print(f"Saved evaluation metrics to {metrics_path}")
    print(f"Saved model to '{CONFIG['model_save_path']}'")

    print("\nFinal evaluation results:")
    print(f"  Final Val Loss:     {history['val_loss'][-1]:.4f}")
    print(f"  Final Val IoU:      {history['val_iou'][-1]:.4f}")
    print(f"  Final Val Dice:     {history['val_dice'][-1]:.4f}")
    print(f"  Final Val Accuracy: {history['val_acc'][-1]:.4f}")
    print(f"  Best Val IoU:       {best_iou:.4f}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
