"""
test.py — Duality AI Offroad Semantic Segmentation Hackathon
Runs inference on unseen test images and saves colorized segmentation outputs.

Usage:
    conda activate EDU
    python test.py
    python test.py --model segmentation_head.pth --test_dir dataset/testImages --output_dir predictions
"""

import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── CONFIG ──────────────────────────────────────────────────────────────────

CONFIG = {
    "model_path":   "segmentation_head.pth",
    "test_rgb_dir": "dataset/testImages",
    "output_dir":   "predictions",
    "img_size":     448,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
    "num_classes":  10,
    "class_names": [
        "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
        "Ground Clutter", "Flowers", "Logs", "Rocks", "Landscape", "Sky"
    ],
    # High-contrast color palette (RGB) for visualization
    "palette": [
        (34,  139, 34),    # Trees         — Forest Green
        (0,   200, 83),    # Lush Bushes   — Bright Green
        (210, 180, 140),   # Dry Grass     — Tan
        (160, 120, 60),    # Dry Bushes    — Brown
        (139, 90,  43),    # Ground Clutter— Saddle Brown
        (255, 20,  147),   # Flowers       — Deep Pink
        (101, 67,  33),    # Logs          — Dark Brown
        (128, 128, 128),   # Rocks         — Gray
        (194, 178, 128),   # Landscape     — Sand
        (30,  144, 255),   # Sky           — Dodger Blue
    ],
    "class_map": {
        100:   0,
        200:   1,
        300:   2,
        500:   3,
        550:   4,
        600:   5,
        700:   6,
        800:   7,
        7100:  8,
        10000: 9,
    },
    "ignore_index": 255,
}

# ─── MODEL (must match train.py) ─────────────────────────────────────────────

class DINOv2SegmentationModel(nn.Module):
    def __init__(self, num_classes, backbone_name="dinov2_vits14"):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", backbone_name, pretrained=False
        )
        embed_dim = self.backbone.embed_dim
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        patch_size  = self.backbone.patch_size
        h_patches   = H // patch_size
        w_patches   = W // patch_size
        features     = self.backbone.forward_features(x)
        patch_tokens = features["x_norm_patchtokens"]
        B, N, D      = patch_tokens.shape
        logits       = self.head(patch_tokens.reshape(B * N, D))
        logits       = logits.reshape(B, N, -1).permute(0, 2, 1)
        logits       = logits.reshape(B, -1, h_patches, w_patches)
        logits       = nn.functional.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )
        return logits


# ─── UTILITIES ───────────────────────────────────────────────────────────────

def mask_to_color(mask_np, palette):
    """Convert integer class mask to RGB color image."""
    h, w    = mask_np.shape
    color   = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, rgb in enumerate(palette):
        color[mask_np == cls_idx] = rgb
    return color


def compute_iou(preds, targets, num_classes, ignore_index=255):
    ious = []
    for cls in range(num_classes):
        pred_mask   = (preds == cls)
        target_mask = (targets == cls) & (targets != ignore_index)
        inter = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        if union == 0:
            continue
        ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def map_seg_mask(seg_np, class_map, ignore_index=255):
    out = np.full(seg_np.shape, ignore_index, dtype=np.int64)
    for raw, idx in class_map.items():
        out[seg_np == raw] = idx
    return out


# ─── INFERENCE ───────────────────────────────────────────────────────────────

def run_inference(model, img_path, transform, device, img_size):
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    inp = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    # Resize back to original resolution
    pred_pil = Image.fromarray(pred.astype(np.uint8))
    pred_pil = pred_pil.resize((orig_w, orig_h), Image.NEAREST)
    return np.array(pred_pil), img


def save_visualization(orig_img, pred_mask, save_path, palette, class_names,
                        gt_mask=None):
    """Save side-by-side: Original | Prediction (| Ground Truth if available)."""
    n_panels = 3 if gt_mask is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

    axes[0].imshow(orig_img)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    color_pred = mask_to_color(pred_mask, palette)
    axes[1].imshow(color_pred)
    axes[1].set_title("Predicted Segmentation", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    if gt_mask is not None and n_panels == 3:
        color_gt = mask_to_color(gt_mask, palette)
        axes[2].imshow(color_gt)
        axes[2].set_title("Ground Truth", fontsize=12, fontweight="bold")
        axes[2].axis("off")

    # Legend
    legend_patches = [
        mpatches.Patch(color=np.array(c) / 255.0, label=name)
        for c, name in zip(palette, class_names)
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test DINOv2 segmentation model")
    parser.add_argument("--model",      default=CONFIG["model_path"])
    parser.add_argument("--test_dir",   default=CONFIG["test_rgb_dir"])
    parser.add_argument("--output_dir", default=CONFIG["output_dir"])
    parser.add_argument("--img_size",   type=int, default=CONFIG["img_size"])
    parser.add_argument("--seg_dir",    default=None,
                        help="Optional: path to ground truth masks for IoU evaluation")
    args = parser.parse_args()

    device = CONFIG["device"]
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from '{args.model}'...")
    checkpoint = torch.load(args.model, map_location=device)
    model = DINOv2SegmentationModel(CONFIG["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Model loaded (trained to epoch {checkpoint.get('epoch', '?')}, "
          f"best val IoU={checkpoint.get('val_iou', 0):.4f})")

    transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # Gather test images
    test_dir   = Path(args.test_dir)
    img_paths  = sorted(test_dir.glob("*.png")) + \
                 sorted(test_dir.glob("*.jpg")) + \
                 sorted(test_dir.glob("*.jpeg"))
    print(f"Found {len(img_paths)} test images in '{test_dir}'")

    os.makedirs(args.output_dir, exist_ok=True)
    color_dir = os.path.join(args.output_dir, "colorized")
    viz_dir   = os.path.join(args.output_dir, "visualizations")
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(viz_dir,   exist_ok=True)

    all_ious   = []
    has_gt     = args.seg_dir is not None

    for i, img_path in enumerate(img_paths):
        print(f"  [{i+1}/{len(img_paths)}] Processing {img_path.name} ...", end=" ")

        pred_mask, orig_img = run_inference(
            model, img_path, transform, device, args.img_size
        )

        # Save colorized mask as PNG
        color_mask = Image.fromarray(mask_to_color(pred_mask, CONFIG["palette"]))
        color_path = os.path.join(color_dir, img_path.stem + "_pred.png")
        color_mask.save(color_path)

        # Load GT if available for IoU
        gt_mask = None
        if has_gt:
            gt_path = Path(args.seg_dir) / img_path.name
            if gt_path.exists():
                gt_raw   = np.array(Image.open(gt_path))
                if gt_raw.ndim == 3:
                    gt_raw = gt_raw[..., 0].astype(np.int32) * 100
                else:
                    gt_raw = gt_raw.astype(np.int32)
                gt_mask  = map_seg_mask(gt_raw, CONFIG["class_map"])
                iou      = compute_iou(pred_mask, gt_mask, CONFIG["num_classes"])
                all_ious.append(iou)
                print(f"IoU={iou:.4f}", end=" ")

        # Save visualization
        viz_path = os.path.join(viz_dir, img_path.stem + "_viz.png")
        save_visualization(
            orig_img, pred_mask, viz_path,
            CONFIG["palette"], CONFIG["class_names"], gt_mask
        )
        print("✓")

    print(f"\nResults saved to '{args.output_dir}/'")
    print(f"  Colorized masks  → {color_dir}/")
    print(f"  Visualizations   → {viz_dir}/")

    if all_ious:
        mean_iou = np.mean(all_ious)
        print(f"\n{'='*40}")
        print(f"Test Mean IoU: {mean_iou:.4f}  ({len(all_ious)} images evaluated)")
        print(f"{'='*40}")
        # Save results to file
        results_path = os.path.join(args.output_dir, "test_results.txt")
        with open(results_path, "w") as f:
            f.write("Test Evaluation Results\n")
            f.write("=" * 40 + "\n")
            f.write(f"Number of images: {len(all_ious)}\n")
            f.write(f"Mean IoU:         {mean_iou:.4f}\n\n")
            f.write("Per-image IoU:\n")
            for path, iou in zip(img_paths, all_ious):
                f.write(f"  {path.name}: {iou:.4f}\n")
        print(f"Test results saved to '{results_path}'")
    else:
        print("\nNote: No ground truth masks found. Predictions saved without IoU evaluation.")
        print("To evaluate IoU, pass --seg_dir path/to/gt/masks")

    print("\nInference complete!")


if __name__ == "__main__":
    main()
