import subprocess, sys

def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

pip_install("timm")
pip_install("scikit-fuzzy")
pip_install("scikit-learn")
pip_install("matplotlib")
pip_install("seaborn")
 
import os, random, copy, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from PIL import Image, ImageEnhance

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import timm

import skfuzzy as fuzz
import skfuzzy.control as ctrl

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, roc_auc_score
)

warnings.filterwarnings("ignore")


  
SEED         = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE     = 224
BATCH_SIZE   = 16        # safe for T4 + Swin-Tiny
LR           = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS_SUP   = 40        # epochs per semi-supervised cycle
MAX_CYCLES   = 5         # total semi-supervised cycles
PATIENCE     = 5         # early-stopping patience (epochs)
CONF_THRESH  = 0.85      # fuzzy confidence threshold for pseudo-labels
NUM_CLASSES  = 2
NUM_WORKERS  = 2
 
BASE_INPUT    = Path("/.......your dataset........./")
DATA_ROOT     = BASE_INPUT / "dataset_merged"
LABELED_DIR   = DATA_ROOT / "labeled"
UNLABELED_DIR = DATA_ROOT / "unlabeled"
OUTPUT_DIR    = Path("/kaggle/working")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

 
CLASS_NAMES   = ["normal", "osteoporosis"]   # 0=normal, 1=osteoporosis
CLASS_FOLDERS = {
    "normal"       : 0,
    "osteoporosis" : 1,
}

  
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True

set_seed()

print(f"Device      : {DEVICE}")
print(f"Data root   : {DATA_ROOT}")
print(f"Labeled dir : {LABELED_DIR}")
print(f"Unlabeled   : {UNLABELED_DIR}")
print(f"Output dir  : {OUTPUT_DIR}")

 

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def base_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

def aug_transform():
    """Extra augmentation applied only during labeled/pseudo training."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    ])


class MultiViewTransform:
    """
    Input  : single PIL.Image
    Output : tuple (v1, v2, v3) — three tensors (3, 224, 224)

    View 1: Original image
    View 2: Contrast + Sharpness enhanced  → reveals bone density changes
    View 3: Center-cropped ROI + resized   → zoomed joint/bone area
    """
    def __init__(self, augment: bool = False):
        self.augment = augment
        self.base    = base_transform()
        self.aug     = aug_transform()

    @staticmethod
    def _contrast_view(img: Image.Image) -> Image.Image:
        img = ImageEnhance.Contrast(img).enhance(2.2)
        img = ImageEnhance.Sharpness(img).enhance(1.8)
        return img

    @staticmethod
    def _roi_view(img: Image.Image) -> Image.Image:
        w, h = img.size
        mw, mh = int(w * 0.15), int(h * 0.15)
        img = img.crop((mw, mh, w - mw, h - mh))
        return img.resize((w, h), Image.BILINEAR)

    def __call__(self, img: Image.Image):
        if self.augment:
            img = self.aug(img)
        v1 = self.base(img)
        v2 = self.base(self._contrast_view(img.copy()))
        v3 = self.base(self._roi_view(img.copy()))
        return v1, v2, v3

 
class LabeledDataset(Dataset):
    """
    Reads labeled/normal/ and labeled/osteoporosis/.
    Returns (v1, v2, v3, label_tensor).
    """
    def __init__(self, root: Path, augment: bool = False):
        self.transform = MultiViewTransform(augment=augment)
        self.samples   = []

        avail = {f.name.lower(): f for f in root.iterdir() if f.is_dir()}

        for cls_name, cls_idx in CLASS_FOLDERS.items():
            matched = [v for k, v in avail.items() if k == cls_name.lower()]
            if not matched:
                print(f"  [WARN] Folder '{cls_name}' not found in {root}")
                print(f"         Available folders: {list(avail.keys())}")
                continue
            folder = matched[0]
            imgs   = [p for p in folder.rglob("*") if is_image(p)]
            self.samples += [(p, cls_idx) for p in imgs]
            print(f"  Class '{cls_name}' ({cls_idx}) : {len(imgs)} images")

        print(f"  Total labeled samples   : {len(self.samples)}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        v1, v2, v3  = self.transform(img)
        return v1, v2, v3, torch.tensor(label, dtype=torch.long)


class UnlabeledDataset(Dataset):
    """
    Reads all images from unlabeled/ (flat folder).
    Returns (v1, v2, v3, path_string).
    """
    def __init__(self, root: Path):
        self.transform = MultiViewTransform(augment=False)
        self.paths     = [p for p in root.rglob("*") if is_image(p)]
        print(f"  Total unlabeled images  : {len(self.paths)}\n")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img  = Image.open(path).convert("RGB")
        v1, v2, v3 = self.transform(img)
        return v1, v2, v3, str(path)


class PseudoLabeledDataset(Dataset):
    """
    Accumulates pseudo-labeled samples across semi-supervised cycles.
    """
    def __init__(self):
        self.samples   = []
        self.transform = MultiViewTransform(augment=True)

    def add(self, path_str: str, label: int):
        self.samples.append((path_str, label))

    def clear(self):
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        v1, v2, v3  = self.transform(img)
        return v1, v2, v3, torch.tensor(label, dtype=torch.long)

 
class MultiViewSwinTransformer(nn.Module):
    """
    Architecture:
      View1 ──► Swin-Tiny ──► f1 (768-d) ─┐
      View2 ──► Swin-Tiny ──► f2 (768-d) ──► [f1‖f2‖f3] ──► Fusion FC ──► Prediction
      View3 ──► Swin-Tiny ──► f3 (768-d) ─┘   (2304-d)

    Per-view heads (head_v1/v2/v3) output probabilities fed to fuzzy logic.
    Shared backbone weights for all 3 views.
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()

        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0
        )
        feat_dim = self.backbone.num_features   # 768

 
        self.head_v1 = nn.Linear(feat_dim, num_classes)
        self.head_v2 = nn.Linear(feat_dim, num_classes)
        self.head_v3 = nn.Linear(feat_dim, num_classes)

        # Fusion MLP: 2304 → 512 → 256 → num_classes
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 3, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, v1, v2, v3):
        f1 = self.backbone(v1)    # (B, 768)
        f2 = self.backbone(v2)
        f3 = self.backbone(v3)
 
        p1 = F.softmax(self.head_v1(f1), dim=1)
        p2 = F.softmax(self.head_v2(f2), dim=1)
        p3 = F.softmax(self.head_v3(f3), dim=1)

        # Feature-level fusion → final prediction
        fused  = torch.cat([f1, f2, f3], dim=1)
        logits = self.fusion(fused)
        probs  = F.softmax(logits, dim=1)

        return logits, probs, p1, p2, p3

 
class KneeFuzzySystem:
    """
    Mamdani Fuzzy Inference System
      ─────────────────
    Inputs  : p1, p2, p3  — osteoporosis prob from each view [0,1]
    Output  : confidence  — final osteoporosis confidence     [0,1]
              > 0.5  →  Osteoporosis
              ≤ 0.5  →  Normal

    Membership functions (same for all 3 inputs):
      LOW    : trapmf [0.00, 0.00, 0.30, 0.45]
      MEDIUM : trimf  [0.35, 0.50, 0.65]
      HIGH   : trapmf [0.55, 0.70, 1.00, 1.00]

    Output MFs:
      normal_high : trapmf [0.00, 0.00, 0.15, 0.25]
      normal_med  : trimf  [0.15, 0.28, 0.40]
      uncertain   : trimf  [0.35, 0.50, 0.65]
      osteo_med   : trimf  [0.55, 0.65, 0.78]
      osteo_high  : trapmf [0.72, 0.85, 1.00, 1.00]

    Rules (18 rules covering all combinations):
      R1 : ALL HIGH               → osteo_high    (strongest OA signal)
      R2 : H & H & M              → osteo_high
      R3 : H & M & H              → osteo_high
      R4 : M & H & H              → osteo_high
      R5 : H & H & L              → osteo_med
      R6 : H & L & H              → osteo_med
      R7 : L & H & H              → osteo_med
      R8 : H & M & M              → osteo_med
      R9 : M & H & M              → osteo_med
      R10: M & M & H              → osteo_med
      R11: ALL MEDIUM             → uncertain     (model not sure)
      R12: H & L & L              → uncertain
      R13: L & H & L              → uncertain
      R14: L & L & H              → uncertain
      R15: L & L & M              → normal_med
      R16: L & M & L              → normal_med
      R17: M & L & L              → normal_med
      R18: ALL LOW                → normal_high   (strongest Normal signal)
    """

    def __init__(self):
        universe = np.linspace(0, 1, 200)

        # Antecedents
        self.p1  = ctrl.Antecedent(universe, "p1")
        self.p2  = ctrl.Antecedent(universe, "p2")
        self.p3  = ctrl.Antecedent(universe, "p3")

        # Consequent
        self.out = ctrl.Consequent(universe, "confidence")

        # Membership functions
        for ant in [self.p1, self.p2, self.p3]:
            ant["low"]    = fuzz.trapmf(universe, [0.00, 0.00, 0.30, 0.45])
            ant["medium"] = fuzz.trimf (universe, [0.35, 0.50, 0.65])
            ant["high"]   = fuzz.trapmf(universe, [0.55, 0.70, 1.00, 1.00])

        self.out["normal_high"] = fuzz.trapmf(universe, [0.00, 0.00, 0.15, 0.25])
        self.out["normal_med"]  = fuzz.trimf (universe, [0.15, 0.28, 0.40])
        self.out["uncertain"]   = fuzz.trimf (universe, [0.35, 0.50, 0.65])
        self.out["osteo_med"]   = fuzz.trimf (universe, [0.55, 0.65, 0.78])
        self.out["osteo_high"]  = fuzz.trapmf(universe, [0.72, 0.85, 1.00, 1.00])

        p1, p2, p3, out = self.p1, self.p2, self.p3, self.out

        rules = [
            # R1–R4: Strong OA (2+ HIGH)
            ctrl.Rule(p1["high"]   & p2["high"]   & p3["high"],   out["osteo_high"]),  # R1
            ctrl.Rule(p1["high"]   & p2["high"]   & p3["medium"], out["osteo_high"]),  # R2
            ctrl.Rule(p1["high"]   & p2["medium"] & p3["high"],   out["osteo_high"]),  # R3
            ctrl.Rule(p1["medium"] & p2["high"]   & p3["high"],   out["osteo_high"]),  # R4
            # R5–R10: Moderate OA signal
            ctrl.Rule(p1["high"]   & p2["high"]   & p3["low"],    out["osteo_med"]),   # R5
            ctrl.Rule(p1["high"]   & p2["low"]    & p3["high"],   out["osteo_med"]),   # R6
            ctrl.Rule(p1["low"]    & p2["high"]   & p3["high"],   out["osteo_med"]),   # R7
            ctrl.Rule(p1["high"]   & p2["medium"] & p3["medium"], out["osteo_med"]),   # R8
            ctrl.Rule(p1["medium"] & p2["high"]   & p3["medium"], out["osteo_med"]),   # R9
            ctrl.Rule(p1["medium"] & p2["medium"] & p3["high"],   out["osteo_med"]),   # R10
            # R11–R14: Uncertain
            ctrl.Rule(p1["medium"] & p2["medium"] & p3["medium"], out["uncertain"]),   # R11
            ctrl.Rule(p1["high"]   & p2["low"]    & p3["low"],    out["uncertain"]),   # R12
            ctrl.Rule(p1["low"]    & p2["high"]   & p3["low"],    out["uncertain"]),   # R13
            ctrl.Rule(p1["low"]    & p2["low"]    & p3["high"],   out["uncertain"]),   # R14
            # R15–R17: Normal (2+ LOW)
            ctrl.Rule(p1["low"]    & p2["low"]    & p3["medium"], out["normal_med"]),  # R15
            ctrl.Rule(p1["low"]    & p2["medium"] & p3["low"],    out["normal_med"]),  # R16
            ctrl.Rule(p1["medium"] & p2["low"]    & p3["low"],    out["normal_med"]),  # R17
            # R18: Strong Normal
            ctrl.Rule(p1["low"]    & p2["low"]    & p3["low"],    out["normal_high"]), # R18
        ]

        system   = ctrl.ControlSystem(rules)
        self.sim = ctrl.ControlSystemSimulation(system)

    def infer(self, p1_val: float, p2_val: float, p3_val: float) -> float:
        """Returns fuzzy osteoporosis confidence in [0, 1]."""
        try:
            self.sim.input["p1"] = float(np.clip(p1_val, 0.01, 0.99))
            self.sim.input["p2"] = float(np.clip(p2_val, 0.01, 0.99))
            self.sim.input["p3"] = float(np.clip(p3_val, 0.01, 0.99))
            self.sim.compute()
            return float(self.sim.output["confidence"])
        except Exception:
            return float((p1_val + p2_val + p3_val) / 3.0)

    def batch_infer(self, p1_arr, p2_arr, p3_arr) -> np.ndarray:
        return np.array([self.infer(a, b, c)
                         for a, b, c in zip(p1_arr, p2_arr, p3_arr)])


  
# 7. TRAINING UTILITIES
  
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        v1, v2, v3, labels = [x.to(device) for x in batch]
        optimizer.zero_grad()

        logits, probs, p1, p2, p3 = model(v1, v2, v3)

        # Main fusion loss
        loss_main = criterion(logits, labels)

        # Auxiliary per-view losses
        # f1 = model.backbone(v1)
        # f2 = model.backbone(v2)
        # f3 = model.backbone(v3)
        # loss_v1 = criterion(model.head_v1(f1), labels)
        # loss_v2 = criterion(model.head_v2(f2), labels)
        # loss_v3 = criterion(model.head_v3(f3), labels)
        loss_v1 = criterion(p1, labels)
        loss_v2 = criterion(p2, labels)
        loss_v3 = criterion(p3, labels)

        loss = loss_main + 0.2 * (loss_v1 + loss_v2 + loss_v3)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct    += (probs.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for batch in loader:
        v1, v2, v3, labels = [x.to(device) for x in batch]
        logits, probs, *_  = model(v1, v2, v3)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct    += (probs.argmax(1) == labels).sum().item()
        total      += labels.size(0)
        all_probs .append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs  = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    return total_loss / total, correct / total, all_probs, all_labels


  
  
@torch.no_grad()
def generate_pseudo_labels(model, unlabeled_loader, fuzzy_sys,
                            threshold: float, device):
    """
    For each unlabeled image:
      1. Extract per-view osteoporosis probabilities from model
      2. Fuzzy logic combines p1_osteo, p2_osteo, p3_osteo → confidence
      3. Confidence check:
           ≥ threshold → ACCEPT (pseudo-label assigned, used in next training)
           <  threshold → REJECT (image skipped this cycle, re-evaluated next)

    Returns:
      accepted : list of (path_str, label_int, confidence_float)
      rejected : list of (path_str, label_int, confidence_float)
    """
    model.eval()
    accepted, rejected = [], []

    for v1, v2, v3, paths in unlabeled_loader:
        v1, v2, v3 = v1.to(device), v2.to(device), v3.to(device)
        _, probs, p1, p2, p3 = model(v1, v2, v3)

        # Osteoporosis (class 1) probability from each view
        p1_ost = p1[:, 1].cpu().numpy()
        p2_ost = p2[:, 1].cpu().numpy()
        p3_ost = p3[:, 1].cpu().numpy()

        fuzzy_conf = fuzzy_sys.batch_infer(p1_ost, p2_ost, p3_ost)

        for i, path in enumerate(paths):
            conf  = fuzzy_conf[i]
            label = 1 if conf >= 0.5 else 0
            cls_conf = conf if label == 1 else (1.0 - conf)

            if cls_conf >= threshold:
                accepted.append((path, label, cls_conf))
            else:
                rejected.append((path, label, cls_conf))

    n_normal = sum(1 for _, l, _ in accepted if l == 0)
    n_osteo  = sum(1 for _, l, _ in accepted if l == 1)
    print(f"  Pseudo-labels → Accepted: {len(accepted)} "
          f"[normal={n_normal}, osteoporosis={n_osteo}] | "
          f"Rejected: {len(rejected)}")
    return accepted, rejected

 
def plot_training_curves(history: dict, title_suffix: str, save_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-o", ms=4, label="Train")
    axes[0].plot(epochs, history["val_loss"],   "r-s", ms=4, label="Val")
    axes[0].set_title(f"Loss — {title_suffix}", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "b-o", ms=4, label="Train")
    axes[1].plot(epochs, history["val_acc"],   "r-s", ms=4, label="Val")
    axes[1].set_title(f"Accuracy — {title_suffix}", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fname = f"training_curves_{title_suffix.replace(' ', '_')}.png"
    fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  ✔ Saved {fname}")


def plot_confusion_matrix_fig(labels, preds, class_names, title, save_path):
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=1, linecolor="white",
                annot_kws={"size": 14}, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title(title,              fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  ✔ Saved {save_path.name}")


def plot_roc(labels, probs_osteo, title, save_path):
    fpr, tpr, _ = roc_curve(labels, probs_osteo)
    roc_auc     = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2.5,
            label=f"AUC = {roc_auc:.4f}")
    ax.fill_between(fpr, tpr, alpha=0.08, color="darkorange")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(title,                  fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  ✔ Saved {save_path.name}")


def show_pseudo_label_samples(accepted_list: list, save_dir: Path,
                               cycle: int, n: int = 20):
    """Show up to 20 pseudo-labeled images with assigned label + confidence."""
    samples = accepted_list[:n]
    if not samples:
        print("  No pseudo-labels to display.")
        return

    cols = 5
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.5))
    axes = np.array(axes).flatten()

    color_map = {0: "#2ecc71", 1: "#e74c3c"}
    label_map = {0: "Normal", 1: "Osteoporosis"}

    for i, (path_str, label, conf) in enumerate(samples):
        try:
            img = Image.open(path_str).convert("RGB").resize((200, 200))
        except Exception:
            axes[i].axis("off"); continue
        axes[i].imshow(img)
        axes[i].axis("off")
        color = color_map[label]
        axes[i].set_title(f"{label_map[label]}\nconf={conf:.3f}",
                          fontsize=8, color=color, fontweight="bold", pad=3)
        for spine in axes[i].spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)

    for j in range(len(samples), len(axes)):
        axes[j].axis("off")

    patches = [mpatches.Patch(color="#2ecc71", label="Normal"),
               mpatches.Patch(color="#e74c3c", label="Osteoporosis")]
    fig.legend(handles=patches, loc="upper center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 1.01))
    plt.suptitle(
        f"Pseudo-Labeled Samples — Cycle {cycle}  "
        f"(showing {len(samples)} of {len(accepted_list)})",
        fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()
    fname = f"pseudo_labels_cycle{cycle}.png"
    fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  ✔ Saved {fname}")


def print_report(labels, preds, class_names):
    print("\n" + "=" * 58)
    print("  CLASSIFICATION REPORT")
    print("=" * 58)
    print(classification_report(labels, preds,
                                 target_names=class_names, digits=4))

 
def main():
    print("\n" + "=" * 60)
    print("  Semi-Supervised Multi-View Swin Transformer")
    print("  Fuzzy Logic — Knee Osteoporosis Classification")
    print("=" * 60 + "\n")

    for d, name in [(LABELED_DIR,   "labeled"),
                    (UNLABELED_DIR, "unlabeled")]:
        if not d.exists():
            raise FileNotFoundError(
                f"\n❌ Directory not found: {d}\n"
                "   Please verify your Kaggle dataset path above.")
        print(f"  ✔ {name:12s} → {d}")

    print("\n  Folders inside labeled/:")
    for f in LABELED_DIR.iterdir():
        if f.is_dir():
            imgs = [p for p in f.rglob("*") if is_image(p)]
            print(f"    {f.name:25s} : {len(imgs)} images")

 
    print("\n[1/7] Loading datasets...")
    print("  Labeled (with augmentation):")
    full_aug   = LabeledDataset(LABELED_DIR, augment=True)
    print("  Labeled (no augmentation for val):")
    full_clean = LabeledDataset(LABELED_DIR, augment=False)

    if len(full_aug) == 0:
        raise ValueError("❌ No labeled images found. "
                         "Check CLASS_FOLDERS and folder names.")

    n_total = len(full_aug)
    n_val   = max(2, int(0.20 * n_total))
    n_train = n_total - n_val

    gen1 = torch.Generator().manual_seed(SEED)
    train_ds, _ = torch.utils.data.random_split(full_aug,   [n_train, n_val], generator=gen1)
    gen2 = torch.Generator().manual_seed(SEED)
    _, val_ds   = torch.utils.data.random_split(full_clean, [n_train, n_val], generator=gen2)

    print(f"\n  Train labeled : {n_train}")
    print(f"  Validation    : {n_val}")

    print("\n  Unlabeled:")
    unlabeled_ds = UnlabeledDataset(UNLABELED_DIR)

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)
    unlabeled_loader = DataLoader(
        unlabeled_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)

    # ── Build model   
    print("[2/7] Building Swin Transformer model (pretrained=True)...")
    model = MultiViewSwinTransformer(
        num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Backbone        : swin_tiny_patch4_window7_224")
    print(f"  Trainable params: {n_params:,}")

    # ── Fuzzy system   
    print("\n[3/7] Building Fuzzy Logic System...")
    fuzzy_sys = KneeFuzzySystem()
    print(f"  Rules     : 18 rules (LOW/MEDIUM/HIGH x 3 views)")
    print(f"  Threshold : {CONF_THRESH}")

    # ── Class-weighted loss ───────────────────────────────────
    all_labels_list = [lbl for _, lbl in full_aug.samples]
    n0    = all_labels_list.count(0)
    n1    = all_labels_list.count(1)
    total = n0 + n1
    w0    = total / (2.0 * n0 + 1e-8)
    w1    = total / (2.0 * n1 + 1e-8)
    weight    = torch.tensor([w0, w1], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight)
    print(f"\n  Class weights : normal={w0:.3f}, osteoporosis={w1:.3f}")

    # ── Tracking   ───
    pseudo_dataset    = PseudoLabeledDataset()
    best_val_acc      = 0.0
    best_model_wt     = copy.deepcopy(model.state_dict())
    global_history    = {k: [] for k in
                         ["train_loss","val_loss","train_acc","val_acc"]}
    prev_pseudo_count = -1

  
    # SEMI-SUPERVISED CYCLES
  
    print("\n[4/7] Starting Semi-Supervised Training Cycles...\n")

    for cycle in range(1, MAX_CYCLES + 1):
        print("─" * 58)
        print(f"  CYCLE  {cycle} / {MAX_CYCLES}")
        print("─" * 58)

        # Combine labeled + pseudo-labeled
        if len(pseudo_dataset) > 0:
            combined = ConcatDataset([train_ds, pseudo_dataset])
            print(f"  Data  : {n_train} labeled + "
                  f"{len(pseudo_dataset)} pseudo = {len(combined)} total")
        else:
            combined = train_ds
            print(f"  Data  : {n_train} labeled only (no pseudo yet)")

        train_loader = DataLoader(
            combined, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

        optimizer = optim.AdamW(model.parameters(),
                                 lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS_SUP, eta_min=1e-6)

        cycle_hist   = {k: [] for k in global_history}
        patience_cnt = 0
        best_cyc_acc = 0.0

        for epoch in range(1, EPOCHS_SUP + 1):
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, DEVICE)
            vl_loss, vl_acc, vl_probs, vl_labels = evaluate(
                model, val_loader, criterion, DEVICE)
            scheduler.step()

            cycle_hist["train_loss"].append(tr_loss)
            cycle_hist["val_loss"]  .append(vl_loss)
            cycle_hist["train_acc"] .append(tr_acc)
            cycle_hist["val_acc"]   .append(vl_acc)

            if vl_acc > best_val_acc:
                best_val_acc  = vl_acc
                best_model_wt = copy.deepcopy(model.state_dict())
                torch.save(best_model_wt, OUTPUT_DIR / "best_model.pth")

            if vl_acc > best_cyc_acc:
                best_cyc_acc = vl_acc
                patience_cnt = 0
            else:
                patience_cnt += 1

            if epoch % 5 == 0 or epoch == 1:
                print(f"    Ep {epoch:3d}/{EPOCHS_SUP} | "
                      f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
                      f"vl_loss={vl_loss:.4f} vl_acc={vl_acc:.4f}")

            if patience_cnt >= PATIENCE:
                print(f"  ⏹  Early stop at epoch {epoch} (patience={PATIENCE})")
                break

        for k in global_history:
            global_history[k].extend(cycle_hist[k])

        plot_training_curves(cycle_hist, f"Cycle {cycle}", OUTPUT_DIR)

        # ── PSEUDO-LABEL GENERATION ───────────────────────────
        print(f"\n  ► Pseudo-label generation (threshold={CONF_THRESH})...")
        accepted, rejected = generate_pseudo_labels(
            model, unlabeled_loader, fuzzy_sys, CONF_THRESH, DEVICE)

        # Show 20 pseudo-labeled sample images
        show_pseudo_label_samples(accepted, OUTPUT_DIR, cycle, n=20)

        # Refresh pseudo dataset
        pseudo_dataset.clear()
        for path, label, conf in accepted:
            pseudo_dataset.add(path, label)

        # Stopping criterion: negligible growth
        growth = abs(len(accepted) - prev_pseudo_count)
        if prev_pseudo_count >= 0 and growth < 10:
            print(f"\n  ⏹  Pseudo-label growth = {growth} (< 10). "
                  f"Stopping at cycle {cycle}.")
            break
        prev_pseudo_count = len(accepted)
        print()
 
    print("\n" + "=" * 58)
    print("[5/7] Final Evaluation (Best Model on Validation Set)")
    print("=" * 58)

    model.load_state_dict(best_model_wt)
    _, val_acc, val_probs, val_labels = evaluate(
        model, val_loader, criterion, DEVICE)
    val_preds = val_probs.argmax(axis=1)

    print(f"  Best Validation Accuracy : {val_acc:.4f}")
    try:
        auc_score = roc_auc_score(val_labels, val_probs[:, 1])
        print(f"  ROC-AUC Score            : {auc_score:.4f}")
    except Exception:
        pass

    print_report(val_labels, val_preds, CLASS_NAMES)

    plot_confusion_matrix_fig(
        val_labels, val_preds, CLASS_NAMES,
        "Confusion Matrix — Validation Set",
        OUTPUT_DIR / "confusion_matrix_val.png")

    plot_roc(
        val_labels, val_probs[:, 1],
        "ROC Curve — Osteoporosis vs Normal (Validation)",
        OUTPUT_DIR / "roc_curve_val.png")

    if global_history["train_loss"]:
        plot_training_curves(global_history, "All Cycles", OUTPUT_DIR)
 
    test_dir = DATA_ROOT / "test"
    if test_dir.exists():
        print("\n[6/7] Test Set Found — Evaluating...")
        test_ds = LabeledDataset(test_dir, augment=False)
        if len(test_ds) > 0:
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                                      shuffle=False, num_workers=NUM_WORKERS)
            _, test_acc, test_probs, test_labels = evaluate(
                model, test_loader, criterion, DEVICE)
            test_preds = test_probs.argmax(axis=1)
            print(f"  Test Accuracy : {test_acc:.4f}")
            print_report(test_labels, test_preds, CLASS_NAMES)
            plot_confusion_matrix_fig(
                test_labels, test_preds, CLASS_NAMES,
                "Confusion Matrix — Test Set",
                OUTPUT_DIR / "confusion_matrix_test.png")
            plot_roc(
                test_labels, test_probs[:, 1],
                "ROC Curve — Test Set",
                OUTPUT_DIR / "roc_curve_test.png")
        else:
            print("  Test folder empty — skipping.")
    else:
        print("\n[6/7] No test/ folder — using validation as final report.")
 
    print("\n[7/7] Saving final checkpoint...")
    torch.save({
        "model_state_dict" : best_model_wt,
        "class_names"      : CLASS_NAMES,
        "class_folders"    : CLASS_FOLDERS,
        "best_val_acc"     : best_val_acc,
        "config": {
            "img_size"    : IMG_SIZE,
            "num_classes" : NUM_CLASSES,
            "backbone"    : "swin_tiny_patch4_window7_224",
            "conf_thresh" : CONF_THRESH,
            "batch_size"  : BATCH_SIZE,
        }
    }, OUTPUT_DIR / "final_checkpoint.pth")

  
    print("\n" + "=" * 58)
    print("  ✅  TRAINING COMPLETE")
    print("=" * 58)
    print(f"  Best Validation Accuracy : {best_val_acc:.4f}")
    print(f"  Pseudo-labeled samples   : {len(pseudo_dataset)}")
    print(f"  Output directory         : {OUTPUT_DIR}\n")
    print("  Output files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"    📊  {f.name}")
    for f in sorted(OUTPUT_DIR.glob("*.pth")):
        print(f"    💾  {f.name}")
  
if __name__ == "__main__":
    main()