# main.py — single file, Colab-ready, no CLI
# Target: ≥99.4% val (50k/10k split) in ≤20 epochs with <20k params
# Concepts used: #layers, MaxPooling (placement), 3x3 convs, receptive field growth,
# Softmax (for metrics only), LR schedule (OneCycle), #kernels rationale,
# BatchNorm, Image Normalization, "transition layers" (Pools + GAP),
# Dropout (positioning), early failure signals, batch size effects.
# Optional fully-connected vs GAP: we use GAP + small Linear head.

import os, time, random, math
from contextlib import nullcontext
import numpy as np

# Try import; fallback to pip if needed.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torch.optim.lr_scheduler import OneCycleLR
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision"])
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torch.optim.lr_scheduler import OneCycleLR

# -------------------------
# Config (no CLI — simple & reproducible)
# -------------------------
EPOCHS        = 20
BATCH_SIZE    = 128        # smaller batch -> better generalization (lecture point)
LR_MAX        = 1.0e-2     # OneCycle peak LR (tuned for BN + aug)
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 2
SEED          = 1337
CHANNELS_LAST = False
LS_EPS        = 0.10       # label smoothing
EARLY_STOP    = 0.994      # stop when val ≥ 99.4%
PARAM_BUDGET  = 20_000

# -------------------------
# Repro
# -------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

def evaluate(model, loader, device):
    model.eval()
    n, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)
    return correct / max(1, n)

# -------------------------
# Model (≤20k params): 4 convs, 2 pools, GAP, Dropout, Linear
# Design rationale (maps to your bullets):
# - Only 3x3 convs (steady receptive field growth, strong inductive bias).
# - BN after every conv, before activation (stability, faster convergence).
# - MaxPool(2) AFTER each pair of convs (transition layers, controlled RF).
# - GAP instead of big FC stack (regularization + very few params).
# - Dropout only near the head (regularize classifier, avoid hurting features).
# - Channels: 1→8→16→28→52 (gradual increase; total params ~19.2k).
# - Softmax used only for reporting, not in loss (CE expects logits).
# -------------------------
class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 8, 3, padding=1, bias=True),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 16, 3, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  # 28 -> 14 (transition layer)

            # Block 2
            nn.Conv2d(16, 28, 3, padding=1, bias=True),
            nn.BatchNorm2d(28),
            nn.ReLU(inplace=True),

            nn.Conv2d(28, 52, 3, padding=1, bias=True),
            nn.BatchNorm2d(52),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  # 14 -> 7 (transition layer)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)  # GAP (transition to classifier)
        self.dropout = nn.Dropout(0.20)     # Dropout near the head only
        self.head = nn.Linear(52, 10)       # Tiny classifier head

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.head(x)  # logits
        return x

# -------------------------
# Train
# -------------------------
def run():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Data & Aug ---
    # Image normalization (lecture: image normalization is necessary)
    # RandomAffine adds translation/rotation/scale invariance (prevents overfitting).
    # RandomErasing (light) AFTER ToTensor/Normalize (expects a tensor).
    train_tfm = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.10, 0.10), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.06), ratio=(0.3, 3.3)),
    ])
    test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Datasets
    train_full = datasets.MNIST(root="./data", train=True, download=True, transform=train_tfm)
    test_ds    = datasets.MNIST(root="./data", train=False, download=True, transform=test_tfm)

    # Split exactly as requested: 50k/10k from the TRAIN set
    val_size   = 10_000
    train_size = len(train_full) - val_size  # 60k - 10k
    train_ds, val_ds = torch.utils.data.random_split(train_full, [train_size, val_size])

    pin = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=512, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader  = DataLoader(test_ds, batch_size=512, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    # Model
    model = SmallNet().to(device)
    if CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)

    params = count_params(model)
    print(f"Trainable params: {params}")
    assert params < PARAM_BUDGET, f"Param budget exceeded: {params} >= {PARAM_BUDGET}"

    # Optimizer, schedule, AMP (new API), label smoothing
    opt = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_loader)
    sched = OneCycleLR(opt, max_lr=LR_MAX, epochs=EPOCHS,
                       steps_per_epoch=steps_per_epoch, pct_start=0.25)

    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    autocast_ctx = (lambda: torch.amp.autocast('cuda')) if use_amp else nullcontext

    criterion = nn.CrossEntropyLoss(label_smoothing=LS_EPS)

    os.makedirs("checkpoints", exist_ok=True)
    log_path = "logs.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_acc,test_acc,params,secs\n")

    best_val = 0.0
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, running_acc, n_batches = 0.0, 0.0, 0

        for xb, yb in train_loader:
            if CHANNELS_LAST:
                xb = xb.to(memory_format=torch.channels_last)
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad(set_to_none=True)
            with (autocast_ctx() if use_amp else nullcontext()):
                logits = model(xb)
                loss = criterion(logits, yb)

            if use_amp:
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

            sched.step()
            running_loss += loss.item()
            running_acc  += accuracy(logits.detach(), yb)
            n_batches    += 1

        train_loss = running_loss / max(1, n_batches)
        train_acc  = running_acc  / max(1, n_batches)

        # Evaluate on the 50k/10k split AND the official test set
        val_acc  = evaluate(model, val_loader, device)
        test_acc = evaluate(model, test_loader, device)

        secs = time.time() - start
        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_acc:.4f},{test_acc:.4f},{params},{secs:.1f}\n")

        # Save best by validation (10k from train)
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict(), "params": params}, "checkpoints/best.pt")

        print(f"Epoch {epoch}/{EPOCHS} | "
              f"train_loss={train_loss:.3f} | train_acc={train_acc*100:.1f}% | "
              f"val_acc={val_acc*100:.2f}% | test_acc={test_acc*100:.2f}% | params={params}")

        # Early diagnostic (how to know early it's off)
        if epoch == 3 and val_acc < 0.97:
            print("[Signal] Val<97% by epoch 3 — increase LR_MAX slightly (e.g., 1.5e-2) "
                  "or strengthen augmentation.")

        # Stop once requirement met on the 10k validation
        if val_acc >= EARLY_STOP:
            print(f"Early stopping at epoch {epoch}: val_acc={val_acc:.4f} (>= 0.994).")
            break

    print("\nDone.")

if __name__ == "__main__":
    run()
