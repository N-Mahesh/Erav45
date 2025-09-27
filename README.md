# MNIST: Sub-20k Param CNN (99.4% in ≤20 Epochs)

**Key Requirements (all met):**
- **Total Parameter Count Test**: 19,202 trainable parameters (<20k ✅)
- **Use of Batch Normalization**: ✅ BN after every convolution layer
- **Use of Dropout**: ✅ 0.20 Dropout applied before classifier head
- **Use of a Fully Connected Layer or GAP**: ✅ Global Average Pooling + Linear head

---

## Architecture
- **Layers**: 4 conv layers + 2 MaxPools + GAP + Linear
- **Conv kernels**: 3×3 only
- **BN**: after each convolution
- **ReLU** activations
- **Pooling**: MaxPool after every two conv layers (28→14, then 14→7)
- **Transition layers**: MaxPools + GAP
- **Head**: GAP → Dropout(0.20) → Linear(52→10)
- **Softmax**: used only for reporting; loss uses logits (CrossEntropy)

---

## Results

- **Validation Accuracy (10k held-out)**: 98.5%
- **Test Accuracy (official 10k)**: **99.4%**
- **Epochs**: 20
  
## Logs

Device: cuda
Trainable params: 19202
Epoch 1/20 | train_loss=1.657 | train_acc=56.5% | val_acc=86.78% | test_acc=93.27% | params=19202
Epoch 2/20 | train_loss=0.929 | train_acc=88.1% | val_acc=94.20% | test_acc=96.58% | params=19202
Epoch 3/20 | train_loss=0.793 | train_acc=92.6% | val_acc=94.01% | test_acc=96.48% | params=19202
[Signal] Val<97% by epoch 3 — increase LR_MAX slightly (e.g., 1.5e-2) or strengthen augmentation.
Epoch 4/20 | train_loss=0.727 | train_acc=94.9% | val_acc=95.97% | test_acc=97.94% | params=19202
Epoch 5/20 | train_loss=0.693 | train_acc=96.1% | val_acc=96.42% | test_acc=98.42% | params=19202
Epoch 6/20 | train_loss=0.663 | train_acc=97.0% | val_acc=97.77% | test_acc=98.97% | params=19202
Epoch 7/20 | train_loss=0.654 | train_acc=97.2% | val_acc=97.49% | test_acc=98.85% | params=19202
Epoch 8/20 | train_loss=0.647 | train_acc=97.3% | val_acc=97.63% | test_acc=98.90% | params=19202
Epoch 9/20 | train_loss=0.641 | train_acc=97.6% | val_acc=97.90% | test_acc=98.89% | params=19202
Epoch 10/20 | train_loss=0.638 | train_acc=97.6% | val_acc=98.29% | test_acc=99.26% | params=19202
Epoch 11/20 | train_loss=0.630 | train_acc=97.7% | val_acc=98.14% | test_acc=99.31% | params=19202
Epoch 12/20 | train_loss=0.627 | train_acc=97.9% | val_acc=98.15% | test_acc=99.25% | params=19202
Epoch 13/20 | train_loss=0.624 | train_acc=97.9% | val_acc=98.45% | test_acc=99.31% | params=19202
Epoch 14/20 | train_loss=0.621 | train_acc=98.1% | val_acc=98.29% | test_acc=99.22% | params=19202
Epoch 15/20 | train_loss=0.618 | train_acc=98.1% | val_acc=98.56% | test_acc=99.35% | params=19202
Epoch 16/20 | train_loss=0.616 | train_acc=98.1% | val_acc=98.38% | test_acc=99.39% | params=19202
Epoch 17/20 | train_loss=0.613 | train_acc=98.3% | val_acc=98.42% | test_acc=99.37% | params=19202
Epoch 18/20 | train_loss=0.613 | train_acc=98.2% | val_acc=98.47% | test_acc=99.36% | params=19202
Epoch 19/20 | train_loss=0.613 | train_acc=98.2% | val_acc=98.42% | test_acc=99.36% | params=19202
Epoch 20/20 | train_loss=0.611 | train_acc=98.3% | val_acc=98.52% | test_acc=99.39% | params=19202

Done.
