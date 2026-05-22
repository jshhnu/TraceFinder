import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, accuracy_score
from deep_forensic_net import DeepForensicClassifier


class ForensicPatchDataset(Dataset):
    """
    Optimized Dataset wrapper that streams raw image data from disk via memory-mapping.
    Forensic filtering is deferred to the GPU pipeline to maximize throughput.
    """

    def __init__(self, tensors, labels):
        self.tensors = tensors
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.tensors[idx].astype(np.float32)
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)  # Shape: (1, 512, 512)
        return torch.tensor(data, dtype=torch.float32), self.labels[idx]


def torch_wiener_filter(img_tensor, window_size=5):
    """
    GPU-accelerated 2D Wiener deconvolution filter implemented natively in PyTorch.
    Strips away macroscopic text edges, leaving pure hardware sensor grain.
    """
    padding = window_size // 2
    local_mean = F.avg_pool2d(img_tensor, kernel_size=window_size, stride=1, padding=padding)
    local_var = F.avg_pool2d(img_tensor ** 2, kernel_size=window_size, stride=1, padding=padding) - (local_mean ** 2)
    local_var = torch.clamp(local_var, min=1e-5)
    noise_variance = torch.mean(local_var)

    ratio = torch.clamp((local_var - noise_variance) / local_var, min=0.0)
    denoised = local_mean + ratio * (img_tensor - local_mean)
    noise_residual = img_tensor - denoised

    mean = torch.mean(noise_residual, dim=(2, 3), keepdim=True)
    std = torch.std(noise_residual, dim=(2, 3), keepdim=True) + 1e-6
    return (noise_residual - mean) / std


def run_deep_training_pipeline():
    # CRITICAL: Clear out lingering fragmented artifacts from the previous crash
    torch.cuda.empty_cache()

    print("Loading source data tensors via memory-mapping...")
    try:
        X_raw = np.load("X_final.npy", mmap_mode='r')
        y_raw = np.load("y_final.npy")
        print(f"Data successfully mapped. Dataset shape: {X_raw.shape}")
    except FileNotFoundError:
        print("Critical Error: 'X_final.npy' or 'y_final.npy' missing.")
        return

    total_samples = len(X_raw)
    groups = np.floor(np.arange(total_samples) / 4).astype(int)

    gkf = GroupKFold(n_splits=5)
    train_idx, test_idx = next(gkf.split(X_raw, y_raw, groups=groups))

    train_dataset = ForensicPatchDataset(X_raw[train_idx], y_raw[train_idx])
    test_dataset = ForensicPatchDataset(X_raw[test_idx], y_raw[test_idx])

    # SAFETY FIX: Reduced batch size to 4 to prevent 4GB VRAM overflow on laptop GPUs
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target execution hardware verified: {device}")

    model = DeepForensicClassifier(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    # Gradient accumulation configuration (4 steps * batch size 4 = virtual batch size 16)
    accumulation_steps = 4

    print("\nStarting GPU-accelerated deep training sequence (10 Epochs)...")
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()  # Initialize gradients outside the loop

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs_filtered = torch_wiener_filter(inputs, window_size=5)

            outputs = model(inputs_filtered)
            loss = criterion(outputs, targets)

            # Normalize the loss value based on our tracking interval
            loss = loss / accumulation_steps
            loss.backward()

            running_loss += loss.item() * inputs.size(0) * accumulation_steps

            # Update weights only after accumulating enough micro-batches
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            if (batch_idx + 1) % 40 == 0:
                print(
                    f"Epoch [{epoch + 1}/10] | Step [{batch_idx + 1}/{len(train_loader)}] | Loss: {loss.item() * accumulation_steps:.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"--> Epoch {epoch + 1} finalized. Aggregated Loss Indicator: {epoch_loss:.4f}\n")

    print("Initiating strict out-of-sample performance validation...")
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            inputs_filtered = torch_wiener_filter(inputs, window_size=5)
            outputs = model(inputs_filtered)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.numpy())

    final_acc = accuracy_score(all_targets, all_predictions) * 100
    print("\n" + "=" * 50)
    print(f"DEEP LEARNING PIPELINE METRIC STABILIZED: {final_acc:.2f}%")
    print("=" * 50)
    print("\nDetailed Forensic Classification Report:")
    print(classification_report(all_targets, all_predictions, zero_division=0))

    torch.save(model.state_dict(), "tracefinder_deep_production.pth")
    print("State dictionary successfully serialized as 'tracefinder_deep_production.pth'.")


if __name__ == "__main__":
    run_deep_training_pipeline()