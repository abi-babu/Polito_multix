import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import json
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import signal
import torch.nn.functional as F
from model.hydranet import HydraNet, HydraNetConfig
import model.stem
print(f"Loading stem.py from: {model.stem.__file__}")

# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.waveform_length = 1000
        self.freeze_stems = False
        self.pretrained_stems = True
        self.pretrained_branches = True
        self.pretrained_fusion = True
        self.fusion_type = 'hierarchical'
        self.batch_size = 2
        self.learning_rate = 0.00002
        self.num_global_rounds = 20          # <-- federated: global rounds
        self.local_epochs = 1                # <-- federated: local epochs per client
        self.train_ratio = 0.8
        self.resume_training = False
        self.num_clients = 3                 # <-- federated: number of clients

config = Config()
device = torch.device(config.device)
print(f"Using device: {device}")

FS = 250
SEG_LEN = 30 * FS
TARGET_LENGTH = config.waveform_length

# ============================================================
# DATASET
# ============================================================
class HydraNetDataset(Dataset):
    def __init__(self, paths_dict, waveform, indices):
        self.ecg_paths = paths_dict['ecg']
        self.eeg_paths = paths_dict['eeg']
        self.resp_paths = paths_dict['resp']
        self.waveform = waveform
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        segment_idx = self.indices[idx]
        ecg = self._load_image(self.ecg_paths[segment_idx])
        eeg = self._load_image(self.eeg_paths[segment_idx])
        resp = self._load_image(self.resp_paths[segment_idx])

        start = segment_idx * 7500
        end = start + 7500
        wav = self.waveform[start:end]
        wav = signal.resample(wav, 1000)
        wav = (wav - np.mean(wav)) / (np.std(wav) + 1e-8)

        return ecg, eeg, resp, torch.FloatTensor(wav)

    def _load_image(self, path):
        img = Image.open(path).convert('RGB')
        img = torch.FloatTensor(np.array(img)).permute(2, 0, 1) / 255.0
        return img

# ============================================================
# DATA LOADING
# ============================================================
print("\n" + "="*60)
print("LOADING DATA")
print("="*60)

paths_dict = {
    'ecg': [os.path.join("ECG_Spec", f"ecg_{i:03d}.png") for i in range(1, 181)],
    'eeg': [os.path.join("EEG_Heat", f"eeg_{i:03d}.png") for i in range(1, 181)],
    'resp': [os.path.join("Resp_RGB", f"resp_{i:03d}.png") for i in range(1, 181)]
}

with open("ground_truth_sleep.json", 'r') as f:
    gt_data = json.load(f)
    waveform = np.array(gt_data["respiration_waveform"], dtype=np.float32)

print(f"Loaded {len(paths_dict['ecg'])} frames per modality")
print(f"Waveform length: {len(waveform)}")

num_segments = len(waveform) // SEG_LEN
indices = list(range(num_segments))
print(f"Total segments: {num_segments}")

train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
print(f"Train segments: {len(train_idx)}, Test segments: {len(test_idx)}")

test_dataset = HydraNetDataset(paths_dict, waveform, test_idx)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
print(f"Test batches: {len(test_loader)}")

# ============================================================
# FEDERATED PARTITIONING
# ============================================================
# Simple IID-ish split: equal chunks of train_idx per client
train_idx = np.array(train_idx)
client_indices = np.array_split(train_idx, config.num_clients)

client_loaders = []
for cid, idxs in enumerate(client_indices):
    idxs = idxs.tolist()
    ds = HydraNetDataset(paths_dict, waveform, idxs)
    dl = DataLoader(ds, batch_size=config.batch_size, shuffle=True)
    client_loaders.append(dl)
    print(f"Client {cid}: {len(ds)} segments, {len(dl)} batches")

# ============================================================
# MODEL INITIALIZATION
# ============================================================
print("\n" + "="*60)
print("INITIALIZING HYDRANET")
print("="*60)

model = HydraNet(config).to(device)

start_round = 1
best_test_corr = -float('inf')

best_epoch_path = "checkpoints/epoch_50.pth"

if os.path.exists(best_epoch_path):
    print(f"Loading best model from {best_epoch_path}")
    checkpoint = torch.load(best_epoch_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model weights")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights")

    best_test_corr = 0.3395
    print(f"Best known correlation (pre-federated): {best_test_corr:.4f}")
else:
    print("Best epoch model not found. Starting from scratch.")

print("\nVerifying stem pretrained weights:")
for name, stem in model.stems.items():
    first_conv = stem.conv1.weight
    print(f"  {name}: std={first_conv.std().item():.4f}")

print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# LOSS FUNCTION
# ============================================================
class PhaseAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        pred_centered = pred - pred.mean(dim=1, keepdim=True)
        target_centered = target - target.mean(dim=1, keepdim=True)
        corr = (pred_centered * target_centered).sum(dim=1) / (
            torch.sqrt((pred_centered**2).sum(dim=1) * (target_centered**2).sum(dim=1)) + 1e-8
        )
        corr_loss = 1 - corr.mean()
        return mse_loss + 0.7 * corr_loss

criterion = PhaseAwareLoss()

# ============================================================
# FEDERATED UTILITIES
# ============================================================
def get_model_weights(model):
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}

def set_model_weights(model, weights):
    model.load_state_dict(weights)

def average_weights(weights_list):
    # FedAvg: simple mean of each parameter
    avg = {}
    for k in weights_list[0].keys():
        avg[k] = sum(w[k] for w in weights_list) / len(weights_list)
    return avg

def client_update(global_model, dataloader, lr, local_epochs):
    # Copy global model to client
    client_model = HydraNet(config).to(device)
    client_model.load_state_dict(global_model.state_dict())
    client_model.train()

    optimizer = optim.Adam(client_model.parameters(), lr=lr)

    for _ in range(local_epochs):
        for ecg, eeg, resp, target in dataloader:
            ecg, eeg, resp, target = ecg.to(device), eeg.to(device), resp.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = client_model(ecg_img=ecg, eeg_img=eeg, resp_img=resp)
            loss = criterion(outputs['waveform'], target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=0.5)
            optimizer.step()

    return get_model_weights(client_model)

def evaluate_global(model, test_loader):
    model.eval()
    raw_preds, raw_targets = [], []

    with torch.no_grad():
        for ecg, eeg, resp, target in test_loader:
            ecg, eeg, resp, target = ecg.to(device), eeg.to(device), resp.to(device), target.to(device)
            outputs = model(ecg_img=ecg, eeg_img=eeg, resp_img=resp)
            for i in range(len(outputs['waveform'])):
                raw_preds.append(outputs['waveform'][i].cpu().numpy())
                raw_targets.append(target[i].cpu().numpy())

    raw_corrs = []
    for p, t in zip(raw_preds, raw_targets):
        corr = np.corrcoef(p, t)[0, 1]
        if not np.isnan(corr):
            raw_corrs.append(corr)

    if len(raw_corrs) == 0:
        return 0.0, 0.0, 0.0, None, None

    best_idx = np.argmax(raw_corrs)
    best_corr = raw_corrs[best_idx]
    mean_corr = np.mean(raw_corrs)
    std_corr = np.std(raw_corrs)

    return best_corr, mean_corr, std_corr, raw_preds[best_idx], raw_targets[best_idx]

# ============================================================
# FEDERATED TRAINING LOOP
# ============================================================
print("\n" + "="*60)
print("FEDERATED TRAINING (FedAvg)")
print("="*60)

for rnd in range(start_round, config.num_global_rounds + 1):
    print(f"\n--- Global Round {rnd}/{config.num_global_rounds} ---")

    global_weights = get_model_weights(model)
    client_weights_list = []

    # Each client trains locally
    for cid, dl in enumerate(client_loaders):
        print(f"  Client {cid}: local training...")
        w = client_update(model, dl, lr=config.learning_rate, local_epochs=config.local_epochs)
        client_weights_list.append(w)

    # Server aggregates
    new_global_weights = average_weights(client_weights_list)
    set_model_weights(model, new_global_weights)

    # Evaluate on global test set
    best_corr, mean_corr, std_corr, best_pred, best_target = evaluate_global(model, test_loader)
    print(f"Round {rnd} | Best Corr: {best_corr:.4f} | Mean: {mean_corr:.4f} ± {std_corr:.4f}")

    if mean_corr > best_test_corr:
        best_test_corr = mean_corr
        os.makedirs("checkpoints_fed", exist_ok=True)
        save_path = os.path.join("checkpoints_fed", f"global_round_{rnd}.pth")
        torch.save({'model_state_dict': model.state_dict(), 'round': rnd}, save_path)
        print(f"New best model saved to {save_path}")

# ============================================================
# FINAL EVALUATION & PLOT
# ============================================================
print("\n" + "="*60)
print("FINAL EVALUATION AFTER FEDERATED TRAINING")
print("="*60)

best_corr, mean_corr, std_corr, best_pred, best_target = evaluate_global(model, test_loader)
print(f"\nFinal Best Corr: {best_corr:.4f}")
print(f"Final Mean Corr: {mean_corr:.4f} ± {std_corr:.4f}")

if best_pred is not None:
    plt.figure(figsize=(14, 6))
    time_axis = np.linspace(0, 30, 1000)
    plt.plot(time_axis, best_target, 'b-', linewidth=2.5, label='Ground Truth')
    plt.plot(time_axis, best_pred, 'r--', linewidth=2.5, label=f'Prediction (corr={best_corr:.3f})')
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Normalized Amplitude', fontsize=14)
    plt.title('Best Raw Prediction vs Ground Truth (Federated)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 30])
    plt.tight_layout()
    plt.savefig('best_raw_prediction_federated.png', dpi=300)
    plt.show()
    print(f"\nPlot saved to: best_raw_prediction_federated.png")
print("="*60)
