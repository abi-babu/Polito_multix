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
        self.learning_rate = 0.00002  # MUCH lower for fine-tuning
        self.num_epochs = 80  # Fewer epochs for fine-tuning
        self.train_ratio = 0.8
        self.resume_training = False

config = Config()
device = torch.device(config.device)
print(f"Using device: {device}")

# Constants
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

train_dataset = HydraNetDataset(paths_dict, waveform, train_idx)
test_dataset = HydraNetDataset(paths_dict, waveform, test_idx)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# ============================================================
# MODEL INITIALIZATION
# ============================================================
print("\n" + "="*60)
print("INITIALIZING HYDRANET")
print("="*60)

model = HydraNet(config).to(device)

start_epoch = 1
best_test_corr = -float('inf')
train_losses = []
test_corrs = []

# ============================================================
# LOAD THE BEST EPOCH MODEL (epoch_50.pth)
# ============================================================
best_epoch_path = "checkpoints/epoch_50.pth" 

if os.path.exists(best_epoch_path):
    print(f"📂 Loading best model from {best_epoch_path}")
    checkpoint = torch.load(best_epoch_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model weights")
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"Model from epoch {checkpoint['epoch']}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights")

    # Set best correlation to your known value
    best_test_corr = 0.3395
    print(f"Best known correlation: {best_test_corr:.4f}")
else:
    print("Best epoch model not found. Starting fresh.")

# Verify stems
print("\nVerifying stem pretrained weights:")
for name, stem in model.stems.items():
    first_conv = stem.conv1.weight
    print(f"  {name}: std={first_conv.std().item():.4f}")

print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# LOSS FUNCTION - WORKING VERSION
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

# ============================================================
# OPTIONAL: SKIP TRAINING AND JUST EVALUATE
# ============================================================
SKIP_TRAINING = True # Set to False if you want to fine-tune

if not SKIP_TRAINING:
    print("\n" + "="*60)
    print("FINE-TUNING WITH LOW LR")
    print("="*60)
    
    criterion = PhaseAwareLoss()
    # Use VERY low learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
    for epoch in range(start_epoch, config.num_epochs + 1):
        model.train()
        train_loss = 0
        
        for ecg, eeg, resp, target in train_loader:
            ecg, eeg, resp, target = ecg.to(device), eeg.to(device), resp.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(ecg_img=ecg, eeg_img=eeg, resp_img=resp)
            loss = criterion(outputs['waveform'], target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item()
        
        if epoch % 5 == 0:
            model.eval()
            test_corrs_epoch = []
            
            with torch.no_grad():
                for ecg, eeg, resp, target in test_loader:
                    ecg, eeg, resp, target = ecg.to(device), eeg.to(device), resp.to(device), target.to(device)
                    outputs = model(ecg_img=ecg, eeg_img=eeg, resp_img=resp)
                    pred = outputs['waveform']
                    for i in range(len(pred)):
                        corr = np.corrcoef(pred[i].cpu().numpy(), target[i].cpu().numpy())[0, 1]
                        if not np.isnan(corr):
                            test_corrs_epoch.append(corr)
            
            avg_test_corr = np.mean(test_corrs_epoch) if test_corrs_epoch else 0
            print(f"Epoch {epoch:3d} | Loss: {train_loss/len(train_loader):.4f} | Test Corr: {avg_test_corr:.4f}")
else:
    print("\n" + "="*60)
    print("SKIPPING TRAINING - USING EPOCH 50 MODEL")
    print("="*60)

# ============================================================
# FINAL EVALUATION - JUST RAW PREDICTION PLOT
# ============================================================
print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)

model.eval()

# Get predictions
raw_preds, raw_targets = [], []

with torch.no_grad():
    for ecg, eeg, resp, target in test_loader:
        ecg, eeg, resp, target = ecg.to(device), eeg.to(device), resp.to(device), target.to(device)
        outputs = model(ecg_img=ecg, eeg_img=eeg, resp_img=resp)
        for i in range(len(outputs['waveform'])):
            raw_preds.append(outputs['waveform'][i].cpu().numpy())
            raw_targets.append(target[i].cpu().numpy())

# Calculate correlations
raw_corrs = []
for p, t in zip(raw_preds, raw_targets):
    corr = np.corrcoef(p, t)[0, 1]
    if not np.isnan(corr):
        raw_corrs.append(corr)

best_idx = np.argmax(raw_corrs)
best_corr = raw_corrs[best_idx]

print(f"\nBest Raw Correlation: {best_corr:.4f}")
print(f"Mean Correlation: {np.mean(raw_corrs):.4f} ± {np.std(raw_corrs):.4f}")

# Plot
plt.figure(figsize=(14, 6))
time_axis = np.linspace(0, 30, 1000)
plt.plot(time_axis, raw_targets[best_idx], 'b-', linewidth=2.5, label='Ground Truth')
plt.plot(time_axis, raw_preds[best_idx], 'r--', linewidth=2.5, label=f'Prediction (corr={best_corr:.3f})')
plt.xlabel('Time (seconds)', fontsize=14)
plt.ylabel('Normalized Amplitude', fontsize=14)
plt.title('Best Raw Prediction vs Ground Truth', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim([0, 30])
plt.tight_layout()
plt.savefig('best_raw_prediction.png', dpi=300)
plt.show()

print(f"\nPlot saved to: best_raw_prediction.png")
print("="*60)
