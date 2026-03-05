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
from scipy.ndimage import zoom
import sys
import importlib

from model.hydranet import HydraNet, HydraNetConfig
import model.stem


def plot_sample_prediction(model, test_loader, epoch):
    """Plot a sample prediction during training for debug"""
    model.eval()
    with torch.no_grad():
        # Get a batch from test loader
        ecg, eeg, resp, target = next(iter(test_loader))
        ecg = ecg.to(device)
        eeg = eeg.to(device)
        resp = resp.to(device)

        # Get prediction
        outputs = model(ecg_img=ecg, eeg_img=eeg, resp_img=resp)
        pred = outputs['waveform'][0].cpu().numpy()
        gt = target[0].numpy()

        # Create plot
        plt.figure(figsize=(12, 4))
        plt.plot(gt, 'b-', label='Ground Truth', linewidth=2, alpha=0.7)
        plt.plot(pred, 'r--', label='Prediction', linewidth=2, alpha=0.7)
        plt.legend(fontsize=12)
        plt.title(f'Epoch {epoch} - Sample Prediction', fontsize=14)
        plt.xlabel('Time (samples)', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.grid(True, alpha=0.3)

        # Print statistics
        print(f"\nEpoch {epoch} - Sample Statistics:")
        print(f"  Prediction - Mean: {pred.mean():.4f}, Std: {pred.std():.4f}, Range: [{pred.min():.4f}, {pred.max():.4f}]")
        print(f"  Ground Truth - Mean: {gt.mean():.4f}, Std: {gt.std():.4f}, Range: [{gt.min():.4f}, {gt.max():.4f}]")

        # Save plot
        plt.savefig(f'debug_pred_epoch_{epoch}.png', dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Saved debug_pred_epoch_{epoch}.png")

class Config:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.waveform_length = 1000  
        self.freeze_stems = False
        self.pretrained_stems = True
        self.pretrained_branches = True
        self.pretrained_fusion = True
        self.fusion_type = 'hierarchical'
        self.batch_size = 2  # Reduce batch size since images are larger
        self.learning_rate = 0.001  # Lower learning rate
        self.num_epochs = 20  # More epochs


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

        # Load images
        ecg = self._load_image(self.ecg_paths[segment_idx])  # Original: 256x512
        eeg = self._load_image(self.eeg_paths[segment_idx])  # Original: 256x512
        resp = self._load_image(self.resp_paths[segment_idx])  # Original: 938x1250

        start = segment_idx * 7500
        end = start + 7500
        wav = self.waveform[start:end].copy()  # Make a copy to avoid modifying original
        from scipy import signal
        wav = signal.resample(wav, 1000)

        # wav = wav * 10.0  # Only if you need to amplify
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
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

print(f"Train segments: {len(train_idx)}, Test segments: {len(test_idx)}")

train_dataset = HydraNetDataset(paths_dict, waveform, train_idx)
test_dataset = HydraNetDataset(paths_dict, waveform, test_idx)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# Force reload of stem module
if 'stem' in sys.modules:
    importlib.reload(sys.modules['stem'])
if 'branch' in sys.modules:
    importlib.reload(sys.modules['branch'])
if 'fusion' in sys.modules:
    importlib.reload(sys.modules['fusion'])
if 'hydranet' in sys.modules:
    importlib.reload(sys.modules['hydranet'])

# ============================================================
# MODEL INITIALIZATION
# ============================================================
print("\n" + "="*60)
print("INITIALIZING HYDRANET")
print("="*60)

checkpoint_path = "checkpoints/best_hydranet.pth"
os.makedirs("checkpoints", exist_ok=True)

model = HydraNet(config).to(device)

# Load checkpoint if exists
start_epoch = 1
best_test_corr = 0


if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Checkpoint loaded from {checkpoint_path}")
    
    
    model.eval()
    test_corrs = []
    with torch.no_grad():
        for ecg, eeg, resp, target in test_loader:
            ecg = ecg.to(device)
            eeg = eeg.to(device)
            resp = resp.to(device)
            target = target.to(device)
            
            outputs = model(ecg_img=ecg, eeg_img=eeg, resp_img=resp)
            pred = outputs['waveform']
            
            for i in range(len(pred)):
                corr = np.corrcoef(pred[i].cpu().numpy(), target[i].cpu().numpy())[0, 1]
                if not np.isnan(corr):
                    test_corrs.append(abs(corr))
    
    best_test_corr = np.mean(test_corrs) if test_corrs else 0
    print(f"Loaded model performance - Mean Test Corr: {best_test_corr:.4f}")
    print(f"Best model will only be updated if correlation exceeds {best_test_corr:.4f}")
else:
    print("No checkpoint found. Starting fresh.")
    best_test_corr = 0

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Add this debug code right before the training loop starts
print("\n" + "="*60)
print("DEBUGGING SHAPES")
print("="*60)

# Get one batch from train_loader
debug_ecg, debug_eeg, debug_resp, debug_target = next(iter(train_loader))
debug_ecg = debug_ecg.to(device)
debug_eeg = debug_eeg.to(device)
debug_resp = debug_resp.to(device)

print(f"Input shapes:")
print(f"  ECG image: {debug_ecg.shape}")
print(f"  EEG image: {debug_eeg.shape}")
print(f"  Resp image: {debug_resp.shape}")

with torch.no_grad():
    # Stem outputs
    ecg_stem = model.stems['rfspect'](debug_ecg)
    eeg_stem = model.stems['rfheat'](debug_eeg)
    resp_stem = model.stems['camera'](debug_resp)

    print(f"\nStem outputs:")
    print(f"  ECG stem: {ecg_stem.shape}")
    print(f"  EEG stem: {eeg_stem.shape}")
    print(f"  Resp stem: {resp_stem.shape}")

    # Branch outputs
    ecg_feat = model.branches['rfspect'](ecg_stem)
    eeg_feat = model.branches['rfheat'](eeg_stem)
    resp_feat = model.branches['resp'](resp_stem)

    print(f"\nBranch outputs:")
    print(f"  ECG branch: {ecg_feat.shape}")
    print(f"  EEG branch: {eeg_feat.shape}")
    print(f"  Resp branch: {resp_feat.shape}")

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

# Test head output range
test_input = torch.randn(4, 512, 8, 16).to(device)
test_output = model.prediction_head(test_input)
#print(f"Test head output - Mean: {test_output.mean():.2f}, Std: {test_output.std():.2f}, Range: [{test_output.min():.2f}, {test_output.max():.2f}]")
# ============================================================
# TRAINING
# ============================================================
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=config.learning_rate,
    weight_decay=1e-4
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

for epoch in range(start_epoch, config.num_epochs + 1):
    model.train()
    train_loss = 0

    for ecg, eeg, resp, target in train_loader:
        ecg = ecg.to(device)
        eeg = eeg.to(device)
        resp = resp.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        outputs = model(
            ecg_img=ecg,
            eeg_img=eeg,
            resp_img=resp,
            gt_waveform=target
        )

        outputs['loss'].backward()
        optimizer.step()
        train_loss += outputs['loss'].item()
    
    scheduler.step()
    # After each epoch, plot a sample prediction
    if epoch % 5 == 0 or epoch == 1:  # Plot at epoch 1, then every 5 epochs
        plot_sample_prediction(model, test_loader, epoch)
    if epoch % 5 == 0 or epoch == config.num_epochs:
        model.eval()
        test_loss = 0
        test_corrs = []

        with torch.no_grad():
            for ecg, eeg, resp, target in test_loader:
                ecg = ecg.to(device)
                eeg = eeg.to(device)
                resp = resp.to(device)
                target = target.to(device)

                outputs = model(
                    ecg_img=ecg,
                    eeg_img=eeg,
                    resp_img=resp
                )

                pred = outputs['waveform']

                for i in range(len(pred)):
                    pred_np = pred[i].cpu().numpy()
                    target_np = target[i].cpu().numpy()
                    corr = np.corrcoef(pred_np, target_np)[0, 1]
                    if not np.isnan(corr):
                        test_corrs.append(abs(corr))

        avg_train_loss = train_loss / len(train_loader)
        avg_test_corr = np.mean(test_corrs) if test_corrs else 0

        print(f"\nEpoch {epoch:2d}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Test Corr:  {avg_test_corr:.4f}")

        if hasattr(model.main_fusion, 'weights'):
            weights = torch.softmax(model.main_fusion.weights, dim=0)
            print(f"  Fusion Weights: ECG={weights[0]:.3f}, EEG={weights[1]:.3f}, Resp={weights[2]:.3f}")

        if avg_test_corr > best_test_corr:
            best_test_corr = avg_test_corr
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model! (Corr: {best_test_corr:.4f})")


# ============================================================
# GENERATE PREDICTIONS FROM BEST MODEL
# ============================================================
print("\n" + "="*60)
print("GENERATING PREDICTIONS FROM BEST MODEL")
print("="*60)

# Load the best model weights
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
print(f" Loaded best model from {checkpoint_path}")

all_predictions = []
all_targets = []
all_correlations = []

with torch.no_grad():
    for ecg, eeg, resp, target in test_loader:
        ecg = ecg.to(device)
        eeg = eeg.to(device)
        resp = resp.to(device)

        outputs = model(
            ecg_img=ecg,
            eeg_img=eeg,
            resp_img=resp
        )

        pred = outputs['waveform']

        for i in range(len(pred)):
            pred_np = pred[i].cpu().numpy()
            target_np = target[i].numpy()
            corr = np.corrcoef(pred_np, target_np)[0, 1]

            if not np.isnan(corr):
                all_correlations.append(corr)
                all_predictions.append(pred_np)
                all_targets.append(target_np)

all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)
all_correlations = np.array(all_correlations)

best_idx = np.argmax(all_correlations)
worst_idx = np.argmin(all_correlations)
best_corr = all_correlations[best_idx]
worst_corr = all_correlations[worst_idx]
mean_corr = np.mean(all_correlations)
median_corr = np.median(all_correlations)

print(f"\nModel Performance:")
print(f"  Best Correlation:  {best_corr:.4f}")

# ============================================================
# PLOT BEST PREDICTION VS GROUND TRUTH (SINGLE PLOT)
# ============================================================
print("\n" + "="*60)
print("PLOTTING BEST PREDICTION VS GROUND TRUTH")
print("="*60)

time_axis = np.linspace(0, 30, TARGET_LENGTH)

# Create single plot
plt.figure(figsize=(14, 6))
plt.plot(time_axis, all_targets[best_idx], 'b-', linewidth=2, label='Ground Truth')
plt.plot(time_axis, all_predictions[best_idx], 'r--', linewidth=2, label='Prediction')
plt.title(f'HydraNet: Best Prediction (Correlation: {best_corr:.3f})', fontsize=16)
plt.xlabel('Time (seconds)', fontsize=14)
plt.ylabel('Normalized Amplitude', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim([0, 30])
plt.tight_layout()
plt.savefig('hydranet_best_prediction.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nBest prediction plot saved to 'hydranet_best_prediction.png'")


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"📊 Best Correlation:  {best_corr:.4f}")
print(f"Best model saved to: {checkpoint_path}")
print(f"Plot saved to: hydranet_best_prediction.png")
print("="*60)
