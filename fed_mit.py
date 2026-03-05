import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from PIL import Image
import json
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import signal
import sys
import importlib
import copy

from model.hydranet import HydraNet, HydraNetConfig
import model.stem

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
        self.learning_rate = 0.001
        self.num_clients = 3
        self.num_rounds = 20  # Number of federated rounds
        self.local_epochs = 2  # Number of local epochs per client per round
        self.fraction_fit = 1.0  # Fraction of clients used per round


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
        ecg = self._load_image(self.ecg_paths[segment_idx])
        eeg = self._load_image(self.eeg_paths[segment_idx])
        resp = self._load_image(self.resp_paths[segment_idx])

        start = segment_idx * 7500
        end = start + 7500
        wav = self.waveform[start:end].copy()
        from scipy import signal
        wav = signal.resample(wav, 1000)

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

# Create central test dataset (same for all clients)
test_dataset = HydraNetDataset(paths_dict, waveform, test_idx)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# ============================================================
# FEDERATED CLIENT DATA PARTITIONING
# ============================================================
print("\n" + "="*60)
print("PARTITIONING DATA FOR FEDERATED CLIENTS")
print("="*60)

# Split training indices among clients
client_indices = np.array_split(train_idx, config.num_clients)

client_datasets = []
client_loaders = []

for client_id in range(config.num_clients):
    client_dataset = HydraNetDataset(paths_dict, waveform, client_indices[client_id].tolist())
    client_datasets.append(client_dataset)
    
    client_loader = DataLoader(
        client_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        drop_last=True
    )
    client_loaders.append(client_loader)
    
    print(f"Client {client_id}: {len(client_dataset)} segments, {len(client_loader)} batches")


# ============================================================
# MODEL INITIALIZATION
# ============================================================
print("\n" + "="*60)
print("INITIALIZING GLOBAL MODEL")
print("="*60)

checkpoint_path = "checkpoints/federated_best_model.pth"
os.makedirs("checkpoints", exist_ok=True)

global_model = HydraNet(config).to(device)

# Load checkpoint if exists
start_round = 1
best_test_corr = 0

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    global_model.load_state_dict(checkpoint['model_state_dict'])
    best_test_corr = checkpoint.get('best_corr', 0)
    start_round = checkpoint.get('round', 1) + 1
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Best correlation so far: {best_test_corr:.4f}")
    print(f"Resuming from round {start_round}")
else:
    print("No checkpoint found. Starting fresh.")

total_params = sum(p.numel() for p in global_model.parameters())
print(f"Total parameters: {total_params:,}")


# ============================================================
# FEDERATED LEARNING UTILITIES
# ============================================================
def get_model_weights(model):
    """Extract model weights as state dict"""
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def set_model_weights(model, weights):
    """Set model weights from state dict"""
    model.load_state_dict(weights)


def federated_averaging(weights_list):
    """FedAvg: average model weights from multiple clients"""
    avg_weights = {}
    for key in weights_list[0].keys():
        avg_weights[key] = sum(w[key] for w in weights_list) / len(weights_list)
    return avg_weights


def client_local_train(global_model, client_loader, local_epochs, lr):
    """Train client model locally"""
    # Create a copy of the global model for this client
    client_model = HydraNet(config).to(device)
    client_model.load_state_dict(global_model.state_dict())
    client_model.train()
    
    optimizer = optim.AdamW(client_model.parameters(), lr=lr, weight_decay=1e-4)
    
    for epoch in range(local_epochs):
        epoch_loss = 0
        for ecg, eeg, resp, target in client_loader:
            ecg = ecg.to(device)
            eeg = eeg.to(device)
            resp = resp.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            outputs = client_model(
                ecg_img=ecg,
                eeg_img=eeg,
                resp_img=resp,
                gt_waveform=target
            )
            
            outputs['loss'].backward()
            torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += outputs['loss'].item()
    
    return get_model_weights(client_model)


def evaluate_model(model, test_loader):
    """Evaluate model on test set"""
    model.eval()
    all_correlations = []
    all_predictions = []
    all_targets = []
    
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
    
    all_correlations = np.array(all_correlations)
    
    if len(all_correlations) == 0:
        return 0, 0, None, None
    
    best_idx = np.argmax(all_correlations)
    best_corr = all_correlations[best_idx]
    mean_corr = np.mean(all_correlations)
    
    return mean_corr, best_corr, all_predictions[best_idx] if len(all_predictions) > 0 else None, all_targets[best_idx] if len(all_targets) > 0 else None


# ============================================================
# FEDERATED TRAINING LOOP
# ============================================================
print("\n" + "="*60)
print("STARTING FEDERATED TRAINING")
print("="*60)
print(f"Number of clients: {config.num_clients}")
print(f"Number of rounds: {config.num_rounds}")
print(f"Local epochs per round: {config.local_epochs}")
print("="*60)

for round_num in range(start_round, config.num_rounds + 1):
    print(f"\n--- Federated Round {round_num}/{config.num_rounds} ---")
    
    # Select clients for this round
    num_clients_to_use = max(1, int(config.fraction_fit * config.num_clients))
    selected_clients = np.random.choice(range(config.num_clients), num_clients_to_use, replace=False)
    
    print(f"Selected clients: {selected_clients}")
    
    # Collect client updates
    client_weights = []
    
    for client_id in selected_clients:
        print(f"  Training client {client_id}...")
        client_loader = client_loaders[client_id]
        
        # Local training
        updated_weights = client_local_train(
            global_model, 
            client_loader, 
            config.local_epochs, 
            config.learning_rate
        )
        client_weights.append(updated_weights)
    
    # Federated averaging
    print("  Aggregating client updates...")
    avg_weights = federated_averaging(client_weights)
    set_model_weights(global_model, avg_weights)
    
    # Evaluate global model
    mean_corr, best_corr, best_pred, best_target = evaluate_model(global_model, test_loader)
    print(f"Round {round_num} - Mean Test Corr: {mean_corr:.4f}, Best Test Corr: {best_corr:.4f}")
    
    # Save best model
    if mean_corr > best_test_corr:
        best_test_corr = mean_corr
        torch.save({
            'round': round_num,
            'model_state_dict': global_model.state_dict(),
            'best_corr': best_test_corr,
            'mean_corr': mean_corr
        }, checkpoint_path)
        print(f"  New best model saved! (Mean Corr: {best_test_corr:.4f})")
    
    # Periodic debug plot every 5 rounds
    if round_num % 5 == 0 or round_num == 1:
        # Create a simple debug prediction plot
        global_model.eval()
        with torch.no_grad():
            ecg, eeg, resp, target = next(iter(test_loader))
            ecg = ecg.to(device)
            eeg = eeg.to(device)
            resp = resp.to(device)
            
            outputs = global_model(ecg_img=ecg, eeg_img=eeg, resp_img=resp)
            pred_sample = outputs['waveform'][0].cpu().numpy()
            target_sample = target[0].numpy()
            
            plt.figure(figsize=(12, 4))
            plt.plot(target_sample, 'b-', label='Ground Truth', alpha=0.7)
            plt.plot(pred_sample, 'r--', label='Prediction', alpha=0.7)
            plt.legend()
            plt.title(f'Federated Round {round_num} - Sample Prediction')
            plt.xlabel('Time (samples)')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'federated_round_{round_num}.png', dpi=100, bbox_inches='tight')
            plt.close()
            print(f"  Saved federated_round_{round_num}.png")


# ============================================================
# FINAL EVALUATION WITH BEST MODEL
# ============================================================
print("\n" + "="*60)
print("FINAL EVALUATION WITH BEST FEDERATED MODEL")
print("="*60)

# Load best model
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    global_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from round {checkpoint['round']} with correlation {checkpoint['best_corr']:.4f}")

# Generate predictions with best model
all_predictions = []
all_targets = []
all_correlations = []

global_model.eval()
with torch.no_grad():
    for ecg, eeg, resp, target in test_loader:
        ecg = ecg.to(device)
        eeg = eeg.to(device)
        resp = resp.to(device)

        outputs = global_model(
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

all_correlations = np.array(all_correlations)
best_idx = np.argmax(all_correlations)
best_corr = all_correlations[best_idx]
mean_corr = np.mean(all_correlations)
median_corr = np.median(all_correlations)
std_corr = np.std(all_correlations)

print(f"\nFinal Model Performance:")
print(f"  Best Correlation:  {best_corr:.4f}")
print(f"  Mean Correlation:  {mean_corr:.4f} ± {std_corr:.4f}")
print(f"  Median Correlation: {median_corr:.4f}")

# ============================================================
# PLOT BEST PREDICTION
# ============================================================
print("\n" + "="*60)
print("PLOTTING BEST PREDICTION VS GROUND TRUTH")
print("="*60)

time_axis = np.linspace(0, 30, TARGET_LENGTH)

plt.figure(figsize=(14, 6))
plt.plot(time_axis, all_targets[best_idx], 'b-', linewidth=2, label='Ground Truth')
plt.plot(time_axis, all_predictions[best_idx], 'r--', linewidth=2, label=f'Prediction (corr={best_corr:.3f})')
plt.title(f'Federated HydraNet: Best Prediction', fontsize=16)
plt.xlabel('Time (seconds)', fontsize=14)
plt.ylabel('Normalized Amplitude', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim([0, 30])
plt.tight_layout()
plt.savefig('federated_best_prediction.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nBest prediction plot saved to 'federated_best_prediction.png'")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("FEDERATED LEARNING SUMMARY")
print("="*60)
print(f"Total clients: {config.num_clients}")
print(f"Total rounds completed: {config.num_rounds}")
print(f"Local epochs per round: {config.local_epochs}")
print(f"Best correlation achieved: {best_corr:.4f}")
print(f"Mean correlation: {mean_corr:.4f} ± {std_corr:.4f}")
print(f"Best model saved to: {checkpoint_path}")
print(f"Plot saved to: federated_best_prediction.png")
print("="*60)
