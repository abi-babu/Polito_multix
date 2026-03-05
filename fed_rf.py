import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from PIL import Image
import json
import os
import matplotlib.pyplot as plt
import copy
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks, butter, filtfilt
from pathlib import Path
import random

from model.hydranet import HydraNet, HydraNetConfig


def estimate_bpm(waveform, fs=150):
    """Estimate BPM from waveform with better peak detection"""
    waveform_np = waveform.squeeze().cpu().numpy()
    
    nyquist = fs / 2
    low = 0.1 / nyquist
    high = 0.8 / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, waveform_np)
    
    peaks, properties = find_peaks(
        filtered, 
        distance=fs//3,
        prominence=0.1,
        width=5
    )
    
    if len(peaks) > 1:
        peak_intervals = np.diff(peaks) / fs
        mean_interval = np.mean(peak_intervals)
        bpm = 60 / mean_interval
    else:
        bpm = 0.0
    
    return bpm, len(peaks)


def calculate_bpm_accuracy(pred_bpm, true_bpm):
    """Calculate BPM accuracy metrics"""
    if true_bpm == 0 or pred_bpm == 0:
        return 0.0, 100.0
    
    abs_error = abs(pred_bpm - true_bpm)
    rel_error = (abs_error / true_bpm) * 100
    accuracy = max(0, 100 - rel_error)
    
    return abs_error, accuracy


class Config:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.waveform_length = 8998
        self.freeze_stems = False
        self.pretrained_stems = True
        self.pretrained_branches = True
        self.fusion_type = 'hierarchical'
        self.batch_size = 4
        self.learning_rate = 0.001
        self.num_rounds = 20
        self.num_local_epochs = 3
        self.checkpoint_dir = 'federated_checkpoints_rf'
        self.resp_fs = 150
        self.num_clients = 2
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)


# ============================================================
# DATASET
# ============================================================
class RFFusionDataset(Dataset):
    def __init__(self, data_dir, segment_ids, waveform_length=8998, augment=False):
        self.data_dir = Path(data_dir)
        self.segment_ids = segment_ids
        self.waveform_length = waveform_length
        self.augment = augment
        
        with open(self.data_dir / 'rf_ground_truth.json', 'r') as f:
            self.all_data = json.load(f)
        
        self.samples_dict = {}
        for sample in self.all_data['samples']:
            sample_id = str(sample['sample_id'])
            self.samples_dict[sample_id] = sample
    
    def __len__(self):
        return len(self.segment_ids)
    
    def _load_image(self, path, augment=False):
        img = Image.open(path).convert('RGB')
        
        if augment:
            if np.random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.random() > 0.5:
                angle = np.random.uniform(-5, 5)
                img = img.rotate(angle, expand=False, fillcolor=0)
        
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img
    
    def _preprocess_waveform(self, waveform):
        waveform = np.array(waveform, dtype=np.float32)
        waveform = (waveform - np.mean(waveform)) / (np.std(waveform) + 1e-6)
        
        if len(waveform) > self.waveform_length:
            if self.augment:
                start = np.random.randint(0, len(waveform) - self.waveform_length)
            else:
                start = (len(waveform) - self.waveform_length) // 2
            waveform = waveform[start:start + self.waveform_length]
        elif len(waveform) < self.waveform_length:
            waveform = np.pad(waveform, (0, self.waveform_length - len(waveform)), 'constant')
        
        return torch.from_numpy(waveform)
    
    def __getitem__(self, idx):
        segment_id = self.segment_ids[idx]
        segment_data = self.samples_dict[segment_id]
        
        heatmap_path = self.data_dir / 'rf_heatmaps' / f"heatmap_{segment_id}.png"
        spec_path = self.data_dir / 'rf_spec' / f"spectrogram_{segment_id}.png"
        
        heatmap = self._load_image(heatmap_path, augment=self.augment)
        spectrogram = self._load_image(spec_path, augment=self.augment)
        waveform = self._preprocess_waveform(segment_data['ground_truth'])
        
        return {
            'ecg_img': spectrogram,
            'eeg_img': heatmap,
            'waveform': waveform,
            'segment_id': segment_id
        }


def create_dummy_resp_input(batch_size, device):
    """Create a dummy input for respiration modality"""
    dummy = torch.zeros(batch_size, 3, 224, 224).to(device)
    return dummy


# ============================================================
# FEDERATED LEARNING CLIENT
# ============================================================
class FederatedClient:
    def __init__(self, client_id, dataset, config, device):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.device = device
        self.dataloader = DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=True
        )
    
    def local_train(self, global_model, local_epochs):
        """Train on local data"""
        # Create a copy of the global model
        local_model = copy.deepcopy(global_model)
        local_model.train()
        
        # Optimizer for local training
        optimizer = optim.AdamW([
            {'params': local_model.stems.parameters(), 'lr': self.config.learning_rate * 0.1},
            {'params': local_model.branches.parameters(), 'lr': self.config.learning_rate},
            {'params': local_model.main_fusion.parameters(), 'lr': self.config.learning_rate},
            {'params': local_model.prediction_head.parameters(), 'lr': self.config.learning_rate * 2}
        ], weight_decay=1e-4)
        
        # Local training loop
        for epoch in range(local_epochs):
            epoch_loss = 0
            for batch in self.dataloader:
                ecg = batch['ecg_img'].to(self.device)
                eeg = batch['eeg_img'].to(self.device)
                target = batch['waveform'].to(self.device)
                
                dummy_resp = create_dummy_resp_input(ecg.shape[0], self.device)
                
                optimizer.zero_grad()
                outputs = local_model(
                    ecg_img=ecg,
                    eeg_img=eeg,
                    resp_img=dummy_resp,
                    gt_waveform=target
                )
                outputs['loss'].backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += outputs['loss'].item()
            
            avg_loss = epoch_loss / len(self.dataloader)
            print(f"    Client {self.client_id} - Local Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        return local_model.state_dict(), len(self.dataset)


# ============================================================
# FIXED FEDERATED AVERAGING
# ============================================================
def federated_averaging(global_model, client_updates, client_sizes):
    """Aggregate client updates using FedAvg - Fixed for integer parameters"""
    global_dict = global_model.state_dict()
    
    total_samples = sum(client_sizes)
    
    # Initialize aggregated weights (as float)
    avg_dict = {}
    for key in global_dict.keys():
        # Check if parameter is integer type (like batch norm running stats)
        if global_dict[key].dtype in [torch.int64, torch.int32, torch.long]:
            # For integer parameters, just take from first client (they should be identical)
            avg_dict[key] = client_updates[0][key].clone()
        else:
            # For float parameters, do weighted average
            avg_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
    
    # Weighted average of client updates (only for float parameters)
    for client_update, client_size in zip(client_updates, client_sizes):
        weight = client_size / total_samples
        for key in avg_dict.keys():
            if global_dict[key].dtype not in [torch.int64, torch.int32, torch.long]:
                avg_dict[key] += weight * client_update[key].float()
    
    # Convert back to original dtype if needed
    for key in global_dict.keys():
        if global_dict[key].dtype in [torch.int64, torch.int32, torch.long]:
            # Keep as is
            global_dict[key] = avg_dict[key]
        else:
            # Convert back to original dtype
            global_dict[key] = avg_dict[key].to(global_dict[key].dtype)
    
    global_model.load_state_dict(global_dict)
    return global_model


# ============================================================
# EVALUATION FUNCTION
# ============================================================
def evaluate_model(model, eval_loader, config, device, round_num=None):
    """Evaluate model on evaluation set"""
    model.eval()
    eval_corrs = []
    bpm_errors = []
    bpm_accuracies = []
    all_predictions = []
    all_targets = []
    all_segments = []
    
    with torch.no_grad():
        for batch in eval_loader:
            ecg = batch['ecg_img'].to(device)
            eeg = batch['eeg_img'].to(device)
            target = batch['waveform'].to(device)
            segment_ids = batch['segment_id']
            
            dummy_resp = create_dummy_resp_input(ecg.shape[0], device)
            
            outputs = model(
                ecg_img=ecg,
                eeg_img=eeg,
                resp_img=dummy_resp
            )
            pred = outputs['waveform']
            
            for i in range(len(pred)):
                pred_np = pred[i].cpu().numpy()
                target_np = target[i].cpu().numpy()
                
                corr = np.corrcoef(pred_np, target_np)[0, 1]
                if not np.isnan(corr):
                    abs_corr = abs(corr)
                    eval_corrs.append(abs_corr)
                    all_predictions.append(pred_np)
                    all_targets.append(target_np)
                    all_segments.append(segment_ids[i])
                    
                    pred_bpm, _ = estimate_bpm(pred[i:i+1], fs=config.resp_fs)
                    true_bpm, _ = estimate_bpm(target[i:i+1], fs=config.resp_fs)
                    
                    if true_bpm > 0 and pred_bpm > 0:
                        abs_error, accuracy = calculate_bpm_accuracy(pred_bpm, true_bpm)
                        bpm_errors.append(abs_error)
                        bpm_accuracies.append(accuracy)
    
    results = {
        'correlations': eval_corrs,
        'bpm_errors': bpm_errors,
        'bpm_accuracies': bpm_accuracies,
        'predictions': all_predictions,
        'targets': all_targets,
        'segments': all_segments
    }
    
    return results


# ============================================================
# MAIN SCRIPT
# ============================================================
def main():
    config = Config()
    device = torch.device(config.device)
    print(f"Using device: {device}")
    print(f"Federated Learning with {config.num_clients} clients")
    
    # Load data
    data_dir = '.'
    gt_path = Path(data_dir) / 'rf_ground_truth.json'
    
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    
    # Get all segment IDs
    all_segment_ids = [str(s['sample_id']) for s in gt_data['samples']]
    print(f"Found {len(all_segment_ids)} segments in JSON")
    
    # Split data: training set and evaluation set
    train_ids, eval_ids = train_test_split(all_segment_ids, test_size=0.2, random_state=42)
    
    # Split training data among clients
    client_data = {}
    for client_id in range(config.num_clients):
        client_size = len(train_ids) // config.num_clients
        start_idx = client_id * client_size
        if client_id == config.num_clients - 1:
            client_data[client_id] = train_ids[start_idx:]
        else:
            client_data[client_id] = train_ids[start_idx:start_idx + client_size]
    
    #print(f"\nData distribution:")
    #for client_id, ids in client_data.items():
    #    print(f"  Client {client_id}: {len(ids)} segments")
    #    print(f"    IDs: {sorted([int(id) for id in ids])}")
    #print(f"  Evaluation: {len(eval_ids)} segments")
    
    # Create datasets
    train_datasets = {}
    for client_id, ids in client_data.items():
        train_datasets[client_id] = RFFusionDataset(data_dir, ids, augment=True)
    
    eval_dataset = RFFusionDataset(data_dir, eval_ids, augment=False)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # Initialize global model
    print("\n" + "="*60)
    print("INITIALIZING GLOBAL MODEL")
    print("="*60)
    
    model_config = HydraNetConfig(
        device=device,
        waveform_length=config.waveform_length,
        freeze_stems=config.freeze_stems,
        pretrained_stems=config.pretrained_stems,
        pretrained_branches=config.pretrained_branches,
        fusion_type=config.fusion_type
    )
    global_model = HydraNet(model_config).to(device)
    
    total_params = sum(p.numel() for p in global_model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create clients
    clients = []
    for client_id in range(config.num_clients):
        client = FederatedClient(client_id, train_datasets[client_id], config, device)
        clients.append(client)
    
    # Checkpoint path
    checkpoint_path = os.path.join(config.checkpoint_dir, 'federated_rf_model.pth')
    
    # Federated training loop
    print("\n" + "="*60)
    print("STARTING FEDERATED TRAINING")
    print("="*60)
    
    best_eval_corr = 0
    best_bpm_acc = 0
    
    for round_num in range(1, config.num_rounds + 1):
        print(f"\n--- Federated Round {round_num}/{config.num_rounds} ---")
        
        client_updates = []
        client_sizes = []
        
        # Each client does local training
        for client in clients:
            print(f"\nTraining Client {client.client_id}:")
            local_update, client_size = client.local_train(global_model, config.num_local_epochs)
            client_updates.append(local_update)
            client_sizes.append(client_size)
        
        # Federated averaging
        global_model = federated_averaging(global_model, client_updates, client_sizes)
        
        # Evaluate every 5 rounds
        if round_num % 5 == 0 or round_num == config.num_rounds:
            results = evaluate_model(global_model, eval_loader, config, device, round_num)
            
            mean_corr = np.mean(results['correlations'])
            mean_bpm_acc = np.mean(results['bpm_accuracies']) if results['bpm_accuracies'] else 0
            
            print(f"\n--- Round {round_num} Evaluation ---")
            print(f"  Mean Correlation: {mean_corr:.4f}")
            print(f"  Mean BPM Accuracy: {mean_bpm_acc:.2f}%")
            
            # Save best model
            combined_score = mean_corr * 0.5 + (mean_bpm_acc / 100) * 0.5
            best_combined = best_eval_corr * 0.5 + (best_bpm_acc / 100) * 0.5
            
            if combined_score > best_combined:
                best_eval_corr = mean_corr
                best_bpm_acc = mean_bpm_acc
                torch.save({
                    'round': round_num,
                    'model_state_dict': global_model.state_dict(),
                    'best_eval_corr': best_eval_corr,
                    'best_bpm_acc': best_bpm_acc,
                    'config': config
                }, checkpoint_path)
                print(f"New best model saved! (Corr: {best_eval_corr:.4f}, BPM Acc: {best_bpm_acc:.2f}%)")
    
    print("\n" + "="*60)
    print("FEDERATED TRAINING COMPLETED")
    print("="*60)
    
    # ============================================================
    # FINAL EVALUATION
    # ============================================================
    print("\n" + "="*60)
    print("FINAL EVALUATION OF BEST MODEL")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    global_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Detailed evaluation
    results = evaluate_model(global_model, eval_loader, config, device)
    
    all_correlations = np.array(results['correlations'])
    mean_corr = np.mean(all_correlations)
    std_corr = np.std(all_correlations)
    best_idx = np.argmax(all_correlations)
    
    mean_bpm_error = np.mean(results['bpm_errors']) if results['bpm_errors'] else 0
    mean_bpm_acc = np.mean(results['bpm_accuracies']) if results['bpm_accuracies'] else 0
    
    print(f"\nPer-Segment Results:")
    for i, (seg, corr) in enumerate(zip(results['segments'], results['correlations'])):
        print(f"  Segment {seg}: Correlation = {corr:.3f}")
    
    print(f"\nFINAL RESULTS:")
    print(f"Waveform Correlation:")
    print(f"  Mean: {mean_corr:.4f} ± {std_corr:.4f}")
    print(f"  Best: {all_correlations[best_idx]:.4f} (Segment: {results['segments'][best_idx]})")
    print(f"\nBPM Accuracy:")
    print(f"  Mean Error: {mean_bpm_error:.2f} BPM")
    print(f"  Mean Accuracy: {mean_bpm_acc:.2f}%")
    
    # ============================================================
    # PLOT BEST PREDICTION
    # ============================================================
    print("\n" + "="*60)
    print("PLOTTING BEST PREDICTION")
    print("="*60)
    
    time_axis = np.linspace(0, 60, config.waveform_length)
    
    plt.figure(figsize=(14, 8))
    
    # Main plot
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, results['targets'][best_idx], 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    plt.plot(time_axis, results['predictions'][best_idx], 'r--', linewidth=2, label='Prediction', alpha=0.8)
    
    pred_bpm, _ = estimate_bpm(torch.tensor(results['predictions'][best_idx]).unsqueeze(0), fs=config.resp_fs)
    true_bpm, _ = estimate_bpm(torch.tensor(results['targets'][best_idx]).unsqueeze(0), fs=config.resp_fs)
    
    plt.title(f'Federated Learning - Best Prediction (Corr: {all_correlations[best_idx]:.3f}, BPM: {pred_bpm:.1f}/{true_bpm:.1f})', fontsize=14)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Normalized Amplitude', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Error plot
    plt.subplot(2, 1, 2)
    error = results['predictions'][best_idx] - results['targets'][best_idx]
    plt.plot(time_axis, error, 'g-', linewidth=1.5, alpha=0.7)
    plt.fill_between(time_axis, error, 0, where=(error>0), color='red', alpha=0.3, label='Overestimation')
    plt.fill_between(time_axis, error, 0, where=(error<0), color='blue', alpha=0.3, label='Underestimation')
    plt.title(f'Prediction Error (RMSE: {np.sqrt(np.mean(error**2)):.3f})', fontsize=14)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(config.checkpoint_dir, 'federated_best_prediction.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved to: {plot_path}")
    print(f"Best model saved to: {checkpoint_path}")
    print("\n" + "="*60)
    print("FEDERATED LEARNING COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()