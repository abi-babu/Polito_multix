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
from scipy.signal import find_peaks, butter, filtfilt
from pathlib import Path

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
        distance=fs//3,  # Minimum 2 seconds between breaths
        prominence=0.1,   # Minimum peak prominence
        width=5          # Minimum peak width
    )
    
    if len(peaks) > 1:
        # Calculate BPM from peak intervals
        peak_intervals = np.diff(peaks) / fs  # intervals in seconds
        mean_interval = np.mean(peak_intervals)
        bpm = 60 / mean_interval
    else:
        bpm = 0.0
    
    return bpm, len(peaks)


def calculate_bpm_accuracy(pred_bpm, true_bpm):
    """Calculate BPM accuracy metrics"""
    if true_bpm == 0 or pred_bpm == 0:
        return 0.0, 100.0  # No accuracy if can't detect
    
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
        self.num_epochs = 20  # Increased epochs
        self.checkpoint_dir = 'checkpoints_rf'
        self.resp_fs = 150  # Sampling frequency of respiration signal
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)


# ============================================================
# DATASET WITH AUGMENTATION
# ============================================================
class RFFusionDataset(Dataset):
    def __init__(self, data_dir, segment_ids, waveform_length=8998, augment=False):
        self.data_dir = Path(data_dir)
        self.segment_ids = segment_ids
        self.waveform_length = waveform_length
        self.augment = augment
        
        # Load the JSON file
        with open(self.data_dir / 'rf_ground_truth.json', 'r') as f:
            self.all_data = json.load(f)
        
        # Create a mapping from sample_id to its data
        self.samples_dict = {}
        for sample in self.all_data['samples']:
            sample_id = str(sample['sample_id'])
            self.samples_dict[sample_id] = sample
    
    def __len__(self):
        return len(self.segment_ids)
    
    def _load_image(self, path, augment=False):
        img = Image.open(path).convert('RGB')
        
        if augment:
            # Simple augmentations
            if np.random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.random() > 0.5:
                # Small random rotation
                angle = np.random.uniform(-5, 5)
                img = img.rotate(angle, expand=False, fillcolor=0)
        
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img
    
    def _preprocess_waveform(self, waveform):
        waveform = np.array(waveform, dtype=np.float32)
        
        # Normalize
        waveform = (waveform - np.mean(waveform)) / (np.std(waveform) + 1e-6)
        
        if len(waveform) > self.waveform_length:
            # Random crop for augmentation during training
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
        
        # Load images with augmentation
        heatmap_path = self.data_dir / 'rf_heatmaps' / f"heatmap_{segment_id}.png"
        spec_path = self.data_dir / 'rf_spect' / f"spectrogram_{segment_id}.png"
        
        heatmap = self._load_image(heatmap_path, augment=self.augment)
        spectrogram = self._load_image(spec_path, augment=self.augment)
        
        # Load waveform
        waveform = self._preprocess_waveform(segment_data['ground_truth'])
        
        return {
            'ecg_img': spectrogram,
            'eeg_img': heatmap,
            'waveform': waveform,
            'segment_id': segment_id
        }

#current model needs a bit of refinement to adapt to not using resp input
def create_dummy_resp_input(batch_size, device):
    """Create a dummy input for respiration modality"""
    dummy = torch.zeros(batch_size, 3, 224, 224).to(device)
    return dummy


# ============================================================
# MAIN SCRIPT
# ============================================================
def main():
    config = Config()
    device = torch.device(config.device)
    print(f"Using device: {device}")
    print(f"Fusion type: {config.fusion_type}")
    
    # Load data
    data_dir = '.'
    gt_path = Path(data_dir) / 'rf_ground_truth.json'
    
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    
    # Get all segment IDs
    all_segment_ids = [str(s['sample_id']) for s in gt_data['samples']]
    print(f"Found {len(all_segment_ids)} segments in JSON")
    
    # 80/20 train/eval split
    train_ids, eval_ids = train_test_split(all_segment_ids, test_size=0.2, random_state=42)
    
    print(f"\nDataset split (80/20):")

    
    # Create datasets with augmentation for training
    train_dataset = RFFusionDataset(data_dir, train_ids, augment=True)
    eval_dataset = RFFusionDataset(data_dir, eval_ids, augment=False)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Initialize model
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    
    model_config = HydraNetConfig(
        device=device,
        waveform_length=config.waveform_length,
        freeze_stems=config.freeze_stems,
        pretrained_stems=config.pretrained_stems,
        pretrained_branches=config.pretrained_branches,
        fusion_type=config.fusion_type
    )
    model = HydraNet(model_config).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Checkpoint path
    checkpoint_path = os.path.join(config.checkpoint_dir, 'rf_hydranet.pth')
    
    # Load checkpoint if exists
    start_epoch = 1
    best_eval_corr = 0
    best_bpm_acc = 0
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_eval_corr = checkpoint.get('best_eval_corr', 0)
        best_bpm_acc = checkpoint.get('best_bpm_acc', 0)
        print(f"\nLoaded checkpoint from {checkpoint_path}")
        print(f"  Best eval correlation: {best_eval_corr:.4f}")
        print(f"  Best BPM accuracy: {best_bpm_acc:.2f}%")
    
    # Improved optimizer with different learning rates
    optimizer = optim.AdamW([
        {'params': model.stems.parameters(), 'lr': config.learning_rate * 0.1},
        {'params': model.branches.parameters(), 'lr': config.learning_rate},
        {'params': model.main_fusion.parameters(), 'lr': config.learning_rate},
        {'params': model.prediction_head.parameters(), 'lr': config.learning_rate * 2}
    ], weight_decay=1e-4)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Training history
    train_losses = []
    eval_corrs = []
    eval_bpm_accs = []
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    for epoch in range(start_epoch, config.num_epochs + 1):
        # Training
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            ecg = batch['ecg_img'].to(device)
            eeg = batch['eeg_img'].to(device)
            target = batch['waveform'].to(device)
            
            # Create dummy input for respiration
            batch_size = ecg.shape[0]
            dummy_resp = create_dummy_resp_input(batch_size, device)
            
            optimizer.zero_grad()
            outputs = model(
                ecg_img=ecg, 
                eeg_img=eeg, 
                resp_img=dummy_resp,
                gt_waveform=target
            )
            outputs['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += outputs['loss'].item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Update scheduler
        scheduler.step()
        
        # Evaluation every 5 epochs
        if epoch % 5 == 0 or epoch == config.num_epochs:
            model.eval()
            eval_corrs_epoch = []
            bpm_errors = []
            bpm_accuracies = []
            
            with torch.no_grad():
                for batch in eval_loader:
                    ecg = batch['ecg_img'].to(device)
                    eeg = batch['eeg_img'].to(device)
                    target = batch['waveform'].to(device)
                    
                    # Create dummy input for respiration
                    batch_size = ecg.shape[0]
                    dummy_resp = create_dummy_resp_input(batch_size, device)
                    
                    outputs = model(
                        ecg_img=ecg, 
                        eeg_img=eeg, 
                        resp_img=dummy_resp
                    )
                    pred = outputs['waveform']
                    
                    for i in range(len(pred)):
                        pred_np = pred[i].cpu().numpy()
                        target_np = target[i].cpu().numpy()
                        
                        # Calculate correlation
                        corr = np.corrcoef(pred_np, target_np)[0, 1]
                        if not np.isnan(corr):
                            eval_corrs_epoch.append(abs(corr))
                        
                        # Calculate BPM from both signals
                        pred_bpm, pred_peaks = estimate_bpm(pred[i:i+1], fs=config.resp_fs)
                        
                        # For true BPM, we need to estimate from ground truth
                        # You can also store pre-calculated BPM in your JSON
                        true_bpm, true_peaks = estimate_bpm(target[i:i+1], fs=config.resp_fs)
                        
                        if true_bpm > 0 and pred_bpm > 0:
                            abs_error, accuracy = calculate_bpm_accuracy(pred_bpm, true_bpm)
                            bpm_errors.append(abs_error)
                            bpm_accuracies.append(accuracy)
            
            avg_eval_corr = np.mean(eval_corrs_epoch) if eval_corrs_epoch else 0
            avg_bpm_error = np.mean(bpm_errors) if bpm_errors else 0
            avg_bpm_acc = np.mean(bpm_accuracies) if bpm_accuracies else 0
            
            eval_corrs.append(avg_eval_corr)
            eval_bpm_accs.append(avg_bpm_acc)
            
            print(f"\nEpoch {epoch:2d}:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Eval Corr:  {avg_eval_corr:.4f}")
            print(f"  BPM Error:  {avg_bpm_error:.2f} BPM")
            print(f"  BPM Acc:    {avg_bpm_acc:.2f}%")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model based on combined metric
            combined_score = avg_eval_corr * 0.5 + (avg_bpm_acc / 100) * 0.5
            
            if combined_score > (best_eval_corr * 0.5 + (best_bpm_acc / 100) * 0.5):
                best_eval_corr = avg_eval_corr
                best_bpm_acc = avg_bpm_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_eval_corr': best_eval_corr,
                    'best_bpm_acc': best_bpm_acc,
                    'train_losses': train_losses,
                    'eval_corrs': eval_corrs,
                    'config': config
                }, checkpoint_path)
                print(f"New best model saved! (Corr: {best_eval_corr:.4f}, BPM Acc: {best_bpm_acc:.2f}%)")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    
    # ============================================================
    # FINAL EVALUATION WITH BPM METRICS
    # ============================================================
    print("\n" + "="*60)
    print("FINAL EVALUATION WITH BPM METRICS")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Detailed evaluation
    all_predictions = []
    all_targets = []
    all_segments = []
    all_correlations = []
    all_bpm_errors = []
    all_bpm_accs = []
    
    with torch.no_grad():
        for batch in eval_loader:
            ecg = batch['ecg_img'].to(device)
            eeg = batch['eeg_img'].to(device)
            target = batch['waveform'].to(device)
            segment_ids = batch['segment_id']
            
            # Create dummy input for respiration
            batch_size = ecg.shape[0]
            dummy_resp = create_dummy_resp_input(batch_size, device)
            
            outputs = model(
                ecg_img=ecg, 
                eeg_img=eeg, 
                resp_img=dummy_resp
            )
            pred = outputs['waveform']
            
            for i in range(len(pred)):
                pred_np = pred[i].cpu().numpy()
                target_np = target[i].cpu().numpy()
                
                # Correlation
                corr = np.corrcoef(pred_np, target_np)[0, 1]
                if not np.isnan(corr):
                    abs_corr = abs(corr)
                    all_correlations.append(abs_corr)
                    all_predictions.append(pred_np)
                    all_targets.append(target_np)
                    all_segments.append(segment_ids[i])
                    
                    # BPM calculation
                    pred_bpm, pred_peaks = estimate_bpm(pred[i:i+1], fs=config.resp_fs)
                    true_bpm, true_peaks = estimate_bpm(target[i:i+1], fs=config.resp_fs)
                    
                    if true_bpm > 0 and pred_bpm > 0:
                        abs_error, accuracy = calculate_bpm_accuracy(pred_bpm, true_bpm)
                        all_bpm_errors.append(abs_error)
                        all_bpm_accs.append(accuracy)
                        
                        print(f"Segment {segment_ids[i]}:")
                        print(f"  Correlation: {abs_corr:.3f}")
                        print(f"  True BPM: {true_bpm:.1f}, Pred BPM: {pred_bpm:.1f}")
                        print(f"  BPM Accuracy: {accuracy:.1f}%")
    
    # Overall statistics
    all_correlations = np.array(all_correlations)
    mean_corr = np.mean(all_correlations)
    std_corr = np.std(all_correlations)
    best_idx = np.argmax(all_correlations)
    
    mean_bpm_error = np.mean(all_bpm_errors) if all_bpm_errors else 0
    mean_bpm_acc = np.mean(all_bpm_accs) if all_bpm_accs else 0
    
    print(f"\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Waveform Correlation:")
    print(f"  Mean: {mean_corr:.4f} ± {std_corr:.4f}")
    print(f"  Best: {all_correlations[best_idx]:.4f} (Segment: {all_segments[best_idx]})")
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
    plt.plot(time_axis, all_targets[best_idx], 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    plt.plot(time_axis, all_predictions[best_idx], 'r--', linewidth=2, label='Prediction', alpha=0.8)
    
    # Calculate BPM for this segment
    pred_bpm, _ = estimate_bpm(torch.tensor(all_predictions[best_idx]).unsqueeze(0), fs=config.resp_fs)
    true_bpm, _ = estimate_bpm(torch.tensor(all_targets[best_idx]).unsqueeze(0), fs=config.resp_fs)
    
    plt.title(f'Best Prediction - Segment {all_segments[best_idx]} (Corr: {all_correlations[best_idx]:.3f}, BPM: {pred_bpm:.1f}/{true_bpm:.1f})', fontsize=14)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Normalized Amplitude', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Error plot
    plt.subplot(2, 1, 2)
    error = all_predictions[best_idx] - all_targets[best_idx]
    plt.plot(time_axis, error, 'g-', linewidth=1.5, alpha=0.7)
    plt.fill_between(time_axis, error, 0, where=(error>0), color='red', alpha=0.3, label='Overestimation')
    plt.fill_between(time_axis, error, 0, where=(error<0), color='blue', alpha=0.3, label='Underestimation')
    plt.title(f'Prediction Error (RMSE: {np.sqrt(np.mean(error**2)):.3f})', fontsize=14)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(config.checkpoint_dir, 'rf_best_prediction_detailed.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nDetailed plot saved to: {plot_path}")
    print(f"Best model saved to: {checkpoint_path}")
    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == "__main__":
    main()
