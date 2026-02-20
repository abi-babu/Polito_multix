import torch
import numpy as np
from PIL import Image
import json
from config import Config
from model.hydranet import HydraFusion
from scipy.signal import find_peaks
import os
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


# CONFIGURATION

args = [
    '--activation', 'relu',
    '--dropout', '0.0',
    '--batch_size', '1',
    '--device', 'cpu',
    '--pretrained', 'false',
    '--use_custom_transforms', 'false',
    '--fusion_sweep', 'false',
    '--resume', 'true',
    '--enable_rf_heatmap', 'true',
    '--enable_rf_spectrogram', 'true',
    '--enable_rf_fusion', 'true',
    '--waveform_length' , '8998'
]
cfg = Config(args)
device = cfg.device

# UTILITIES

def load_image_tensor(path):
    img = Image.open(path).convert('RGB')
    return torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

def load_waveform_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    waveform = np.array(data['ground_truth'], dtype=np.float32)
    waveform = (waveform - np.mean(waveform)) / (np.std(waveform) + 1e-6)
    return (
        torch.tensor(waveform).unsqueeze(0).to(device),
        torch.tensor([[data['gt_bpm']]], dtype=torch.float32).to(device)
    )

def normalize(w):
    return (w - np.mean(w)) / (np.std(w) + 1e-6)

def estimate_bpm(waveform, fs=150):
    waveform_np = waveform.squeeze().cpu().numpy()
    peaks, _ = find_peaks(waveform_np, distance=fs // 2, prominence=0.2)
    duration_sec = len(waveform_np) / fs
    bpm = len(peaks) / duration_sec * 60
    return torch.tensor([[bpm]], dtype=torch.float32).to(device), peaks


# DATASET
rf_heatmap = load_image_tensor("rf_heatmap.jpg").to(device)
rf_spectrogram = load_image_tensor("rf_spectrogram.jpg").to(device)
gt_waveform, gt_bpm = load_waveform_from_json("ground_truth.json")

# MODEL 
checkpoint_path = "checkpoints/hydrafusion_rf_trained_split_rf_normal.pth"
model = HydraFusion(cfg).to(device)

if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Checkpoint loaded successfully.")
else:
    print("No checkpoint found. Starting fresh.")

# one optimizer, no local/global split
lr = 1e-4  # you can try 1e-3 if loss plateaus
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# NORMAL TRAINING LOOP
num_epochs = 40
target_loss = 0.0

for epoch in range(1, num_epochs + 1):
    model.train()
    optimizer.zero_grad()

    # same forward logic as your global_model call
    _, _, output = model(
        rf_heatmap_x=rf_heatmap,
        rf_spectrogram_x=rf_spectrogram,
        rf_y=gt_waveform
    )

    loss = torch.nn.functional.mse_loss(output['fused_waveform'], gt_waveform)
    loss.backward()
    optimizer.step()

    print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")

    if loss.item() < target_loss:
        print("Target loss reached. Stopping training.")
        break

# EVALUATION 
model.eval()
rf_heatmap = load_image_tensor("rf_heatmap.jpg").to(device)
rf_spectrogram = load_image_tensor("rf_spectrogram.jpg").to(device)
gt_waveform, gt_bpm = load_waveform_from_json("ground_truth.json")

with torch.no_grad():
    _, output_detections, output = model(
        rf_heatmap_x=rf_heatmap,
        rf_spectrogram_x=rf_spectrogram,
        rf_y=gt_waveform
    )

print("\n--- Fused Waveform Output ---")
print(output['fused_waveform'].detach().cpu().numpy())
print("\n--- RF Fused Feature Map ---")
print(output_detections['rf_fusion_features'])

# ACCURACY METRICS 
gt = normalize(gt_waveform.squeeze().cpu().numpy())
pred_raw = output['fused_waveform'].detach().cpu().squeeze().numpy()

pred_rescaled = (
    (pred_raw - np.min(pred_raw)) /
    (np.max(pred_raw) - np.min(pred_raw) + 1e-6) *
    (np.max(gt) - np.min(gt)) + np.min(gt)
)
pred = normalize(pred_rescaled)

true_bpm = gt_bpm.item()
smoothed_pred = gaussian_filter1d(pred_rescaled, sigma=2)
pred_bpm_tensor, detected_peaks = estimate_bpm(torch.tensor(smoothed_pred))
pred_bpm = pred_bpm_tensor.item()

bpm_error = abs(pred_bpm - true_bpm)
bpm_accuracy = 100 - ((bpm_error / true_bpm) * 100)

rmse = np.sqrt(mean_squared_error(gt, pred))
nrmse = rmse / (np.max(gt) - np.min(gt))
waveform_accuracy = 100 - (nrmse * 100)

print(f"BPM Accuracy: {bpm_accuracy:.2f}%")
print(f"Waveform Accuracy: {waveform_accuracy:.2f}%")

# VISUALIZATION
try:
    time_axis = np.linspace(0, 60, num=len(smoothed_pred))
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, smoothed_pred, color='red', label='Prediction')
    plt.plot(time_axis, gt, color='blue', label='Ground Truth')
    plt.title("Prediction vs Ground Truth Respiration Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig("waveform_projection_normal.jpg")
    plt.close()
    print("Waveform projection saved as waveform_projection.jpg")
except Exception as e:
    print(f"Visualization failed: {e}")


# SAVE MODEL

os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), checkpoint_path)
print("Model saved.")
