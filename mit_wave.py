import torch
import numpy as np
from PIL import Image
import json
from config import Config
from model.hydranet import HydraFusion
from scipy.signal import find_peaks
import os
import copy
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


# CONFIG
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
    '--enable_camera', 'true',
    '--enable_cam_fusion', 'false',
    '--waveform_length' , '7500' ,
]

cfg = Config(args)
device = cfg.device

FS = 250
SEG_DURATION = 30
SEG_LEN = FS * SEG_DURATION  # 7500 samples per segment


# HELPERS
def load_image_tensor(path):
    img = Image.open(path).convert('RGB')
    return (
        torch.tensor(np.array(img))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float() / 255.0
    )


def load_waveform_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    if "respiration_waveform" not in data:
        raise KeyError(f"'respiration_waveform' not found. Available keys: {list(data.keys())}")

    waveform = np.array(data["respiration_waveform"], dtype=np.float32)
    waveform = (waveform - waveform.min()) / (waveform.max() - waveform.min() + 1e-6)

    gt_waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(device)
    gt_bpm = None
    return gt_waveform, gt_bpm


def normalize(w):
    return (w - w.mean()) / (w.std() + 1e-6)


def estimate_bpm(waveform, fs=250):
    waveform_np = waveform.squeeze().cpu().numpy()
    peaks, _ = find_peaks(waveform_np, distance=fs * 3.5, prominence=0.6, width=fs * 0.5)
    duration_sec = len(waveform_np) / fs
    bpm = len(peaks) / duration_sec * 60
    return torch.tensor([[bpm]], dtype=torch.float32).to(device), peaks


def corr_loss(pred, gt):
    pred_c = pred - pred.mean()
    gt_c = gt - gt.mean()
    return 1 - (pred_c * gt_c).sum() / (pred_c.norm() * gt_c.norm() + 1e-6)

def contrast_loss(waveform):
    diff = waveform[:, 1:] - waveform[:, :-1]
    return -torch.mean(torch.abs(diff))

def peak_count_loss(pred, expected_count, fs=250):
    pred_np = pred.squeeze().detach().cpu().numpy()
    peaks, _ = find_peaks(pred_np, distance=fs * 2.5, prominence=0.3, width=fs * 0.3)
    return torch.abs(torch.tensor(len(peaks) - expected_count, dtype=torch.float32).to(pred.device))

def slope_loss(pred, gt):
    pred_diff = pred[:, 1:] - pred[:, :-1]
    gt_diff = gt[:, 1:] - gt[:, :-1]
    return torch.nn.functional.mse_loss(pred_diff, gt_diff)

def derivative_loss(pred, gt):
    pred_dx = pred[:, 1:] - pred[:, :-1]
    gt_dx = gt[:, 1:] - gt[:, :-1]
    return torch.mean(torch.abs(pred_dx - gt_dx))

def contrast_balance_loss(waveform):
    return -torch.std(waveform, dim=1).mean()


# LOAD CAMERA FRAMES

camera_frames = []
for i in range(1, 181):
    fname = f"resp_segments/resp_rgb_{i:03d}.png"
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Missing camera frame: {fname}")
    camera_frames.append(load_image_tensor(fname).to(device))


camera_x = camera_frames[np.random.randint(0, len(camera_frames))]


# MODEL INIT

checkpoint_path = "checkpoints/hydrafusion_rf_trained_mit_normal.pth"
global_model = HydraFusion(cfg).to(device)

if os.path.exists(checkpoint_path):
    global_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Checkpoint loaded successfully.")
else:
    print("No checkpoint found. Starting fresh.")

# NORMALTRAINING LOOP
global_model.train()

num_epochs = 60
lr = 1e-3
optimizer = torch.optim.Adam(global_model.parameters(), lr=lr)

all_samples = [
    ("ecg_heatmap.png", "ground_truth_sleep.json"),
    ("eeg_spectrogram.png", "ground_truth_sleep.json"),
    ("camera_only", "ground_truth_sleep.json"),
]

for epoch in range(1, num_epochs + 1):
    total_loss = 0.0
    total_steps = 0

    for rf_path, gt_path in all_samples:

        # Load GT waveform
        gt_waveform, _ = load_waveform_from_json(gt_path)
        gt_segment = gt_waveform
        # RF selection (same as before)
        if rf_path == "ecg_heatmap.png":
            rf_heatmap = load_image_tensor("ecg_heatmap.png").to(device)
            rf_spectrogram = load_image_tensor("eeg_spectrogram.png").to(device)
        elif rf_path == "eeg_spectrogram.png":
            rf_heatmap = load_image_tensor("ecg_heatmap.png").to(device)
            rf_spectrogram = load_image_tensor("eeg_spectrogram.png").to(device)
        else:
            rf_heatmap = load_image_tensor("ecg_heatmap.png").to(device)
            rf_spectrogram = load_image_tensor("eeg_spectrogram.png").to(device)

        optimizer.zero_grad()

        loss, _, output = global_model(
            rf_heatmap_x=rf_heatmap,
            rf_spectrogram_x=rf_spectrogram,
            rightcamera_x=camera_x,
            leftcamera_x=camera_x,
            rf_y=gt_segment,
        )
        pred = output['fused_waveform']
        gt = gt_segment[:, :pred.shape[1]]

        # Losses 
        mse = torch.nn.functional.mse_loss(pred, gt)
        _, gt_peaks = estimate_bpm(gt, fs=FS)
        expected_peaks = len(gt_peaks)

        reg_loss = ( 
            0.2 * corr_loss(pred, gt) + 
            0.05 * contrast_loss(pred) + 
            0.1 * slope_loss(pred, gt) + 
            0.02 * peak_count_loss(pred, expected_peaks) + 
            0.05 * derivative_loss(pred, gt) + 
            0.02 * contrast_balance_loss(pred) 
        ) 
        loss = 0.5 * mse + reg_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_steps += 1

    avg_loss = total_loss / max(1, total_steps)
    print(f"[Epoch {epoch}] Training Loss: {avg_loss:.4f}")



# POST-TRAIN EVALUATION (30s SEGMENT)
global_model.eval()

rf_heatmap = load_image_tensor("ecg_heatmap.png").to(device)
rf_spectrogram = load_image_tensor("eeg_spectrogram.png").to(device)
gt_waveform, _ = load_waveform_from_json("ground_truth_sleep.json")

with torch.no_grad():
    first_seg = gt_waveform
    _, output_detections, output = global_model(
        rf_heatmap_x=rf_heatmap,
        rf_spectrogram_x=rf_spectrogram,
        rightcamera_x=camera_x,
        leftcamera_x=camera_x,
        rf_y=first_seg,
    )


# PROCESS PREDICTION + GT
pred_raw = output['fused_waveform'].detach().cpu().squeeze().numpy()
gt_full  = first_seg.squeeze().cpu().numpy()

min_len = min(len(pred_raw), len(gt_full))
pred_raw = pred_raw[:min_len]
gt_full  = gt_full[:min_len]


pred = (pred_raw - np.min(pred_raw)) / (np.max(pred_raw) - np.min(pred_raw) + 1e-6)
gt   = gt_full

# ADVANCED SMOOTHING

from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

pred_sg = savgol_filter(pred, window_length=151, polyorder=3)
pred_smoothed = gaussian_filter1d(pred_sg, sigma=3.5)
gt_smoothed = gaussian_filter1d(gt, sigma=5.0)


# BPM ESTIMATION

gt_bpm_tensor, _   = estimate_bpm(torch.tensor(gt_smoothed).unsqueeze(0).to(device), fs=FS)
pred_bpm_tensor, detected_peaks = estimate_bpm(torch.tensor(pred_smoothed).unsqueeze(0).to(device), fs=FS)

true_bpm = gt_bpm_tensor.item()
pred_bpm = pred_bpm_tensor.item()

if true_bpm <= 0:
    bpm_accuracy = 0.0
else:
    bpm_error = abs(pred_bpm - true_bpm)
    bpm_accuracy = 100 - ((bpm_error / true_bpm) * 100)


# WAVEFORM ACCURACY

rmse = np.sqrt(mean_squared_error(gt_smoothed, pred_smoothed))
nrmse = rmse / (np.max(gt_smoothed) - np.min(gt_smoothed) + 1e-6)
waveform_accuracy = 100 - (nrmse * 100)

print(f"BPM Accuracy: {bpm_accuracy:.2f}%")
print(f"Waveform Accuracy: {waveform_accuracy:.2f}%")
print(f"True BPM: {true_bpm:.2f}, Predicted BPM: {pred_bpm:.2f}")
print(f"Detected Peaks: {len(detected_peaks)}")


# VISUALIZATION

try:
    duration_sec = len(pred_smoothed) / FS
    time_axis = np.linspace(0, duration_sec, num=len(pred_smoothed))

    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, pred_smoothed, color='red', label='Prediction')
    plt.plot(time_axis, gt_smoothed[:len(pred_smoothed)], color='blue', label='Ground Truth')
    plt.title("Prediction vs Ground Truth Respiration Signal (30s Segment)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig("waveform_projection_sleep_mit_normal.jpg")
    plt.close()
except Exception as e:
    print(f"Visualization failed: {e}")

os.makedirs("checkpoints", exist_ok=True)
torch.save(global_model.state_dict(), "checkpoints/hydrafusion_rf_trained_mit_normal.pth")
print("Model saved.")
