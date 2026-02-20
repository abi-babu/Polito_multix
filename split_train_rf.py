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
import pickle 

# Configuration
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

# Utilities
def load_image_tensor(path):
    img = Image.open(path).convert('RGB')
    return torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

def load_waveform_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    waveform = np.array(data['ground_truth'], dtype=np.float32)
    waveform = (waveform - np.mean(waveform)) / (np.std(waveform) + 1e-6)
    return torch.tensor(waveform).unsqueeze(0).to(device), torch.tensor([[data['gt_bpm']]], dtype=torch.float32).to(device)

def normalize(w): return (w - np.mean(w)) / (np.std(w) + 1e-6)

def estimate_bpm(waveform, fs=150):
    waveform_np = waveform.squeeze().cpu().numpy()
    peaks, _ = find_peaks(waveform_np, distance=fs//2, prominence=0.2)
    duration_sec = len(waveform_np) / fs
    bpm = len(peaks) / duration_sec * 60
    return torch.tensor([[bpm]], dtype=torch.float32).to(device), peaks

# Client data: modality-specific
client_data = {
    "client_1": [("rf_heatmap.jpg", "ground_truth.json")],
    "client_2": [("rf_spectrogram.jpg", "ground_truth.json")]
}

# Global model
checkpoint_path = "checkpoints/hydrafusion_rf_trained_split_rf.pth"
global_model = HydraFusion(cfg).to(device)

if os.path.exists(checkpoint_path):
    global_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Checkpoint loaded successfully.")
else:
    print("No checkpoint found. Starting fresh.")

global_client_model = copy.deepcopy(global_model)

# Split Federated Training
num_rounds =20
target_loss = 0.0
lr = 1e-4

for round_num in range(1, num_rounds + 1):
    client_models = {}
    client_losses = []
    global_model.train()

    for client_id, samples in client_data.items():
        local_model = copy.deepcopy(global_client_model)
        optimizer_local = torch.optim.Adam(local_model.parameters(), lr=lr)
        optimizer_global = torch.optim.Adam(global_model.parameters(), lr=lr)

        local_model.train()
        local_loss = 0.0

        for rf_path, gt_path in samples:
            gt_waveform, gt_bpm = load_waveform_from_json(gt_path)
            rf_heatmap = load_image_tensor("rf_heatmap.jpg").to(device)
            rf_spectrogram = load_image_tensor("rf_spectrogram.jpg").to(device)


            # --- Client-side feature extraction ---
            local_model.eval()
            with torch.no_grad():
                _, output_detections, _ = local_model(
                    rf_heatmap_x=rf_heatmap,
                    rf_spectrogram_x=rf_spectrogram,
                    rf_y=gt_waveform
                )
            local_model.train()

            rf_features = output_detections['rf_fusion_features']

            # --- Server-side fusion and prediction ---
            optimizer_global.zero_grad()
            optimizer_local.zero_grad()

            _, _, output = global_model(
                rf_heatmap_x=rf_heatmap,
                rf_spectrogram_x=rf_spectrogram,
                rf_y=gt_waveform
            )

            loss = torch.nn.functional.mse_loss(output['fused_waveform'], gt_waveform)
            loss.backward()

            optimizer_local.step()
            optimizer_global.step()

            local_loss += loss.item()
        serialized_state = pickle.dumps(local_model.state_dict())
        bytes_sent = len(serialized_state)  

        #print(f"[Round {round_num}] {client_id} sent {bytes_sent/1024:.2f} KB to server")
        client_models[client_id] = copy.deepcopy(local_model.state_dict())
        client_losses.append(local_loss / len(samples))

    # --- Federated Averaging of client models ---
    avg_state_dict = copy.deepcopy(client_models["client_1"])
    for key in avg_state_dict:
        avg_state_dict[key] = sum([client_models[c][key] for c in client_models]) / len(client_models)
    global_client_model.load_state_dict(avg_state_dict)

    round_loss = sum(client_losses) / len(client_losses)
    print(f"[Round {round_num}] Global Loss: {round_loss:.4f}")
    if round_loss < target_loss:
        print("Target loss reached. Stopping training.")
        break

# Evaluation
global_model.eval()
rf_heatmap = load_image_tensor("rf_heatmap.jpg").to(device)
rf_spectrogram = load_image_tensor("rf_spectrogram.jpg").to(device)
gt_waveform, gt_bpm = load_waveform_from_json("ground_truth.json")

with torch.no_grad():
    _, output_detections, output = global_model(
        rf_heatmap_x=rf_heatmap,
        rf_spectrogram_x=rf_spectrogram,
        rf_y=gt_waveform
    )

print("\n--- Fused Waveform Output ---")
print(output['fused_waveform'].detach().cpu().numpy())
print("\n--- RF Fused Feature Map ---")
#print(output_detections['rf_fusion'].detach().cpu().numpy())
print(output_detections['rf_fusion_features'])
# Accuracy
gt = normalize(gt_waveform.squeeze().cpu().numpy())
pred_raw = output['fused_waveform'].detach().cpu().squeeze().numpy()
pred_rescaled = (pred_raw - np.min(pred_raw)) / (np.max(pred_raw) - np.min(pred_raw)) * (np.max(gt) - np.min(gt)) + np.min(gt)
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

# Visualization
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
    plt.savefig("waveform_projection_split.jpg")
    plt.close()
    print("Waveform projection saved as waveform_projection.jpg")
except Exception as e:
    print(f"Visualization failed: {e}")

# Save model
os.makedirs("checkpoints", exist_ok=True)
torch.save(global_model.state_dict(), "checkpoints/hydrafusion_rf_trained_split_rf.pth")
print("Model saved.")
