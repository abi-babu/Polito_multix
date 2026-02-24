import os
import torch
import copy
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from model.hydranet import HydraFusion
from config import Config
import torch.nn.functional as F
from torch.utils.data import random_split
from collections import defaultdict

#Split dataset for training and evluation
def split_dataset(dataset, split_ratio=0.8):
    train_len = int(len(dataset) * split_ratio)
    val_len = len(dataset) - train_len
    return random_split(dataset, [train_len, val_len])

#Validated the ground truth box for -ve and zero values
def validate_bbox(bbox):
    if bbox is None:
        print("Invalid bbox: None")
        return None

    # Squeeze batch dimension if present
    if bbox.dim() == 3 and bbox.shape[0] == 1:
        bbox = bbox.squeeze(0)

    if bbox.numel() == 0 or bbox.dim() != 2 or bbox.shape[1] != 4:
        print("Invalid or empty bbox after squeeze.")
        return None

    # Check for negative area
    x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    invalid_mask = (x2 <= x1) | (y2 <= y1)
    if invalid_mask.any():
        print(f"Found {invalid_mask.sum().item()} invalid boxes with negative area.")
        bbox = bbox[~invalid_mask]

    if bbox.numel() == 0:
        print("All boxes were invalid after filtering.")
        return None

    return bbox

#Load the dataset.pkl file into the model
class HydraFusionDataset(Dataset):
    def __init__(self, pickle_path):
        with open(pickle_path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

#Conditions to check the prediction results 
def filter_predictions_by_iou(pred_boxes, gt_boxes, threshold=0.2):
    if pred_boxes.size == 0 or gt_boxes.size == 0:
        return np.zeros((0, 4)), []
    filtered_boxes, indices = [], []
    for i, pbox in enumerate(pred_boxes):
        for gtbox in gt_boxes:
            if box_iou(pbox, gtbox) >= threshold:
                filtered_boxes.append(pbox)
                indices.append(i)
                break
    return np.array(filtered_boxes), indices

def box_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0

#Convert the dataset.pkl into model accepted tensor values  
def create_bev_from_lidar(xyz, intensity, bev_size=256):
    # Ensure tensors are on CPU and flattened
    xyz = xyz.cpu().numpy().reshape(-1, 3)
    intensity = intensity.cpu().numpy().reshape(-1)

    bev = np.zeros((3, bev_size, bev_size), dtype=np.float32)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # Normalize and index
    x_idx = ((x - x.min()) / (x.max() - x.min() + 1e-6) * (bev_size - 1)).astype(int)
    y_idx = ((y - y.min()) / (y.max() - y.min() + 1e-6) * (bev_size - 1)).astype(int)
    x_idx = np.clip(x_idx, 0, bev_size - 1)
    y_idx = np.clip(y_idx, 0, bev_size - 1)

    # Sanity check
    assert x_idx.shape == y_idx.shape == intensity.shape == z.shape, \
        f"Shape mismatch: x_idx {x_idx.shape}, y_idx {y_idx.shape}, intensity {intensity.shape}, z {z.shape}"

    # Assign values
    bev[0, y_idx, x_idx] = z
    bev[1, y_idx, x_idx] = intensity
    np.add.at(bev[2], (y_idx, x_idx), 1)
    bev[2] = np.clip(bev[2] / max(bev[2].max(), 1e-6), 0, 1)

    return torch.from_numpy(bev).unsqueeze(0)

def create_bev_from_radar(xyz, bev_size=256):
    xyz = xyz.cpu().numpy()
    bev = np.zeros((1, bev_size, bev_size), dtype=np.float32)
    x, y = xyz[:, 0], xyz[:, 1]
    epsilon = 1e-6
    x_idx = np.clip(((x - x.min()) / (x.max() - x.min() + epsilon) * (bev_size - 1)).astype(int), 0, bev_size - 1)
    y_idx = np.clip(((y - y.min()) / (y.max() - y.min() + epsilon) * (bev_size - 1)).astype(int), 0, bev_size - 1)
    for i in range(len(x)):
        bev[0, y_idx[i], x_idx[i]] += 1
    bev[0] = np.clip(bev[0] / max(bev[0].max(), 1e-6), 0, 1)
    return torch.tensor(bev).unsqueeze(0)


def flatten_losses(loss_dict):
    flat = {}
    for k, v in loss_dict.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                flat[f"{k}.{subk}"] = subv
        else:
            flat[k] = v
    return flat

#Federated training on each client 
def train_on_client(model, dataloader, cfg, device, epochs=3):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epoch_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        valid_batches = 0

        for input_dict in dataloader:
            camera_x = input_dict.get("camera")
            lidar_xyz = input_dict.get("lidar_xyz")
            lidar_intensity = input_dict.get("lidar_intensity")
            radar_xyz = input_dict.get("radar_xyz")
            bbox_2d = validate_bbox(input_dict.get("bbox_2d"))
            target_labels = input_dict.get("labels")

            if bbox_2d is None or bbox_2d.shape[0] <= 1:
                continue

            if camera_x.dim() == 4 and camera_x.shape[-1] == 3:
                camera_x = camera_x.permute(0, 3, 1, 2)

            camera_x = camera_x.to(device)

            bev_lidar_x = create_bev_from_lidar(lidar_xyz, lidar_intensity)
            bev_lidar_x = F.interpolate(bev_lidar_x, size=(camera_x.shape[-2], camera_x.shape[-1]),
                                        mode='bilinear').to(device)

            radar_x = create_bev_from_radar(radar_xyz)
            radar_x = F.interpolate(radar_x, size=(camera_x.shape[-2], camera_x.shape[-1]),
                                    mode='bilinear').to(device)

            cam_y = [{'boxes': bbox_2d.to(device), 'labels': target_labels.to(device)}]
            radar_y = [{'boxes': bbox_2d.to(device), 'labels': target_labels.to(device)}]

            output_losses, _ = model(
                rightcamera_x=camera_x, leftcamera_x=camera_x,
                cam_y=cam_y,
                bev_lidar_x=bev_lidar_x,
                r_lidar_x=bev_lidar_x,
                radar_x=[radar_x.squeeze(0)],
                radar_y=radar_y
            )

            if not output_losses:
                continue

            flat_losses = flatten_losses(output_losses)
            loss = sum(v for v in flat_losses.values())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            valid_batches += 1

        if valid_batches > 0:
            avg_epoch_loss = epoch_loss / valid_batches
            epoch_losses.append(avg_epoch_loss)
            print(f"  Epoch {epoch+1} - Avg Loss: {avg_epoch_loss:.4f}")

    # return the mean across all epochs for this client
    avg_client_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
    return model.state_dict(), avg_client_loss

#Federated averaging on server
def federated_averaging(client_weights):
    avg_weights = copy.deepcopy(client_weights[0])
    for key in avg_weights.keys():
        for i in range(1, len(client_weights)):
            avg_weights[key] += client_weights[i][key]
        avg_weights[key] = avg_weights[key] / len(client_weights)
    return avg_weights

#Main method 
def run_federated_training(pickle_path, num_clients=2, rounds=5, train_val_split=0.8):
    args = [
        '--activation', 'relu', '--dropout', '1', '--batch_size', '1', '--device', 'cuda',
        '--fusion_type', '1', '--pretrained', 'false', '--enable_radar', 'true',
        '--enable_camera', 'true', '--enable_lidar', 'true', '--enable_cam_fusion', 'false',
        '--enable_cam_lidar_fusion', 'true', '--enable_radar_lidar_fusion', 'true',
        '--use_custom_transforms', 'true', '--fusion_sweep', '0.5', '--resume', 'true'
    ]
    cfg = Config(args)
    device = cfg.device
    model_path = "saved_global_trained_model.pth" #save the trained model

    full_dataset = HydraFusionDataset(pickle_path)
    split_size = len(full_dataset) // num_clients
    client_subsets = [Subset(full_dataset, range(i * split_size, (i + 1) * split_size)) for i in range(num_clients)]
    client_train_sets, client_val_sets = zip(*[split_dataset(subset, train_val_split) for subset in client_subsets])

    global_model = HydraFusion(cfg).to(device)

    if os.path.exists(model_path):
        print("Loading saved global model weights...")
        global_model.load_state_dict(torch.load(model_path))
    else:
        print("No saved model found — starting fresh.")

    previous_total_loss = None
    best_map = 0.0

    for round_num in range(rounds):
        print(f"\n--- Round {round_num + 1} ---")
        client_weights, total_losses = [], []

        # Train on each client
        for i, train_set in enumerate(client_train_sets):
            client_model = copy.deepcopy(global_model)
            dataloader = DataLoader(train_set, batch_size=1, shuffle=True)
            weights, avg_client_loss = train_on_client(client_model, dataloader, cfg, device)
            client_weights.append(weights)
            total_losses.append(avg_client_loss)

        # Aggregate
        global_weights = federated_averaging(client_weights)
        global_model.load_state_dict(global_weights)

        # Loss reporting
        round_total_loss = sum(total_losses)
        if previous_total_loss is not None and previous_total_loss > 0:
            loss_pct = (round_total_loss / previous_total_loss) * 100
            print(f"Round {round_num + 1} - Total Loss: {round_total_loss:.4f} "
                  f"({loss_pct:.2f}% of previous round)")
        else:
            print(f"Round {round_num + 1} - Total Loss: {round_total_loss:.4f}")
        previous_total_loss = round_total_loss

        # Save best model
        if avg_val_map > best_map:
            best_map = avg_val_map
            torch.save(global_model.state_dict(), "best_global_model.pth")
            print(f"New best model saved (mAP={best_map:.4f})")

    # Final save
    torch.save(global_model.state_dict(), model_path)
    print(f"\nGlobal model saved to {model_path}")

if __name__ == "__main__":
    run_federated_training("test.pkl", num_clients=2, rounds=5)