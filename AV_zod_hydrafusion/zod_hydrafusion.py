import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from model.hydranet import HydraFusion
from config import Config



class HydraFusionDataset(Dataset):
    def __init__(self, pickle_path: str):
        with open(pickle_path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

# Load Dataset & Split
data_path = "hydrafusion_inputs.pkl"

def create_bev_from_lidar(xyz: torch.Tensor, intensity: torch.Tensor, bev_size: int = 256) -> torch.Tensor:
    xyz, intensity = xyz.cpu().numpy(), intensity.cpu().numpy()
    bev = np.zeros((3, bev_size, bev_size), dtype=np.float32)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    x_idx = np.clip(((x - x.min()) / (x.max() - x.min()) * (bev_size - 1)).astype(int), 0, bev_size - 1)
    y_idx = np.clip(((y - y.min()) / (y.max() - y.min()) * (bev_size - 1)).astype(int), 0, bev_size - 1)
    for i in range(len(x)):
        bev[0, y_idx[i], x_idx[i]] = z[i]
        bev[1, y_idx[i], x_idx[i]] = intensity[i]
        bev[2, y_idx[i], x_idx[i]] += 1
    bev[2] = np.clip(bev[2] / max(bev[2].max(), 1e-6), 0, 1)
    return torch.tensor(bev).unsqueeze(0)



def box_iou(boxA, boxB):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0


def filter_predictions_by_iou(pred_boxes, gt_boxes, threshold=0.0):
    """
    Filter predicted boxes with IoU ≥ threshold with any ground truth.
    With threshold=0.0, any predicted box that has even minimal overlap will pass.
    """
    if pred_boxes.size == 0 or gt_boxes.size == 0:
        return np.zeros((0, 4)), []
    filtered_boxes = []
    indices = []
    for i, pbox in enumerate(pred_boxes):
        for gtbox in gt_boxes:
            if box_iou(pbox, gtbox) >= threshold:
                filtered_boxes.append(pbox)
                indices.append(i)
                break
    return np.array(filtered_boxes), indices

def visualize_predictions_vs_ground_truth(
        image: torch.Tensor,
        gt_boxes: torch.Tensor,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        iou_threshold: float = 0.0
    ):
    """
    Visualize ground truth (green) and predicted bounding boxes (red if IoU ≥ threshold).
    If no predicted boxes pass the IoU test, only the GT boxes are drawn.
    """
    image_np = image.squeeze(0).detach().cpu().numpy()
    image_np = np.moveaxis(image_np, 0, -1)  # CxHxW → HxWxC
    if image_np.max() > 1:
        image_np = image_np / 255.0

    fig, ax = plt.subplots(figsize=(image_np.shape[1] / 100, image_np.shape[0] / 100))
    ax.imshow(image_np)

    # Draw all ground truth boxes in green
    for gt in gt_boxes.cpu().numpy():
        x1, y1, x2, y2 = gt
        ax.add_patch(plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            edgecolor='green', linewidth=2, fill=False
        ))
        ax.text(x1, y1 - 5, "GT", fontsize=8, color='green', fontweight="bold")

    # Filter predictions based on IoU
    filtered_preds, indices = filter_predictions_by_iou(pred_boxes, gt_boxes.cpu().numpy(), threshold=iou_threshold)

    # If there are filtered predictions, draw them in red
    if len(filtered_preds) > 0:
        for box, score in zip(filtered_preds, np.array(pred_scores)[indices]):
            x1, y1, x2, y2 = box
            ax.add_patch(plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                edgecolor='red', linewidth=2, fill=False
            ))
            ax.text(x1, y1 - 5, f"", fontsize=8, color='red', fontweight="bold")
        ax.set_title(f"Green: GT, Red: Predictions")
    else:
        # If no predictions passed IoU, just alert user in plot title
        ax.set_title(f"Green: GT. No predictions")

    ax.axis("off")
    plt.show()

def validate_bbox(bbox):
    if bbox is None or bbox.numel() == 0:
        print("Skipping empty bbox.")
        return None
    if bbox.dim() < 2 or bbox.shape[1] != 4:
        print(f"Invalid bbox shape: {bbox.shape}")
        return None
    return bbox


# DataLoader Setup
train_loader = DataLoader(HydraFusionDataset(data_path), batch_size=1, shuffle=True, collate_fn=lambda x: x[0])


# Config & Model Setup
args = [
    '--activation', 'relu',
    '--dropout', '1',
    '--batch_size', '1',
    '--device', 'cpu',
    '--fusion_type', '1',
    '--pretrained', 'true',
    '--enable_radar', 'false',
    '--enable_camera', 'true',
    '--enable_lidar', 'true',
    '--enable_cam_fusion', 'false',
    '--enable_cam_lidar_fusion', 'true',
    '--enable_radar_lidar_fusion', 'false',
    '--use_custom_transforms', 'false',
    '--fusion_sweep', '0.5',   # you can lower this if you want even looser fusion
    '--resume', 'true'
]

cfg = Config(args)
device = cfg.device
model = HydraFusion(cfg).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
checkpoint = "hydrafusion_trained_frame.pth"
if os.path.exists(checkpoint):
    print("Resuming from checkpoint...")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
else:
    print("Starting training from scratch...")
num_epochs = 5
for epoch in range(num_epochs):
    model.train() 
    total_loss = 0
    print(f"\nEpoch {epoch + 1} Training Started")
    for batch_idx, input_dict in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        camera_x = input_dict.get("camera").to(device)
        lidar_xyz = input_dict.get("lidar_xyz").to(device)
        lidar_intensity = input_dict.get("lidar_intensity").to(device)
        bbox_2d = input_dict.get("bbox_2d").to(device)
        target_labels = input_dict.get("labels").to(device)

        # Skip frames missing any sensor or annotation
        if any(x is None for x in [camera_x, lidar_xyz, lidar_intensity, bbox_2d, target_labels]):
            continue

        # Ensure camera image is (1, C, H, W)
        if camera_x.dim() == 3:
            camera_x = camera_x.permute(2, 0, 1).unsqueeze(0)


        # Create and resize BEV from LiDAR
        bev_lidar_x = create_bev_from_lidar(lidar_xyz, lidar_intensity).to(device)
        bev_lidar_x = F.interpolate(
            bev_lidar_x,
            size=(camera_x.shape[-2], camera_x.shape[-1]),
            mode='bilinear',
            align_corners=False
        )

        # Validate ground-truth boxes
        if bbox_2d is None or bbox_2d.numel() == 0 or bbox_2d.shape[-1] != 4:
            continue
        bbox = validate_bbox(bbox_2d)
        if bbox is None:
            continue
        if bbox_2d is None or bbox_2d.numel() == 0 or bbox_2d.shape[-1] != 4 or target_labels is None:
            continue
        if bbox_2d.numel() == 0:
            print(f"Skipping Frame {idx+1}: No ground truth bounding boxes.")
            continue
        bbox = validate_bbox(bbox_2d)
        if bbox is None:
            continue  # Ensures invalid bbox does not proceed
        if bbox.dim() == 1 or bbox.shape[-1] != 4:
            continue  # Ensures only valid bbox tensors are used

        if bbox_2d.shape[0] == 1:  # Single box scenario
            continue
        # Prepare ground-truth dict for the model
        cam_y = [{
            'boxes': bbox_2d.to(device),
            'labels': target_labels.to(device)
        }]

        # Run the model
        output_losses, output_detections = model(
            camera_x=camera_x,
            cam_y=cam_y,
            bev_lidar_x=bev_lidar_x,
            r_lidar_x=bev_lidar_x
        )
        loss = (
            0.5 * output_losses['camera']['loss_classifier'] + 
            0.5 * output_losses['lidar']['loss_classifier'] # Reduce LiDAR influence 
	    + 1.0 * output_losses["camera_lidar"]["loss_classifier"]
        )


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() 
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}") 

    torch.save(model.state_dict(), checkpoint)
    print(f"Checkpoint saved at epoch {epoch+1}")
print("Model Training Completed & Saved.") 

weights_path = "hydrafusion_trained_frame.pth"
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

val_loader = DataLoader(HydraFusionDataset(data_path), batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
with torch.no_grad():  # Disable gradient tracking for validation
    for batch_idx, batch in enumerate(val_loader, start=1):
        input_dict = batch

        camera_x = input_dict.get("camera")
        lidar_xyz = input_dict.get("lidar_xyz")
        lidar_intensity = input_dict.get("lidar_intensity")
        bbox_2d = input_dict.get("bbox_2d")
        target_labels = input_dict.get("labels")

        if any(x is None for x in [camera_x, lidar_xyz, lidar_intensity, bbox_2d, target_labels]):
            continue

        # Ensure camera image is (1, C, H, W)
        if camera_x.dim() == 3:
            camera_x = camera_x.permute(2, 0, 1).unsqueeze(0)
        camera_x = camera_x.to(device)

        # Create and resize BEV from LiDAR
        bev_lidar_x = create_bev_from_lidar(lidar_xyz, lidar_intensity).to(device)
        bev_lidar_x = F.interpolate(
            bev_lidar_x,
            size=(camera_x.shape[-2], camera_x.shape[-1]),
            mode='bilinear',
            align_corners=False
        )

        # Validate ground-truth boxes
        if bbox_2d is None or bbox_2d.numel() == 0 or bbox_2d.shape[-1] != 4:
            continue
        bbox = validate_bbox(bbox_2d)
        if bbox is None:
            continue
        if bbox_2d is None or bbox_2d.numel() == 0 or bbox_2d.shape[-1] != 4 or target_labels is None:
            continue
        if bbox_2d.numel() == 0:
            print(f"Skipping Frame {idx+1}: No ground truth bounding boxes.")
            continue
        bbox = validate_bbox(bbox_2d)
        if bbox is None:
            continue  # Ensures invalid bbox does not proceed
        if bbox.dim() == 1 or bbox.shape[-1] != 4:
            continue  # Ensures only valid bbox tensors are used

        if bbox_2d.shape[0] == 1:  # Single box scenario
            continue
        # Prepare ground-truth dict for the model
        cam_y = [{
            'boxes': bbox_2d.to(device),
            'labels': target_labels.to(device)
        }]

        # Run the model
        output_losses, output_detections = model(
            camera_x=camera_x,
            cam_y=cam_y,
            bev_lidar_x=bev_lidar_x,
            r_lidar_x=bev_lidar_x
        )

        # Get fused1 predictions (highest-confidence fusion stage)
        _, final_detections = model.fusion_block(output_losses, output_detections, cfg.fusion_sweep)
        pred = final_detections.get('fused3', [{}])[0]

        # Extract predicted boxes and scores
        pred_boxes = pred.get('boxes', torch.zeros((0, 4))).cpu().numpy()
        pred_scores = pred.get('scores', torch.zeros((0,))).cpu().numpy()

        # Ensure correct shape for predicted boxes
        if pred_boxes.ndim == 1 and pred_boxes.shape[0] == 4:
            pred_boxes = pred_boxes.reshape(1, 4)
        elif pred_boxes.ndim != 2 or pred_boxes.shape[1] != 4:
            print(f"Frame {batch_idx}: Invalid pred_boxes shape {pred_boxes.shape}, skipping visualization.")
            continue

        # Scale predicted & ground-truth boxes to match camera resolution
        pred_boxes[:, [0, 2]] *= (camera_x.shape[-1] / 672.0)
        pred_boxes[:, [1, 3]] *= (camera_x.shape[-2] / 672.0)

        gt_boxes = bbox_2d.clone().cpu().numpy()
        gt_boxes[:, [0, 2]] *= (camera_x.shape[-1] / 672.0)
        gt_boxes[:, [1, 3]] *= (camera_x.shape[-2] / 672.0)

        print(f"\nFrame {batch_idx} — Final detections:")
        print(final_detections)

        # Visualize predictions
        visualize_predictions_vs_ground_truth(
            image=camera_x.cpu(),
            gt_boxes=bbox_2d,
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            iou_threshold=0.00
        )

print("Final Detection Visualization Complete.")
