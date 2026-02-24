import torch
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import torchvision.transforms as T

# RADIATE SDK
import sys
sys.path.insert(0, '..')
from radiate_sdk.radiate import Sequence
from config import Config
from model.hydranet import HydraFusion

# Dataset and config paths
root_path = r"F:"
sequence_name = "tiny_foggy"
dt = 0.25

seq = Sequence(os.path.join(root_path, sequence_name), config_file=r"C:\Users\lenovo\hydrafusion\radiate_sdk\config\config.yaml")

# HydraFusion Config: camera disabled, radar + lidar enabled
args = [
    '--activation', 'relu',
    '--dropout', '1',
    '--batch_size', '1',
    '--device', 'cpu',
    '--fusion_type', '1',
    '--pretrained', 'true',
    '--enable_radar', 'true',
    '--enable_camera', 'true',
    '--enable_lidar', 'true',
    '--enable_cam_fusion', 'true',
    '--enable_cam_lidar_fusion', 'true',
    '--enable_radar_lidar_fusion', 'true',
    '--use_custom_transforms', 'false',
    '--fusion_sweep', '0.5',
    '--resume', 'true'
]

cfg = Config(args)
device = cfg.device
model = HydraFusion(cfg)
model.eval()

# Utility Functions
def to_tensor(data):
    return torch.tensor(data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

def process_annotations(annotations, source='camera'):
    boxes, labels = [], []
    for ann in annotations:
        if 'id' in ann:
            label = torch.tensor([ann['id']], dtype=torch.long).to(device)
            points = np.array(ann.get('bbox_3d', []))
            if points.ndim == 2 and points.shape[1] >= 2:
                x1, y1 = np.min(points[:, 0]), np.min(points[:, 1])
                x2, y2 = np.max(points[:, 0]), np.max(points[:, 1])
                boxes.append(torch.tensor([x1, y1, x2, y2], dtype=torch.float32).to(device))
                labels.append(label)

    return {
        'boxes': torch.stack(boxes) if boxes else torch.zeros((0, 4), dtype=torch.float32).to(device),
        'labels': torch.cat(labels, dim=0) if labels else torch.zeros((0,), dtype=torch.long).to(device)
    }
def compute_iou(box1, box2):
    """Compute IoU between two boxes: [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def draw_boxes(image, gt_boxes, pred_boxes, pred_labels, iou_threshold=0.3):
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.imshow(image)

    # Draw GT boxes (green)
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    # Draw only predictions that are close to GT (IoU >= threshold)
    for i, pbox in enumerate(pred_boxes):
        for gt_box in gt_boxes:
            if compute_iou(pbox, gt_box) >= iou_threshold:
                x1, y1, x2, y2 = pbox
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, f"Pred", color='r', fontsize=8)
                break  # Only draw once per match

    ax.set_title("Green: Ground Truth, Red:Predictions")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def compute_map(gt_boxes, pred_boxes, iou_thresholds=[0.1, 0.3]):
    """Compute mAP based on IoU thresholds."""
    ap_values = []
    
    for iou_thresh in iou_thresholds:
        tp, fp, num_gt = 0, 0, len(gt_boxes)

        for pred_box in pred_boxes:
            matched = False
            for gt_box in gt_boxes:
                if compute_iou(pred_box, gt_box) >= iou_thresh:
                    matched = True
                    break

            if matched:
                tp += 1
            else:
                fp += 1

        # Precision and recall
        precision = tp / max(tp + fp, 1)  # Avoid div by zero
        recall = tp / max(num_gt, 1)
        
        ap_values.append(precision)  # Approximate AP as precision at a given IoU

    return sum(ap_values) / len(ap_values)  # mAP = mean AP across IoU thresholds

# --- Sensor Fusion & Inference with mAP ---
for t in np.arange(seq.init_timestamp, seq.end_timestamp, dt):
    output = seq.get_from_timestamp(t)
    if not output:
        continue
    camera_img = output['sensors']['camera_right_rect']
    rightcamera_x = to_tensor(output['sensors']['camera_right_rect'])
    leftcamera_x = to_tensor(output['sensors']['camera_left_rect'])
    radar_x = to_tensor(output['sensors']['radar_cartesian'])
    bev_lidar_x = to_tensor(output['sensors']['lidar_bev_image'])
    r_lidar_x = to_tensor(output['sensors']['proj_lidar_right'])

    cam_y = process_annotations(output['annotations']['camera_right_rect'])
    radar_y = process_annotations(output['annotations']['radar_cartesian'])

    with torch.no_grad():
        output_losses, output_detections = model(
            radar_x=radar_x, bev_lidar_x=bev_lidar_x, r_lidar_x=r_lidar_x,
            cam_y=[cam_y], radar_y=[radar_y], rightcamera_x = rightcamera_x, leftcamera_x = leftcamera_x
        )
        np.savez(f"semantic_outputs/{t:.2f}.npz", features=semantic_features.cpu().numpy())
        _, final_detections = model.fusion_block(
            output_losses, output_detections, cfg.fusion_sweep
        )
        pred = final_detections.get('fused2', [{}])[0]
        pred_boxes = pred.get('boxes', torch.zeros((0, 4))).cpu().numpy()
        pred_labels = pred.get('labels', torch.zeros((0,), dtype=torch.long)).cpu().numpy()

        # Compute mAP
        map_score = compute_map(cam_y['boxes'].cpu().numpy(), pred_boxes)
        print(f"mAP Score: {map_score:.4f}")

        # Visualization
        draw_boxes(camera_img, cam_y['boxes'].cpu().numpy(), pred_boxes, pred_labels)

        print("box",cam_y['boxes'])

        print("GT Labels:", cam_y['labels'].tolist())
        print("Pred Labels:", pred_labels.tolist())
