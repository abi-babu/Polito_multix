import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from model.hydranet import HydraFusion
from config import Config
from torch.optim import Adam
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, average_precision_score

#Loading the dtaste.pkl file in model
class HydraFusionDataset(Dataset):
    def __init__(self, pickle_path):
        with open(pickle_path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def flatten_losses(loss_dict):
    flat_losses = []
    for value in loss_dict.values():
        if isinstance(value, dict):
            flat_losses.extend(flatten_losses(value))  # recursive flatten
        elif torch.is_tensor(value):
            flat_losses.append(value)
    return flat_losses

#Verify the prediction 
def compute_map(all_preds, all_scores, all_labels, all_gt_boxes, iou_threshold=0.2):
    aps = []
    for cls in np.unique(all_labels):
        cls_preds = [box for box, label in zip(all_preds, all_labels) if label == cls]
        cls_scores = [score for score, label in zip(all_scores, all_labels) if label == cls]
        cls_gt = [box for box, label in zip(all_gt_boxes, all_labels) if label == cls]

        if len(cls_gt) == 0 or len(cls_preds) == 0:
            continue

        y_true = []
        y_scores = []

        for pred_box, score in zip(cls_preds, cls_scores):
            matched = any(box_iou(pred_box, gt_box) >= iou_threshold for gt_box in cls_gt)
            y_true.append(1 if matched else 0)
            y_scores.append(score)

        if len(set(y_true)) > 1:
            ap = average_precision_score(y_true, y_scores)
            aps.append(ap)

    return np.mean(aps) if aps else 0.0

#Convert the dataset.pkl file to adapt the model tensor format
def create_bev_from_lidar(xyz, intensity, bev_size=256):
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

#Evaluation 
def box_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0

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

def validate_bbox(bbox):
    if bbox is None or bbox.numel() == 0 or bbox.dim() < 2 or bbox.shape[1] != 4:
        print("nvalid or empty bbox.")
        return None
    return bbox

#main evaluation 
def visualize_predictions_vs_ground_truth(image, gt_boxes, pred_boxes, pred_scores, iou_threshold=0.2):
    image_np = image.squeeze(0).detach().cpu().numpy()
    image_np = np.moveaxis(image_np, 0, -1)
    image_np = image_np / 255.0 if image_np.max() > 1 else image_np
    fig, ax = plt.subplots(figsize=(image_np.shape[1] / 100, image_np.shape[0] / 100))
    ax.imshow(image_np)
    for gt in gt_boxes:
        x1, y1, x2, y2 = gt.cpu().numpy()
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='green', linewidth=2, fill=False))
        ax.text(x1, y1 - 5, "GT", fontsize=8, color='green', fontweight="bold")
    filtered_preds, indices = filter_predictions_by_iou(pred_boxes, gt_boxes, threshold=iou_threshold)
    for box, score in zip(filtered_preds, np.array(pred_scores)[indices]):
        x1, y1, x2, y2 = box
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', linewidth=2, fill=False))
        ax.text(x1, y1 - 5, f"{score:.2f}", fontsize=8, color='red', fontweight="bold")
    ax.set_title("Green: GT, Red: Predictions" if len(filtered_preds) > 0 else "Green: GT. No predictions")
    ax.axis("off")
    #plt.show()
    plt.close(fig)


args = [
    '--activation', 'relu',
    '--dropout', '1',
    '--batch_size', '1',
    '--device', 'cuda',
    '--fusion_type', '1',
    '--pretrained', 'false',
    '--enable_radar', 'true',
    '--enable_camera', 'true',
    '--enable_lidar', 'true',
    '--enable_cam_fusion', 'true',
    '--enable_cam_lidar_fusion', 'true',
    '--enable_radar_lidar_fusion', 'true',
    '--use_custom_transforms', 'true',
    '--fusion_sweep', '0.5',
    '--resume', 'true'
]

cfg = Config(args)
device = cfg.device
model = HydraFusion(cfg).to(device)
checkpoint = torch.load("saved_global_model.pth")
model.load_state_dict(checkpoint, strict=False)

fused_accuracy = defaultdict(lambda: {'tp': 0, 'gt': 0})

def evaluate_model(model, dataset, cfg, device, context="default"):
    model.eval()
    all_preds, all_scores, all_labels, all_gt_boxes = [], [], [], []
    for input_dict in dataset:
        camera_x = input_dict.get("camera")
        lidar_xyz = input_dict.get("lidar_xyz")
        lidar_intensity = input_dict.get("lidar_intensity")
        radar_xyz = input_dict.get("radar_xyz")
        bbox_2d = validate_bbox(input_dict.get("bbox_2d"))
        target_labels = input_dict.get("labels")

        if bbox_2d is None or bbox_2d.shape[0] <= 1:
            continue

        if camera_x.dim() == 3:
            camera_x = camera_x.permute(2, 0, 1).unsqueeze(0)
        camera_x = camera_x.to(device)

        bev_lidar_x = create_bev_from_lidar(lidar_xyz.to(device), lidar_intensity.to(device)).to(device)
        bev_lidar_x = F.interpolate(bev_lidar_x, size=(camera_x.shape[-2], camera_x.shape[-1]), mode='bilinear')

        radar_x = create_bev_from_radar(radar_xyz.to(device)).to(device)
        radar_x = F.interpolate(radar_x, size=(camera_x.shape[-2], camera_x.shape[-1]), mode='bilinear')

        cam_y = [{'boxes': bbox_2d.to(device), 'labels': target_labels.to(device)}]
        radar_y = [{'boxes': bbox_2d.to(device), 'labels': target_labels.to(device)}]
        with torch.no_grad():
            output_losses, output_detections = model(
                rightcamera_x=camera_x, leftcamera_x=camera_x,
                cam_y=cam_y,
                bev_lidar_x=bev_lidar_x,
                r_lidar_x=bev_lidar_x,
                radar_x=[radar_x.squeeze(0)],
                radar_y=radar_y
            )
            _, final_detections = model.fusion_block(output_losses, output_detections, cfg.fusion_sweep)

        pred = final_detections.get('fused2', [{}])[0]
        pred_boxes = pred.get('boxes', torch.zeros((0, 4))).cpu().numpy()
        pred_scores = pred.get('scores', torch.zeros((0,))).cpu().numpy()
        if pred_boxes.ndim != 2 or pred_boxes.shape[1] != 4:
            print("kipping malformed prediction boxes.")
            continue

        pred_boxes[:, [0, 2]] *= camera_x.shape[-1] / 672.0
        pred_boxes[:, [1, 3]] *= camera_x.shape[-2] / 672.0
        pred_boxes[:, [1, 3]] += 150
        pred_labels = pred.get('labels', torch.zeros((0,), dtype=torch.int64)).cpu().numpy()

        gt_boxes = bbox_2d.clone().cpu().numpy()
        gt_boxes[:, [0, 2]] *= camera_x.shape[-1] / 672.0
        gt_boxes[:, [1, 3]] *= camera_x.shape[-2] / 672.0

        filtered_preds, _ = filter_predictions_by_iou(pred_boxes, gt_boxes, threshold=0.2)
        tp_count = len(filtered_preds)
        gt_count = bbox_2d.shape[0]
        pred_labels = pred.get('labels', torch.zeros((0,), dtype=torch.int64)).cpu().numpy()
        gt_labels = target_labels.cpu().numpy()
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            all_preds.append(box)
            all_scores.append(score)
            all_labels.append(label)

        for box, label in zip(gt_boxes, gt_labels):
            all_gt_boxes.append(box)
            all_labels.append(label)  # reuse same label list

        fused_accuracy[context]['tp'] += tp_count
        fused_accuracy[context]['gt'] += gt_count

        #print("Raw pred_boxes:", pred_boxes)
        #print("Raw pred_scores:", pred_scores)
        visualize_predictions_vs_ground_truth(
            image=camera_x.cpu(),
            gt_boxes=bbox_2d,
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            iou_threshold=0.2
        )
    map_score = compute_map(all_preds, all_scores, all_labels, all_gt_boxes, iou_threshold=0.2)
    print(f"\nMean Average Precision (mAP @ IoU=0.5): {map_score:.4f}")

    # Print summary: optional
    print(f"\n{'Context':<20} {'Fused Accuracy (%)':<20}")
    for ctx, stats in fused_accuracy.items():
        acc = (stats['tp'] / stats['gt']) * 100 if stats['gt'] > 0 else 0.0
        print(f"{ctx:<20} {acc:<20.2f}")

if __name__ == "__main__":
    test_data_path = "test.pkl"
    dataset = HydraFusionDataset(test_data_path)

    # Use only the first N samples (e.g., 10)
    subset_size = 15
    subset = [dataset[i] for i in range(min(subset_size, len(dataset)))]

    evaluate_model(model, subset, cfg, device, context="motorway_sunny")