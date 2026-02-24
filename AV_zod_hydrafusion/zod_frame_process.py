import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from zod import ZodFrames, ObjectAnnotation
import zod.constants as constants
from zod.constants import Camera, Lidar,Anonymization, AnnotationProject
from zod.visualization.object_visualization import overlay_object_2d_box_on_image
from zod.data_classes.calibration import LidarCalibration, CameraCalibration, Calibration

# Dataset setup 
dataset_root = r"F:\zod_mini"  # Your local path to ZOD Mini
version = "mini"
zod_frames = ZodFrames(dataset_root=dataset_root, version=version)

# Retrieve all frame IDs
frame_ids = zod_frames.get_all_ids()

# Store HydraFusion input dictionaries
hydrafusion_inputs = []

# Function to display annotated image
def plot_annotations(image, frame_id):
    plt.figure(figsize=(10, 5))
    plt.axis("off")
    plt.imshow(image)
    plt.title(f"Frame ID: {frame_id} - Vehicle Detection")
    plt.show()

# Iterate through all frames
for frame_id in frame_ids:
    try:
        print(f"\nProcessing Frame ID: {frame_id}")
        zod_frame = zod_frames[frame_id]
        input_dict = {}

        # --- IMAGE PROCESSING & VEHICLE ANNOTATION ---
        try:
            camera_core_frame = zod_frame.info.get_key_camera_frame(Anonymization.DNAT)
            image_path = camera_core_frame.filepath
            drive, path = os.path.splitdrive(image_path)
            image_path = path.replace(":", "_")  # Clean colon path
            image_path = os.path.join(drive, image_path)

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = np.array(Image.open(image_path))

            # Retrieve annotations
            annotations = zod_frame.get_annotation(AnnotationProject.OBJECT_DETECTION)
            vehicle_annotations = [anno for anno in annotations if anno.name.lower() == "vehicle"]

            if vehicle_annotations:
                bbox_2d_list = []
                bbox_3d_list = []

                for anno in vehicle_annotations:
                    annotation_2d = anno.box2d
                    annotation_3d = anno.box3d
                    #image = overlay_object_2d_box_on_image(image, annotation_2d, color=(255, 0, 0), line_thickness=10)
                    bbox_2d_list.append(torch.tensor(annotation_2d.xyxy, dtype=torch.float32))
                    if annotation_3d:
                        bbox_3d_list.append(torch.tensor(annotation_3d.center, dtype=torch.float32))

                # Store inputs
                input_dict["camera"] = torch.tensor(image, dtype=torch.float32)
                input_dict["bbox_2d"] = torch.stack(bbox_2d_list)
                input_dict["bbox_3d"] = torch.stack(bbox_3d_list) if bbox_3d_list else None
                input_dict["labels"] = torch.ones(len(vehicle_annotations), dtype=torch.int64)

                # Get camera calibration
                calibrations = zod_frame.calibration
                cam_calib = calibrations.cameras[Camera.FRONT]
                lidar_calib = calibrations.lidars[Lidar.VELODYNE]

# Prepare the dictionary
                input_dict.update({
                    "camera_intrinsics": torch.tensor(cam_calib.intrinsics, dtype=torch.float32),
                    "camera_extrinsics": torch.tensor(cam_calib.extrinsics.transform, dtype=torch.float32),
                    "lidar_extrinsics": torch.tensor(lidar_calib.extrinsics.transform, dtype=torch.float32),
                })
                # Optional visualization
                #plot_annotations(image, frame_id)

            else:
                print(f"No vehicle annotations found for Frame {frame_id}")
                continue  # Skip frames with no vehicle

        except Exception as img_error:
            print(f"Error in image block (Frame {frame_id}): {img_error}")
            continue

        # --- LIDAR PROCESSING ---
        try:
            lidar_core_frame = zod_frame.info.get_key_lidar_frame()
            lidar_path = lidar_core_frame.filepath
            drive, path = os.path.splitdrive(lidar_path)
            lidar_path = path.replace(":", "_")  # Clean colon path
            lidar_path = os.path.join(drive, lidar_path)

            if not os.path.exists(lidar_path):
                raise FileNotFoundError(f"LiDAR file not found: {lidar_path}")

            pc = np.load(lidar_path, allow_pickle=True)
            pc = np.array(pc.tolist())

            input_dict["lidar_xyz"] = torch.tensor(pc[:, :3], dtype=torch.float32)
            input_dict["lidar_intensity"] = torch.tensor(pc[:, 4], dtype=torch.float32)

            hydrafusion_inputs.append(input_dict)

        except Exception as lidar_error:
            print(f"Error in LiDAR block (Frame {frame_id}): {lidar_error}")
            continue

    except Exception as e:
        print(f"General error for Frame {frame_id}: {e}")
        continue

# Save processed data
output_file = "hydrafusion_inputs.pkl"
with open(output_file, "wb") as f:
    pickle.dump(hydrafusion_inputs, f)

print(f"\nSuccessfully processed {len(hydrafusion_inputs)} frames. Data saved to '{output_file}'")
