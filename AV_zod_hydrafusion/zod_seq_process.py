import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from zod import ZodSequences, ObjectAnnotation
import zod.constants as constants
from zod.constants import Camera, Lidar, AnnotationProject
from zod.visualization.object_visualization import overlay_object_2d_box_on_image
from zod.visualization.lidar_on_image import visualize_lidar_on_image
import cv2
# Dataset setup
dataset_root = r"F:\zod_mini"  # Your local path to ZOD Mini
version = "mini"
zod_sequences = ZodSequences(dataset_root=dataset_root, version=version)

# Retrieve all sequence IDs and filter only those from "00002"
sequence_ids = [seq_id for seq_id in zod_sequences.get_all_ids() if "000002" in seq_id]
filtered_sequences = [seq_id for seq_id in sequence_ids if "000002" in seq_id]
# Function to clean file paths
def sanitize_path(filepath):
    drive, path = os.path.splitdrive(filepath)
    cleaned_path = path.replace(":", "_")  # Clean colon path
    return os.path.join(drive, cleaned_path)

# Store HydraFusion input dictionaries
hydrafusion_inputs = []
processed_timestamps = set()
stored_timestamps = set()
# Iterate through filtered sequences
for seq_id in sequence_ids:
    try:
        print(f"\nProcessing Sequence ID: {seq_id}")
        seq = zod_sequences[seq_id]


        # Get camera-lidar mapping for this sequence
        frames = seq.info.get_camera_lidar_map()

        for camera_frame, lidar_frame in frames:
            frame_timestamp = camera_frame.time 
            if frame_timestamp in stored_timestamps:
                print(f"Skipping duplicate frame at time {frame_timestamp}")
                continue
        
            stored_timestamps.add(frame_timestamp) 
            if camera_frame.time in processed_timestamps:
                print(f"Skipping duplicate frame at time {camera_frame.time}")
                continue
            processed_timestamps.add(camera_frame.time)
       
            try:
                # **IMAGE PROCESSING**
                input_dict = {}
                image_path = sanitize_path(camera_frame.filepath)
                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
                    continue

                # Open image and record original size
                with Image.open(image_path) as img:
                    original_width, original_height = img.size
                    img = img.convert("RGB").resize((1024, 512))  # Resize for memory efficiency
                    image = np.array(img, dtype=np.uint8)  # Convert to NumPy array

                scale_x = 1024 / original_width
                scale_y = 512 / original_height


                input_dict["file_name"] = image_path
                # Retrieve annotations
                annotations = seq.get_annotation(AnnotationProject.OBJECT_DETECTION)
                vehicle_annotations = [anno for anno in annotations if anno.name.lower() == "vehicle"]

                bbox_2d_list = []
                bbox_3d_list = []
                for anno in vehicle_annotations:
                    annotation_2d = anno.box2d
                    annotation_3d = anno.box3d
                    if annotation_2d is None:
                        continue
                    x1, y1, x2, y2 = annotation_2d.xyxy
                    x1 *= scale_x
                    x2 *= scale_x
                    y1 *= scale_y
                    y2 *= scale_y
                    bbox_tensor = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
                    bbox_2d_list.append(bbox_tensor)

                    #image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), thickness=2)

                    if annotation_3d:
                        bbox_3d_list.append(torch.tensor(annotation_3d.center, dtype=torch.float32))

                # Store image-related inputs
                input_dict["camera"] = torch.tensor(image, dtype=torch.float32)
                input_dict["bbox_2d"] = torch.stack(bbox_2d_list) if bbox_2d_list else None
                input_dict["bbox_3d"] = torch.stack(bbox_3d_list) if bbox_3d_list else None
                input_dict["labels"] = torch.ones(len(vehicle_annotations), dtype=torch.int64)

                # **LIDAR PROCESSING**
                lidar_path = sanitize_path(lidar_frame.filepath)
                if not os.path.exists(lidar_path):
                    print(f"LiDAR file not found: {lidar_path}")
                    continue

                pc = np.load(lidar_path, allow_pickle=True)
                pc = np.array(pc.tolist())
    

                input_dict["lidar_xyz"] = torch.tensor(pc[:, :3], dtype=torch.float32)
                input_dict["lidar_intensity"] = torch.tensor(pc[:, 4], dtype=torch.float32)

                # Store calibration info
                calibrations = seq.calibration
                cam_calib = calibrations.cameras[Camera.FRONT]
                lidar_calib = calibrations.lidars[Lidar.VELODYNE]

                input_dict.update({
                    "camera_intrinsics": torch.tensor(cam_calib.intrinsics, dtype=torch.float32),
                    "camera_extrinsics": torch.tensor(cam_calib.extrinsics.transform, dtype=torch.float32),
                    "lidar_extrinsics": torch.tensor(lidar_calib.extrinsics.transform, dtype=torch.float32),
                })

                hydrafusion_inputs.append(input_dict)

                # Optional visualization
                #plt.figure(figsize=(10, 5))
                #plt.axis("off")
                #plt.imshow(image)
                #plt.title(f"Sequence ID: {seq_id}, Frame Time: {camera_frame.time}")
                #plt.show()

            except Exception as frame_error:
                print(f"Error processing frame {camera_frame.time}: {frame_error}")
                continue

    except Exception as seq_error:
        print(f"General error for Sequence ID {seq_id}: {seq_error}")
        continue

# Save processed data
output_file = "hydrafusion_inputs_sequences.pkl"
with open(output_file, "wb") as f:
    pickle.dump(hydrafusion_inputs, f)

print(f"\nSuccessfully processed {len(hydrafusion_inputs)} frames from '000002'. Data saved to '{output_file}'")
