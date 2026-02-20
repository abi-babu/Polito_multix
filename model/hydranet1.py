import torch
import torch.nn as nn
import torch.nn.functional as F
from .branch import RadarBranch, CameraBranch, LidarBranch, DualCameraFusionBranch, CameraLidarFusionBranch, RadarLidarFusionBranch, ResNetTail, RFHeatBranch, RFSpectoBranch, RFFusionBranch
from .stem import RadarStem, CameraStem, LidarStem, RFHeatStem, RFSpectoStem
from .fusion import FusionBlock
from torchvision.models.resnet import BasicBlock
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from typing import List
from torchvision.models.detection.image_list import ImageList



'''This file defines our HydraNet-based sensor fusion architecture.'''
class HydraFusion(nn.Module):

    def __init__(self, config):
        super(HydraFusion, self).__init__()
        self.config = config
        self.dropout = config.dropout
        self.activation = F.relu if config.activation == 'relu' else F.leaky_relu
        self.initialize_transforms()
        self.initialize_stems()
        self.initialize_branches()
        #self.fusion_block = FusionBlock(config, fusion_type=1, weights=self.num_branches*[1], iou_thr=0.4, skip_box_thr=0.01, sigma=0.5, alpha=1)
        #self.waveform_regressor = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),nn.Flatten(),nn.Linear(512, 7500),nn.Sigmoid())
        self.fusion_block = FusionBlock(config, fusion_type=1, weights=self.num_branches*[1], iou_thr=0.4, skip_box_thr=0.01, sigma=0.5, alpha=1)
        self.waveform_fusion = WaveformFusion(self.waveform_length)

    '''initializes the normalization/resizing transforms applied to input images.'''
    def initialize_transforms(self):
        if self.config.use_custom_transforms:
            self.image_transform = GeneralizedRCNNTransform(min_size=376, max_size=1000, image_mean=[88.12744903564453,90.560546875,90.5104751586914], image_std=[66.74466705322266,74.3885726928711,75.6873779296875])
            self.radar_transform = GeneralizedRCNNTransform(min_size=376, max_size=1000, image_mean=[15.557413101196289,15.557413101196289,15.557413101196289], image_std=[18.468725204467773,18.468725204467773,18.468725204467773])
            self.lidar_transform = GeneralizedRCNNTransform(min_size=376, max_size=1000, image_mean=[2.1713976860046387,2.1713976860046387,2.1713976860046387], image_std=[20.980266571044922,20.980266571044922,20.980266571044922])
            self.fwd_lidar_transform = GeneralizedRCNNTransform(min_size=376, max_size=1000, image_mean=[0.0005842918762937188,0.0005842918762937188,0.0005842918762937188], image_std=[0.10359727591276169,0.10359727591276169,0.10359727591276169])
            #RFChanges
            self.rf_heatmap_transform = GeneralizedRCNNTransform(min_size=376, max_size=1000, image_mean=[88.12744903564453,90.560546875,90.5104751586914], image_std=[66.74466705322266,74.3885726928711,75.6873779296875])
            self.rf_spectrogram_transform = GeneralizedRCNNTransform(min_size=376, max_size=1000, image_mean=[88.12744903564453,90.560546875,90.5104751586914], image_std=[66.74466705322266,74.3885726928711,75.6873779296875])


        else:
            self.transform = GeneralizedRCNNTransform(min_size=376, max_size=1000, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]) #from ImageNet
            self.image_transform = self.transform
            self.radar_transform = self.transform
            self.lidar_transform = self.transform
            self.fwd_lidar_transform = self.transform
            #RFChanges
            self.rf_heatmap_transform = self.transform
            self.rf_spectrogram_transform = self.transform


    '''initializes the stem modules as the first blocks of resnet-18.'''
    def initialize_stems(self):
        if self.config.enable_radar:
            self.radar_stem = RadarStem(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained).to(self.config.device)  # TODO:define these config values in config.py
        if self.config.enable_camera:  
            self.camera_stem = CameraStem(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained).to(self.config.device)
        if self.config.enable_lidar:
            self.lidar_stem = LidarStem(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained).to(self.config.device)
        if self.config.enable_cam_lidar_fusion:
            self.fwd_lidar_stem = LidarStem(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained).to(self.config.device)
        #RFChanges
        if self.config.enable_rf_heatmap:
            self.rf_heatmap_stem = RFHeatStem(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained).to(self.config.device)
        if self.config.enable_rf_spectrogram:
            self.rf_spectrogram_stem = RFSpectoStem(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained).to(self.config.device)

    '''initializes the branch modules as the remaining blocks of resnet-18 and the RPN.'''
    def initialize_branches(self):
        self.num_branches = 0
        if self.config.enable_radar:
            self.radar_branch = RadarBranch(backbone=ResNetTail(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained), transform=self.radar_transform).to(self.config.device)
            self.num_branches += 1
        if self.config.enable_camera:    
            self.l_cam_branch = CameraBranch(backbone=ResNetTail(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained), transform=self.image_transform).to(self.config.device)
            self.r_cam_branch = CameraBranch(backbone=ResNetTail(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained), transform=self.image_transform).to(self.config.device)
            self.num_branches += 2
        if self.config.enable_lidar:
            self.lidar_branch = LidarBranch(backbone=ResNetTail(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained), transform=self.lidar_transform).to(self.config.device)
            self.num_branches += 1
        if self.config.enable_cam_fusion:
            self.cam_fusion_branch = DualCameraFusionBranch(backbone=ResNetTail(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained), transform=self.image_transform).to(self.config.device)
            self.num_branches += 1
        if self.config.enable_cam_lidar_fusion:
            self.lidar_cam_fusion_branch = CameraLidarFusionBranch(backbone=ResNetTail(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained), transform=self.image_transform).to(self.config.device)
            self.num_branches += 1
        if self.config.enable_radar_lidar_fusion:
            self.radar_lidar_fusion_branch = RadarLidarFusionBranch(backbone=ResNetTail(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained), transform=self.radar_transform).to(self.config.device)
            self.num_branches += 1
        #RFChanges
        if self.config.enable_rf_heatmap:
            self.rf_heatmap_branch = RFHeatBranch(backbone=ResNetTail(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained), transform=self.rf_heatmap_transform).to(self.config.device)
            self.num_branches += 1
        if self.config.enable_rf_spectrogram:
            self.rf_spectrogram_branch = RFSpectoBranch(backbone=ResNetTail(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained), transform=self.rf_spectrogram_transform).to(self.config.device)
            self.num_branches += 1
        if self.config.enable_rf_fusion:
            self.rf_fusion_branch = RFFusionBranch(backbone=ResNetTail(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained), transform=self.rf_heatmap_transform).to(self.config.device)
            self.num_branches += 1


    '''
    <sensor>_x is in the input image/sensor data from each modality for a single frame. 
    radar_y, cam_y contains the target bounding boxes for training BEV and FWD respectively.
    Currently. all enabled branches are executed for every input.
    '''
    def forward(self, leftcamera_x=None, rightcamera_x=None, radar_x=None, bev_lidar_x=None, l_lidar_x=None, r_lidar_x=None, radar_y=None, cam_y=None, rf_heatmap_x=None, rf_spectrogram_x=None, rf_y=None):
        if self.training:
            #if self.config.enable_camera and cam_y is None:
                #raise ValueError("cam_y must be provided when camera is enabled during training.")
            #if self.config.enable_radar and radar_y is None:
                #raise ValueError("radar_y must be provided when radar is enabled during training.")
            if self.config.enable_rf_heatmap or self.config.enable_rf_spectrogram or self.config.enable_rf_fusion:
                if rf_y is None:
                    raise ValueError("rf_y must be provided when RF modalities are enabled during training.")


        branch_selection = []
        output_losses, output_detections, output = {}, {}, {}
        if radar_y is not None:
            radar_y = self.fix_targets(radar_y)
        if cam_y is not None:
            cam_y = self.fix_targets(cam_y)



        if self.config.enable_radar:
            radar_x, radar_y = self.radar_transform(radar_x, radar_y)
            self.check_for_degenerate_bboxes(radar_y)
            branch_selection.append(0)
            radar_output = F.dropout(self.radar_stem(radar_x.tensors), self.dropout, training=self.training)

        if self.config.enable_camera:
            rightcamera_x, _ = self.image_transform(rightcamera_x, None)
            leftcamera_x, _ = self.image_transform(leftcamera_x, None)

            l_camera_output = F.dropout(self.camera_stem(leftcamera_x.tensors), self.dropout, training=self.training)
            r_camera_output = F.dropout(self.camera_stem(rightcamera_x.tensors), self.dropout, training=self.training)

            if cam_y is not None:
                self.check_for_degenerate_bboxes(cam_y)
                branch_selection.append(1)
                branch_selection.append(2)


        if self.config.enable_lidar:
            bev_lidar_x, _ = self.lidar_transform(bev_lidar_x)
            branch_selection.append(3)
            bev_lidar_output = F.dropout(self.lidar_stem(bev_lidar_x.tensors), self.dropout, training=self.training)

        if self.config.enable_cam_lidar_fusion:
            if 1 not in branch_selection: #if cameras have not already been run
                rightcamera_x, cam_y = self.image_transform(rightcamera_x, cam_y)
                leftcamera_x, _ = self.image_transform(leftcamera_x)
                self.check_for_degenerate_bboxes(cam_y)
                l_camera_output = F.dropout(self.camera_stem(leftcamera_x.tensors), self.dropout, training=self.training)
                r_camera_output = F.dropout(self.camera_stem(rightcamera_x.tensors), self.dropout, training=self.training)
            r_lidar_x, _ = self.fwd_lidar_transform(r_lidar_x)
            branch_selection.append(5)
            r_lidar_output = F.dropout(self.fwd_lidar_stem(r_lidar_x.tensors), self.dropout, training=self.training)

        if self.config.enable_cam_fusion:
            if 1 not in branch_selection and 5 not in branch_selection: #if cameras have not already been run
                rightcamera_x, cam_y = self.image_transform(rightcamera_x, cam_y)
                leftcamera_x, _ = self.image_transform(leftcamera_x)
                self.check_for_degenerate_bboxes(cam_y)
                l_camera_output = F.dropout(self.camera_stem(leftcamera_x.tensors), self.dropout, training=self.training)
                r_camera_output = F.dropout(self.camera_stem(rightcamera_x.tensors), self.dropout, training=self.training)
            branch_selection.append(4)

        if self.config.enable_radar_lidar_fusion:
            if 0 not in branch_selection: #if radar has not already been run
                radar_x, radar_y = self.radar_transform(radar_x, radar_y)
                self.check_for_degenerate_bboxes(radar_y)
                radar_output = F.dropout(self.radar_stem(radar_x.tensors), self.dropout, training=self.training)
            if 3 not in branch_selection: #if lidar has not already been run
                bev_lidar_x, _ = self.lidar_transform(bev_lidar_x)
                bev_lidar_output = F.dropout(self.lidar_stem(bev_lidar_x.tensors), self.dropout, training=self.training)
            branch_selection.append(6)
        # RF Heatmap
        if self.config.enable_rf_heatmap and rf_heatmap_x is not None:
            rf_heatmap_x, _ = self.image_transform(rf_heatmap_x, None)
            branch_selection.append(7)
            rf_heatmap_output = F.dropout(
                self.rf_heatmap_stem(rf_heatmap_x.tensors),
                self.dropout,
                training=self.training
            )
            rf_heatmap_features = self.rf_heatmap_branch(rf_heatmap_output, rf_heatmap_x)

        # RF Spectrogram
        if self.config.enable_rf_spectrogram and rf_spectrogram_x is not None:
            rf_spectrogram_x, _ = self.image_transform(rf_spectrogram_x, None)
            branch_selection.append(8)
            rf_spectrogram_output = F.dropout(
                self.rf_spectrogram_stem(rf_spectrogram_x.tensors),
                self.dropout,
                training=self.training
            )
            rf_spectrogram_features = self.rf_spectrogram_branch(rf_spectrogram_output, rf_spectrogram_x)

        # RF Fusion
        if self.config.enable_rf_fusion and rf_heatmap_x is not None and rf_spectrogram_x is not None:
            if 7 not in branch_selection:
                rf_heatmap_x, _ = self.image_transform(rf_heatmap_x, None)
                rf_heatmap_output = F.dropout(
                    self.rf_heatmap_stem(rf_heatmap_x.tensors),
                    self.dropout,
                    training=self.training
                )
                rf_heatmap_features = self.rf_heatmap_branch(rf_heatmap_output, rf_heatmap_x)

            if 8 not in branch_selection:
                rf_spectrogram_x, _ = self.image_transform(rf_spectrogram_x, None)
                rf_spectrogram_output = F.dropout(
                    self.rf_spectrogram_stem(rf_spectrogram_x.tensors),
                    self.dropout,
                    training=self.training
                )
                rf_spectrogram_features = self.rf_spectrogram_branch(rf_spectrogram_output, rf_spectrogram_x)

            branch_selection.append(9)
            rf_fused_features = self.rf_fusion_branch(
                rf_heatmap_features, rf_spectrogram_features, rf_heatmap_x
            )
            fused_waveform = self.waveform_regressor(rf_fused_features)
            fused_waveform = fused_waveform.unsqueeze(1) # (B, 1, 7500) 
            fused_waveform = F.avg_pool1d(fused_waveform, kernel_size=3, stride=1, padding=1)
            fused_waveform = fused_waveform.squeeze(1) # (B, 7500)


        for branch_index in branch_selection:  # only collect output from the branches selected by the gate module
            if branch_index == 0:    # radar 
                output_losses['radar'], output_detections['radar'] = self.radar_branch(radar_output, radar_x, radar_y)
            elif branch_index == 1 and cam_y is not None:  # L camera 
                output_losses['camera_left'], output_detections['camera_left'] = self.l_cam_branch(l_camera_output, leftcamera_x, cam_y)
            elif branch_index == 2 and cam_y is not None:  # R camera
                output_losses['camera_right'], output_detections['camera_right'] = self.r_cam_branch(r_camera_output, rightcamera_x, cam_y)
            elif branch_index == 3:  # bev lidar
                output_losses['lidar'], output_detections['lidar'] = self.lidar_branch(bev_lidar_output, bev_lidar_x, radar_y)
            elif branch_index == 4:  # L+R camera fusion
                output_losses['camera_both'], output_detections['camera_both'] = self.cam_fusion_branch(l_camera_output, r_camera_output, rightcamera_x, cam_y)
            elif branch_index == 5:  # L+R camera + lidar fusion
                output_losses['camera_lidar'], output_detections['camera_lidar'] = self.lidar_cam_fusion_branch(l_camera_output, r_camera_output, r_lidar_output, rightcamera_x, cam_y)
            elif branch_index == 6:  # radar + lidar fusion
                output_losses['radar_lidar'], output_detections['radar_lidar'] = self.radar_lidar_fusion_branch(radar_output, bev_lidar_output, radar_x, radar_y)
            #RFChanges
            elif branch_index == 7:
                output_detections['rf_heatmap'] = rf_heatmap_features
            elif branch_index == 8:
                output_detections['rf_spectrogram'] = rf_spectrogram_features
            elif branch_index == 9:
                output_detections['rf_fusion'] = rf_fused_features
                output['fused_waveform'] = fused_waveform
        if not(self.training) and self.config.create_gate_dataset: #return stem outputs if generating gating dataset
            return output_losses, output_detections, {'radar': radar_output.cpu(), 'camera_left': l_camera_output.cpu(), 'camera_right': r_camera_output.cpu(), 'lidar': bev_lidar_output.cpu(), 'r_lidar': r_lidar_output.cpu(), 'rf_heatmap': rf_heatmap_output.cpu(), 'rf_spectrogram': rf_spectrogram_output.cpu()}, output
        elif not(self.training):
            final_loss, final_detections, output = self.fusion_block(output_losses, output_detections, output, self.config.fusion_sweep) #note: fusion does not alter or output losses at the moment
            return final_loss, final_detections, output
        else:
            return output_losses, output_detections, output


    '''convert the targets to proper format cuda tensors.'''
    def fix_targets(self, targets):
        if targets is None:
            return None
        for t in targets:
            t['labels'] = t['labels'].long().squeeze(0).to(self.config.device)
            t['boxes'] = t['boxes'].squeeze(0).to(self.config.device)
        return targets



    '''throws an error if any invalid bboxes are found.'''
    def check_for_degenerate_bboxes(self, radar_y):
        if radar_y is not None:
            for target_idx, target in enumerate(radar_y):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))
