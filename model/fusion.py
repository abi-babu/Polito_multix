import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class EarlyFusion(nn.Module):
    """
    Early fusion: Concatenate stem features before branches
    """
    def __init__(self):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, stem_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        stem_outputs: dict with keys 'rfheat' (EEG), 'rfspect' (ECG), 'camera' (Resp)
        """
        modalities = ['rfheat', 'rfspect', 'camera']
        features = []

        for mod in modalities:
            if mod in stem_outputs:
                features.append(stem_outputs[mod])

        # Concatenate along channel dimension
        fused = torch.cat(features, dim=1)  # [B, 192, 64, 128]

        # Reduce channels
        fused = self.fusion_conv(fused)  # [B, 64, 64, 128]

        return fused


class LateFusion(nn.Module):
    """
    Late fusion: Combine branch features after individual processing
    """
    def __init__(self, fusion_method='concat', num_modalities=3):
        super().__init__()
        self.fusion_method = fusion_method

        if fusion_method == 'concat':
            in_channels = 512 * num_modalities
            self.fusion_layer = nn.Sequential(
                nn.Conv2d(in_channels, 1024, kernel_size=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
        elif fusion_method == 'attention':
            self.attention = nn.Sequential(
                nn.Conv2d(512, 64, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1),
                nn.Sigmoid()
            )
        elif fusion_method == 'weighted':
            # Learnable weights for each modality
            self.weights = nn.Parameter(torch.ones(num_modalities) / num_modalities)

    def forward(self, branch_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        branch_features: dict with keys from branches
        """
        if self.fusion_method == 'concat':
            # Concatenate all features
            features = list(branch_features.values())
            fused = torch.cat(features, dim=1)  # [B, 512*N, 8, 16]
            fused = self.fusion_layer(fused)    # [B, 512, 8, 16]

        elif self.fusion_method == 'attention':
            # Apply attention to each modality and sum
            fused = 0
            for name, feat in branch_features.items():
                attn = self.attention(feat)
                fused += feat * attn

        elif self.fusion_method == 'weighted':
            # Weighted sum
            fused = 0
            weights = F.softmax(self.weights, dim=0)
            for i, (name, feat) in enumerate(branch_features.items()):
                fused += feat * weights[i]

        return fused


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion that learns which modalities to trust
    """
    def __init__(self, feature_dim=512):
        super().__init__()

        # Modality confidence predictors
        self.confidence_ecg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

        self.confidence_eeg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

        self.confidence_resp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

        # Fusion layer
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim * 3, feature_dim * 2, kernel_size=1),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, branch_features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        # Get individual features
        ecg_feat = branch_features.get('rfspect', None)  # ECG
        eeg_feat = branch_features.get('rfheat', None)   # EEG
        resp_feat = branch_features.get('resp', None)     # Resp

        # Compute confidence scores
        confidence = {}
        weighted_features = []

        if ecg_feat is not None:
            conf_ecg = self.confidence_ecg(ecg_feat)  # [B, 1]
            confidence['ecg'] = conf_ecg
            weighted_features.append(ecg_feat * conf_ecg.view(-1, 1, 1, 1))

        if eeg_feat is not None:
            conf_eeg = self.confidence_eeg(eeg_feat)
            confidence['eeg'] = conf_eeg
            weighted_features.append(eeg_feat * conf_eeg.view(-1, 1, 1, 1))

        if resp_feat is not None:
            conf_resp = self.confidence_resp(resp_feat)
            confidence['resp'] = conf_resp
            weighted_features.append(resp_feat * conf_resp.view(-1, 1, 1, 1))

        # Concatenate and fuse
        if len(weighted_features) > 1:
            fused = torch.cat(weighted_features, dim=1)
            fused = self.fusion_conv(fused)
        else:
            fused = weighted_features[0]

        return fused, confidence


class HierarchicalFusion(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()

        # First level fusion (EEG + ECG)
        self.fusion_1 = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

        # Second level fusion (EEG+ECG + Resp)
        self.fusion_2 = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

        # Final refinement
        self.refine = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim)
        )

    def forward(self, branch_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Check if all required features are present
        required_keys = ['rfheat', 'rfspect', 'resp']
        for key in required_keys:
            if key not in branch_features:
                raise ValueError(f"Missing required feature: {key}")

        # First level: fuse EEG and ECG (they should have same dimensions)
        eeg_ecg = torch.cat([branch_features['rfheat'], branch_features['rfspect']], dim=1)
        fused_1 = self.fusion_1(eeg_ecg)  # [B, 512, H, W]

        # Get resp features
        resp_feat = branch_features['resp']

        if fused_1.shape[2:] != resp_feat.shape[2:]:
            print(f"Shape mismatch: fused_1 {fused_1.shape}, resp_feat {resp_feat.shape}")
            # Upsample or downsample resp_feat to match fused_1
            resp_feat = F.interpolate(
                resp_feat,
                size=fused_1.shape[2:],  # Target size = fused_1 spatial dims
                mode='bilinear',
                align_corners=False
            )
            print(f"After interpolation: resp_feat {resp_feat.shape}")

        # Second level: fuse with Resp
        combined = torch.cat([fused_1, resp_feat], dim=1)  # [B, 1024, H, W]
        fused_2 = self.fusion_2(combined)  # [B, 512, H, W]

        # Refine with residual connection
        output = self.refine(fused_2) + fused_2

        return output

class WaveformPredictionHead(nn.Module):
    def __init__(self, feature_dim=512, output_length=1000):
        super().__init__()
        
        # 1D convolution along temporal dimension (height=8)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=3, padding=1),  # [B, 32, 16] after reshape
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)  # Pool to fixed length
        )
        
        # Process spatial features
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global pooling
        )
        
        # Combine and generate waveform
        self.fc = nn.Sequential(
            nn.Linear(64*32 + 256, 1024),  # Temporal + spatial
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, output_length)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x):
        # x: [B, 512, 8, 16]
        B = x.shape[0]
        
        # Temporal path: treat height as time
        x_temp = x.permute(0, 2, 1, 3)  # [B, 8, 512, 16]
        x_temp = x_temp.reshape(B, 8, -1)  # [B, 8, 512*16=8192]
        x_temp = self.temporal_conv(x_temp)  # [B, 64, 32]
        x_temp = x_temp.flatten(1)  # [B, 64*32=2048]
        
        # Spatial path
        x_spat = self.spatial_conv(x)  # [B, 256, 1, 1]
        x_spat = x_spat.flatten(1)  # [B, 256]
        
        # Combine
        x_combined = torch.cat([x_temp, x_spat], dim=1)
        
        # Generate waveform
        output = self.fc(x_combined)
        
        return output


def create_fusion(fusion_type='hierarchical', **kwargs):

    if fusion_type == 'early':
        return EarlyFusion()
    elif fusion_type == 'late':
        fusion_method = kwargs.get('fusion_method', 'concat')
        num_modalities = kwargs.get('num_modalities', 3)
        return LateFusion(fusion_method=fusion_method, num_modalities=num_modalities)
    elif fusion_type == 'adaptive':
        feature_dim = kwargs.get('feature_dim', 512)
        return AdaptiveFusion(feature_dim=feature_dim)
    elif fusion_type == 'hierarchical':
        feature_dim = kwargs.get('feature_dim', 512)
        return HierarchicalFusion(feature_dim=feature_dim)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
