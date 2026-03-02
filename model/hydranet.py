import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

# Import your actual modules
from model.stem import create_stems
from model.branch import create_branches
from model.fusion import (
    WaveformPredictionHead,
    create_fusion
)
from torchvision.models.resnet import BasicBlock  # Add BasicBlock here
# ============================================================
# HydraNet for Respiratory Waveform Prediction
# ============================================================
class HydraNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.device = config.device
        self.waveform_length = getattr(config, 'waveform_length', 512)
        self.freeze_stems = getattr(config, 'freeze_stems', False)
        self.fusion_type = getattr(config, 'fusion_type', 'hierarchical')

        # ====================================================
        # STEMS (Feature Extraction)
        # ====================================================
        self.stems = create_stems(pretrained=getattr(config, 'pretrained_stems', True))
        self.stems = nn.ModuleDict(self.stems)

        if self.freeze_stems:
            for stem in self.stems.values():
                for param in stem.parameters():
                    param.requires_grad = False

        # ====================================================
        # BRANCHES (Individual Modality Processing)
        # ====================================================
        self.branches = create_branches(pretrained=getattr(config, 'pretrained_branches', True))
        self.branches = nn.ModuleDict(self.branches)

        # ====================================================
        # MAIN FUSION MODULE
        # ====================================================
        fusion_kwargs = {
            'fusion_type': self.fusion_type,
            'feature_dim': 512,
            'num_modalities': 3,
            'fusion_method': getattr(config, 'fusion_method', 'concat')
        }
        self.main_fusion = create_fusion(**fusion_kwargs)

        # For early fusion, we need a post-fusion branch
        if self.fusion_type == 'early':
            from branch import ResNetTail
            from torchvision.models.resnet import BasicBlock
            self.post_fusion_branch = ResNetTail(
                BasicBlock, [2, 2, 2, 2],
                pretrained=getattr(config, 'pretrained_fusion', True)
            )

        # ====================================================
        # WAVEFORM PREDICTION HEAD
        # ====================================================
        self.prediction_head = WaveformPredictionHead(
            feature_dim=512,
            output_length=self.waveform_length
        )

        # ====================================================
        # LOSS FUNCTION
        # ====================================================
        self.criterion = nn.MSELoss()

        self._init_weights()

    def _init_weights(self):
        """Initialize fusion weights based on correlation results"""
        if hasattr(self.main_fusion, 'weights'):
            with torch.no_grad():
                self.main_fusion.weights.data = torch.tensor([0.6, 0.3, 0.1])

    def forward(self,
                ecg_img: Optional[torch.Tensor] = None,
                eeg_img: Optional[torch.Tensor] = None,
                resp_img: Optional[torch.Tensor] = None,
                gt_waveform: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through HydraNet

        Returns:
            Dictionary with predictions and intermediate outputs
        """
        results = {}
        stem_outputs = {}
        branch_features = {}

        # ====================================================
        # STEM FORWARD PASSES
        # ====================================================
        if ecg_img is not None:
            stem_outputs['rfspect'] = self.stems['rfspect'](ecg_img)

        if eeg_img is not None:
            stem_outputs['rfheat'] = self.stems['rfheat'](eeg_img)

        if resp_img is not None:
            stem_outputs['camera'] = self.stems['camera'](resp_img)

        # ====================================================
        # BRANCH FORWARD PASSES
        # ====================================================
        if 'rfspect' in stem_outputs:
            branch_features['rfspect'] = self.branches['rfspect'](stem_outputs['rfspect'])

        if 'rfheat' in stem_outputs:
            branch_features['rfheat'] = self.branches['rfheat'](stem_outputs['rfheat'])

        if 'camera' in stem_outputs:
            branch_features['resp'] = self.branches['resp'](stem_outputs['camera'])

        # ====================================================
        # MAIN FUSION
        # ====================================================
        if self.fusion_type == 'early':
            fused = self.main_fusion(stem_outputs)
            fused = self.post_fusion_branch(fused)
        else:
            fused = self.main_fusion(branch_features)
            if self.fusion_type == 'adaptive' and isinstance(fused, tuple):
                fused, confidence = fused
                results['fusion_confidence'] = {
                    k: v.mean().item() for k, v in confidence.items()
                }

        results['fused_features'] = fused

        # ====================================================
        # WAVEFORM PREDICTION
        # ====================================================
        results['waveform'] = self.prediction_head(fused)

        # ====================================================
        # LOSS COMPUTATION
        # ====================================================
        if gt_waveform is not None:
            results['loss'] = self.criterion(results['waveform'], gt_waveform)

        return results

    def predict(self, ecg_img=None, eeg_img=None, resp_img=None):
        """Inference function"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                ecg_img=ecg_img,
                eeg_img=eeg_img,
                resp_img=resp_img
            )
        return outputs['waveform']


# ============================================================
# Configuration Class
# ============================================================
class HydraNetConfig:
    """Configuration for HydraNet"""
    def __init__(self, **kwargs):
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.waveform_length = kwargs.get('waveform_length', 512)
        self.freeze_stems = kwargs.get('freeze_stems', False)
        self.pretrained_stems = kwargs.get('pretrained_stems', True)
        self.pretrained_branches = kwargs.get('pretrained_branches', True)
        self.pretrained_fusion = kwargs.get('pretrained_fusion', True)
        self.fusion_type = kwargs.get('fusion_type', 'hierarchical')
        self.fusion_method = kwargs.get('fusion_method', 'concat')
