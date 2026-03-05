import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from model.stem import create_stems
from model.branch import create_branches
from model.fusion import (
    WaveformPredictionHead,
    create_fusion
)
from torchvision.models.resnet import BasicBlock


class HydraNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.device = config.device
        self.waveform_length = getattr(config, 'waveform_length', 512)
        self.freeze_stems = getattr(config, 'freeze_stems', False)
        self.fusion_type = getattr(config, 'fusion_type', 'hierarchical')
        self.stems = create_stems(pretrained=getattr(config, 'pretrained_stems', True))
        self.stems = nn.ModuleDict(self.stems)

        if self.freeze_stems:
            for stem in self.stems.values():
                for param in stem.parameters():
                    param.requires_grad = False
        self.branches = create_branches(pretrained=getattr(config, 'pretrained_branches', True))
        self.branches = nn.ModuleDict(self.branches)

        fusion_kwargs = {
            'fusion_type': self.fusion_type,
            'feature_dim': 512,
            'num_modalities': 3,
            'fusion_method': getattr(config, 'fusion_method', 'concat')
        }
        self.main_fusion = create_fusion(**fusion_kwargs)

      
        if self.fusion_type == 'early':
            from model.branch import ResNetTail
            self.post_fusion_branch = ResNetTail(
                BasicBlock, [2, 2, 2, 2],
                pretrained=getattr(config, 'pretrained_fusion', True)
            )


        self.prediction_head = WaveformPredictionHead(
            feature_dim=512,
            output_length=self.waveform_length
        )

        class AmplitudeAwareLoss(nn.Module):
            def __init__(self, alpha=0.2, beta=0.6):
                super().__init__()
                self.mse = nn.MSELoss()
                self.alpha = alpha
                self.beta = beta
                self.eps = 1e-8

            def forward(self, pred, target):
                # MSE loss - basic reconstruction
                mse_loss = self.mse(pred, target)

                # Amplitude matching loss (peak-to-peak range)
                pred_range = pred.max(dim=1)[0] - pred.min(dim=1)[0]
                target_range = target.max(dim=1)[0] - target.min(dim=1)[0]
                range_loss = F.mse_loss(pred_range, target_range)

                # Standard deviation matching loss
                pred_std = pred.std(dim=1)
                target_std = target.std(dim=1)
                std_loss = F.mse_loss(pred_std, target_std)

                # CORRELATION LOSS 
                # Center the signals
                pred_centered = pred - pred.mean(dim=1, keepdim=True)
                target_centered = target - target.mean(dim=1, keepdim=True)
                
                # Normalize
                pred_norm = pred_centered / (pred_centered.norm(dim=1, keepdim=True) + self.eps)
                target_norm = target_centered / (target_centered.norm(dim=1, keepdim=True) + self.eps)
                
                # Correlation (cosine similarity)
                correlation = (pred_norm * target_norm).sum(dim=1)
                correlation_loss = 1 - correlation.mean()  # Want correlation close to 1

                # Combined loss - emphasize correlation for shape matching
                total_loss = (self.alpha * mse_loss + 
                            0.2 * range_loss + 
                            0.2 * std_loss +
                            self.beta * correlation_loss)

                return total_loss

        self.criterion = AmplitudeAwareLoss(alpha=0.2, beta=0.6)  # Correlation is most important
        self._init_weights()

    def _init_weights(self):
        """Initialize fusion weights based on correlation results"""
        if hasattr(self.main_fusion, 'weights'):
            with torch.no_grad():
                # Based on your correlation analysis: 
                # ECG+EEG (0.18), ECG+Resp (0.09), Resp+EEG (-0.16)
                self.main_fusion.weights.data = torch.tensor([0.6, 0.3, 0.1])

    def forward(self,
                ecg_img: Optional[torch.Tensor] = None,
                eeg_img: Optional[torch.Tensor] = None,
                resp_img: Optional[torch.Tensor] = None,
                gt_waveform: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        results = {}
        stem_outputs = {}
        branch_features = {}

        if ecg_img is not None:
            stem_outputs['rfspect'] = self.stems['rfspect'](ecg_img)

        if eeg_img is not None:
            stem_outputs['rfheat'] = self.stems['rfheat'](eeg_img)

        if resp_img is not None:
            stem_outputs['camera'] = self.stems['camera'](resp_img)

        if 'rfspect' in stem_outputs:
            branch_features['rfspect'] = self.branches['rfspect'](stem_outputs['rfspect'])

        if 'rfheat' in stem_outputs:
            branch_features['rfheat'] = self.branches['rfheat'](stem_outputs['rfheat'])

        if 'camera' in stem_outputs:
            branch_features['resp'] = self.branches['resp'](stem_outputs['camera'])

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

        results['waveform'] = self.prediction_head(fused)

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
