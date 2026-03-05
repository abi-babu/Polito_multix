import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, load_state_dict_from_url
from typing import List, Union, Type

resnet_18_pretrained = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'


'''Represents the last three conv blocks in resnet. Used as the backbone for branches'''
class ResNetTail(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained=False
    ) -> None:
        super(ResNetTail, self).__init__(block, layers)

        if pretrained:
            print(f"Loading pretrained weights for ResNetTail...")
            # Load full pretrained model
            full_state_dict = load_state_dict_from_url(resnet_18_pretrained, progress=True)

            # Create a new state dict with only the layers we need (layer2, layer3, layer4)
            filtered_state_dict = {}
            for k, v in full_state_dict.items():
                if k.startswith('layer2.') or k.startswith('layer3.') or k.startswith('layer4.'):
                    filtered_state_dict[k] = v

            # Load the filtered state dict
            self.load_state_dict(filtered_state_dict, strict=False)
            print(f"  Loaded {len(filtered_state_dict)} pretrained layers for layer2, layer3, layer4")

        # remove unused layers from ResNet model
        self.conv1 = None
        self.bn1 = None
        self.maxpool = None
        self.avgpool = None
        self.layer1 = None
        self.relu = None
        self.fc = None
        self.out_channels = 512

    def forward(self, x):
        # Input: [B, 64, 64, 128] (after stem)
        x = self.layer2(x)  # [B, 128, 32, 64]
        x = self.layer3(x)  # [B, 256, 16, 32]
        x = self.layer4(x)  # [B, 512, 8, 16]
        return x


'''Base branch for individual modalities'''
class BaseBranch(nn.Module):
    def __init__(self, backbone, branch_name='Base'):
        super().__init__()
        self.backbone = backbone
        self.branch_name = branch_name

    def forward(self, x):
        return self.backbone(x)


'''RFHeatBranch - For EEG heatmaps'''
class RFHeatBranch(BaseBranch):
    def __init__(self, pretrained=False):
        backbone = ResNetTail(BasicBlock, [2, 2, 2, 2], pretrained=pretrained)
        super().__init__(backbone, branch_name='RFHeatBranch (EEG)')


'''RFSpectoBranch - For ECG spectrograms'''
class RFSpectoBranch(BaseBranch):
    def __init__(self, pretrained=False):
        backbone = ResNetTail(BasicBlock, [2, 2, 2, 2], pretrained=pretrained)
        super().__init__(backbone, branch_name='RFSpectoBranch (ECG)')


'''RespBranch - For Respiration RGB images'''
class RespBranch(BaseBranch):
    def __init__(self, pretrained=False):
        backbone = ResNetTail(BasicBlock, [2, 2, 2, 2], pretrained=pretrained)
        super().__init__(backbone, branch_name='RespBranch')


class BaseFusionBranch(nn.Module):
    def __init__(self, backbone, fusion_name='BaseFusion'):
        super().__init__()
        # Channel reduction - from 1024 to 64 (what backbone expects)
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.backbone = backbone
        self.fusion_name = fusion_name

    def forward(self, feat1, feat2):
        # Input features are from branches: [B, 512, 8, 16]

        # Upsample to match stem output size (64,128) for better fusion
        feat1_up = F.interpolate(feat1, size=(64, 128), mode='bilinear', align_corners=False)
        feat2_up = F.interpolate(feat2, size=(64, 128), mode='bilinear', align_corners=False)

        # Concatenate
        x = torch.cat([feat1_up, feat2_up], dim=1)  # [B, 1024, 64, 128]

        # Reduce channels to 64 (what backbone expects)
        x = self.channel_reduce(x)  # [B, 64, 64, 128]

        # Pass through backbone (expects [B, 64, 64, 128])
        x = self.backbone(x)  # [B, 512, 8, 16]

        return x


'''Heatmap + Spectrogram Fusion (EEG + ECG)'''
class HeatmapSpectFusion(BaseFusionBranch):
    def __init__(self, pretrained=False):
        backbone = ResNetTail(BasicBlock, [2, 2, 2, 2], pretrained=pretrained)
        super().__init__(backbone, fusion_name='HeatmapSpectFusion (EEG+ECG)')


'''Spectrogram + Resp Fusion (ECG + Resp)'''
class SpectRespFusion(BaseFusionBranch):
    def __init__(self, pretrained=False):
        backbone = ResNetTail(BasicBlock, [2, 2, 2, 2], pretrained=pretrained)
        super().__init__(backbone, fusion_name='SpectRespFusion (ECG+Resp)')


'''Resp + Heatmap Fusion (Resp + EEG)'''
class RespHeatFusion(BaseFusionBranch):
    def __init__(self, pretrained=False):
        backbone = ResNetTail(BasicBlock, [2, 2, 2, 2], pretrained=pretrained)
        super().__init__(backbone, fusion_name='RespHeatFusion (Resp+EEG)')

def create_branches(pretrained=True):
    """Factory function to create all branches"""
    branches = nn.ModuleDict({
        # Individual branches
        'rfheat': RFHeatBranch(pretrained=pretrained),      # EEG
        'rfspect': RFSpectoBranch(pretrained=pretrained),   # ECG
        'resp': RespBranch(pretrained=pretrained),          # Resp

        # Fusion branches
        'heatmap_spect': HeatmapSpectFusion(pretrained=pretrained),  # EEG+ECG
        'spect_resp': SpectRespFusion(pretrained=pretrained),        # ECG+Resp
        'resp_heat': RespHeatFusion(pretrained=pretrained),         # Resp+EEG
    })
    return branches
