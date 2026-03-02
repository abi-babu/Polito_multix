import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, load_state_dict_from_url
from typing import Type, Union, List

'''Represents the Stem block in the HydraNet. Backbone is ResNet-18'''

resnet_18_pretrained = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'


class Stem(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained=False,
        stem_name='Stem'
    ) -> None:
        super(Stem, self).__init__(block, layers)

        if pretrained:
            state_dict = load_state_dict_from_url(resnet_18_pretrained, progress=True)
            self.load_state_dict(state_dict, strict=False)

        # remove unused layers
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        self.fc = None
        self.avgpool = None
        self.stem_name = stem_name

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        return x


class RFHeatStem(Stem):      # For EEG
    def __init__(self, pretrained=False):
        super().__init__(BasicBlock, [2, 2, 2, 2], pretrained, stem_name='RFHeatStem (EEG)')
        self.adaptive_pool = nn.AdaptiveAvgPool2d((64, 128))

    def forward(self, x):
        x = super().forward(x)
        x = self.adaptive_pool(x)
        return x


class RFSpectoStem(Stem):    # For ECG
    def __init__(self, pretrained=False):
        super().__init__(BasicBlock, [2, 2, 2, 2], pretrained, stem_name='RFSpectoStem (ECG)')
        self.adaptive_pool = nn.AdaptiveAvgPool2d((64, 128))

    def forward(self, x):
        x = super().forward(x)
        x = self.adaptive_pool(x)
        return x


class CameraStem(Stem):      # For Resp
    def __init__(self, pretrained=False):
        super().__init__(BasicBlock, [2, 2, 2, 2], pretrained, stem_name='CameraStem (Resp)')
        self.adaptive_pool = nn.AdaptiveAvgPool2d((64, 128))

    def forward(self, x):
        x = super().forward(x)
        x = self.adaptive_pool(x)
        return x

def create_stems(pretrained=True):
    """Factory function to create all three stems"""
    stems = nn.ModuleDict({
        'rfheat': RFHeatStem(pretrained=pretrained),    # EEG
        'rfspect': RFSpectoStem(pretrained=pretrained), # ECG
        'camera': CameraStem(pretrained=pretrained)     # Resp
    })
    return stems
