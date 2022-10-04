from torch import nn
import torch
from torchvision.models.mobilenetv3 import MobileNetV3, InvertedResidualConfig
from functools import partial

bneck_conf = partial(InvertedResidualConfig, width_mult=1.0)
inverted_residual_setting_large = [
    bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),  # c1 176 16
    bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),
    bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),  # c2 88 24
    bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),
    bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
    bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),  # c3 44 40
    bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),
    bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
    bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
    bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),  # c4 22 80  exp
    bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
    bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),  # c5 22 112
    bneck_conf(112, 5, 672, 160, True, "HS", 2, 1),
    bneck_conf(160, 5, 960, 160, True, "HS", 1, 1),
    bneck_conf(160, 5, 960, 160, True, "HS", 1, 1),  # c6 11 160
]


class MobileNetV3Encoder_large(MobileNetV3):
    def __init__(self):
        super().__init__(inverted_residual_setting=inverted_residual_setting_large, last_channel=0)

        self.features = self.features[:-1]
        del self.avgpool
        del self.classifier

        self.layer1 = nn.Sequential(self.features[0], self.features[1])  # 16
        self.layer2 = nn.Sequential(self.features[2], self.features[3])  # 24
        self.layer3 = nn.Sequential(self.features[4], self.features[5], self.features[6])  # 40
        self.layer4 = nn.Sequential(self.features[7], self.features[8], self.features[9], self.features[10])  # 80
        self.layer5 = nn.Sequential(self.features[11], self.features[12])  # 112
        self.layer6 = nn.Sequential(self.features[13], self.features[14], self.features[15])  # 160


class MobileNetV3Encoder_small(MobileNetV3):
    def __init__(self):
        super().__init__(inverted_residual_setting=inverted_residual_setting_large, last_channel=0)

        self.features = self.features[:-6]
        del self.avgpool
        del self.classifier

        self.layer1 = nn.Sequential(self.features[0], self.features[1])  # 16
        self.layer2 = nn.Sequential(self.features[2], self.features[3])  # 24
        self.layer3 = nn.Sequential(self.features[4], self.features[5], self.features[6])  # 40
        self.layer4 = nn.Sequential(self.features[7], self.features[8], self.features[9], self.features[10])  # 80


def initialize_weights(model):
    url = 'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth'
    pretrained_dict = torch.hub.load_state_dict_from_url(url)
    all_params = {}
    for k, v in model.state_dict().items():
        if k in pretrained_dict.keys():
            v = pretrained_dict[k]
            all_params[k] = v
    model.load_state_dict(all_params, strict=False)
