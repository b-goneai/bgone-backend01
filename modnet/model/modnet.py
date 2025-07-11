import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import MobileNetV2  # or another backbone import
from .layers import ASPP  # Efficient Atrous Spatial Pyramid Pooling

class MODNet(nn.Module):
    def __init__(self, backbone_pretrained=True, backbone='mobilenetv2'):
        super(MODNet, self).__init__()
        # Semantic segmentation branch (backbone + ASPP)
        self.stage1 = MobileNetV2(backbone_pretrained)
        self.aspp = ASPP(...)
        # Detail and matting branches
        self.stage2 = DetailBranch(...)
        self.stage3 = MatteBranch(...)
    
    def forward(self, x, return_matte_only=False):
        # Stage 1: semantic estimation
        sem = self.aspp(self.stage1(x))
        # Stage 2: detail prediction
        det = self.stage2(x, sem)
        # Stage 3: alpha matte generation
        mat = self.stage3(x, sem, det)
        if return_matte_only:
            return None, None, mat
        return sem, det, mat

