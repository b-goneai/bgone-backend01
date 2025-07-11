import torch
import torch.nn as nn
from torchvision import models

class MODNet(nn.Module):
    def __init__(self, backbone_pretrained=True):
        super(MODNet, self).__init__()

        # Encoder: MobileNetV2
        mnet = models.mobilenet_v2(pretrained=backbone_pretrained)
        self.backbone = mnet.features

        self.downsample = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(1280, 512, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)

        self.deconv3 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.downsample(x)
        x = self.backbone(x)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.final(x)
        x = torch.sigmoid(x)

        return x
