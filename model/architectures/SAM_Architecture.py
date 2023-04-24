import torch
import torch.nn as nn
import sys

from torchsummary import summary

sys.path.append("E:\PythonModels\segment-anything")
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

sys.path.append("E:\PythonModels\segment-anything\modeling")
from segment_anything.modeling import ImageEncoderViT

class SAM_Architecture(nn.Module):
    def __init__(self, num_classes):
        super(SAM_Architecture, self).__init__()
        self.imageEncoderVit = ImageEncoderViT(img_size=256, out_chans=128) # 16 x 16
        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2) # 32 x 32
        self.up2 = nn.ConvTranspose2d(128, 256, kernel_size=2, stride=2) # 64 x 64
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # 128 x 128
        self.up4 = nn.ConvTranspose2d(128, num_classes, kernel_size=2, stride=2) # 256 x 256


    def forward(self, x):
        x = self.imageEncoderVit(x)
        x = nn.ReLU(inplace=True)(self.up1(x))
        x = nn.ReLU(inplace=True)(self.up2(x))
        x = nn.ReLU(inplace=True)(self.up3(x))
        x = nn.ReLU(inplace=True)(self.up4(x))
        return x
