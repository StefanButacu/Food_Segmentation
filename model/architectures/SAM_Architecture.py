import sys

import torch.nn as nn

sys.path.append("E:\PythonModels\segment-anything")
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

  #
  # super(SAM_Architecture, self).__init__()
  #       self.imageEncoderVit = ImageEncoderViT(img_size=256, out_chans=128) # 16 x 16
  #       self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2) # 32 x 32
  #       self.conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
  #
  #       self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2) # 64 x 64
  #       self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
  #
  #       self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2) # 128 x 128
  #       self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
  #
  #       self.up4 = nn.ConvTranspose2d(128, 256, kernel_size=2, stride=2) # 256 x 256
  #       self.conv4 = nn.Conv2d(256, num_classes, kernel_size=3, padding=1)
  #       self.dropout = nn.Dropout2d(p=0.15)  # Add dropout layer