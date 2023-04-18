import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchsummary import summary
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet152, ResNet152_Weights
class UNetBlock(nn.Module):
    def __init__(self, in_channels, mid_channels , out_channels, upsample=False):
        super(UNetBlock, self).__init__()

        self.upsample = upsample
        self.conv1 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if upsample:
            self.up = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2) # ( W - 1) * S_w - 2 * P_w + Dil_w * (K_w-1) + Out_P_w +1
                                #  (2W, 2H)
    def forward(self, x):
        if self.upsample:
            x = self.up(x)
        x = nn.ReLU(inplace=True)(self.conv1(x))
        x = nn.ReLU(inplace=True)(self.conv2(x))
        return x


class UNetResNet152(nn.Module):
    def __init__(self, num_classes):
        super(UNetResNet152, self).__init__()

        resnet = resnet152(weights = ResNet152_Weights.IMAGENET1K_V2)
        for param in resnet.parameters():
            param.requires_grad = False

        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4
        self.center = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder5 = UNetBlock(4096, 4096, 1024, upsample=True)
        self.decoder4 = UNetBlock(2048, 2048, 512, upsample=True)
        self.decoder3 = UNetBlock(1024, 1024, 256, upsample=True)
        self.decoder2 = UNetBlock(512, 512, 128, upsample=True)
        self.decoder1 = UNetBlock(128, 128, 64, upsample=True)

        self.downsampling= nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=2)

        self.final_up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)  # 64 x 64 x 64
        e2 = self.encoder2(e1) # 256 x 64 x64
        e3 = self.encoder3(e2) # 512 x 32 x 22
        e4 = self.encoder4(e3) # 1024 x 16 x 16
        e5 = self.encoder5(e4) # 2048 x 8 x 8

        c = self.center(e5) # 2048 x 8 x 8

        d5 = self.decoder5(torch.cat([c, e5], dim = 1))
        d4 = self.decoder4(torch.cat([d5, e4], dim=1))
        d3 = self.decoder3(torch.cat([d4, e3], dim=1))
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))  # 128 x 128 x 128

        d22 = self.downsampling(d2)  # d22 = 64 x 64 x 64

        d1 = self.decoder1(torch.cat([d22, e1], dim=1))  #64 x 128 x 128

        x = self.final_up(d1)
        return self.final(x)


# model = UNetResNet152(104).to("cuda")
# print(summary(model,  (3,128, 128)))
