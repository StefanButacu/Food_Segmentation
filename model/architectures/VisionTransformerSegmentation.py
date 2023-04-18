# import torch
# import torch.nn as nn
# from torchvision.models import resnet50
#
# class VisionTransformerSegmentation(nn.Module):
#     def __init__(self, vit_model, num_classes):
#         super(VisionTransformerSegmentation, self).__init__()
#         self.vit_model = vit_model
#         self.vit_model.head = nn.Identity()  # Remove the classification head
#         self.segmentation_head = nn.Conv2d(vit_model.embed_dim, num_classes, kernel_size=1)
#         self.up_sample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
#
#     def forward(self, x):
#         # Extract features using the Vision Transformer
#         features = self.vit_model(x)
#
#         # Reshape the features to have a spatial layout (B, C, H, W)
#         # features = features.view(x.size(0), self.vit_model.embed_dim, 16, 16)
#         features = features.view(8, 3, 16, 16)
#         # Pass features through the segmentation head
#         segmentation_map = self.segmentation_head(features)
#
#         # Up-sample the segmentation map to match the input image size
#         segmentation_map = self.up_sample(segmentation_map)
#
#         return segmentation_map
import torch
import torch.nn as nn
import timm
from torchsummary import summary


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class ViTSegmentationModel(nn.Module):
    def __init__(self, vit_model, num_classes):
        super(ViTSegmentationModel, self).__init__()
        self.vit = timm.create_model(vit_model, pretrained=True)
        for param in self.vit.parameters():
            param.requires_grad = False

        self.vit.head = nn.Identity()  # Remove the classification head
        self.segmentation_head = SegmentationHead(self.vit.embed_dim, num_classes)
        self.interpolate = nn.Upsample(scale_factor=14, mode='bilinear', align_corners=True)

    def forward(self, x):
        features = self.vit(x)
        features = features.view(x.shape[0], 3, 16, 16)  # batch_size x channels x W x H
        seg_output = self.segmentation_head(features)
        seg_output = self.interpolate(seg_output)
        return seg_output

# Instantiate the model
num_classes = 104  # Number of classes for the segmentation task
vit_model = "vit_base_patch16_224"  # Pre-trained ViT model name
model = ViTSegmentationModel(vit_model, num_classes).to("cuda")
summary(model,input_size= ( 3, 224, 224))
