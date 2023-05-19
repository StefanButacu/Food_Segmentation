import torch
import torch.nn as nn
from timm.models import create_model
from timm.models.vision_transformer import VisionTransformer

class TransUNet(nn.Module):
    def __init__(self, img_size, num_classes, vit_name='vit_base_patch16_224', cnn_name='resnet50', pretrained=True):
        super(TransUNet, self).__init__()

        # CNN Backbone
        self.cnn = create_model(cnn_name, pretrained=pretrained, features_only=True)

        # Load pretrained ViT model
        self.vit = VisionTransformer(img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, num_classes=0, norm_layer=nn.LayerNorm, pretrained=pretrained)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(384, 192, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 96, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Get CNN backbone features
        cnn_features = self.cnn(x)

        # Transformer encoder
        vit_output = self.vit(x)
        vit_output = vit_output.reshape(x.shape[0], 768, 14, 14)

        # Decoder with skip connections
        x = self.decoder[0](vit_output)
        x = self.decoder[1](x)

        # Skip connection
        x = torch.cat((x, cnn_features[3]), 1)

        x = self.decoder[2](x)
        x = self.decoder[3](x)

        # Skip connection
        x = torch.cat((x, cnn_features[2]), 1)

        x = self.decoder[4](x)
        x = self.decoder[5](x)

        # Skip connection
        x = torch.cat((x, cnn_features[1]), 1)

        x = self.decoder[6](x)

        return x

# Create the TransUNet model
img_size = 224
num_classes = 21
model = TransUNet(img_size=img_size, num_classes=num_classes)
