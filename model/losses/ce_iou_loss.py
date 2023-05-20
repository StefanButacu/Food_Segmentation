import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from model.losses.iou_loss import IoULoss


class CE_IOU_Loss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CE_IOU_Loss, self).__init__()
        self.alpha = alpha
        self.cross_entropy_loss = CrossEntropyLoss()
        self.iou_loss = IoULoss()

    def forward(self, predictions, targets):
        ce_loss = self.cross_entropy_loss(predictions, targets)
        iou_loss = self.iou_loss(predictions, targets)
        combined_loss = self.alpha * ce_loss + (1 - self.alpha) * iou_loss
        return combined_loss