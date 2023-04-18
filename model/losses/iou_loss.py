import torch
import torch.nn as nn

class IoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(IoULoss, self).__init__()
        self.eps = eps

    def forward(self, predictions, targets):
        # Convert predictions to probabilities using softmax
        probabilities = torch.softmax(predictions, dim=1)

        # One-hot encode the targets
        targets_one_hot = torch.zeros_like(predictions).scatter_(1, targets.unsqueeze(1), 1)

        # Compute intersection and union
        intersection = torch.sum(probabilities * targets_one_hot, dim=(2, 3))
        union = torch.sum(probabilities, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3)) - intersection

        # Calculate IoU and then IoU loss
        iou = (intersection + self.eps) / (union + self.eps)
        iou_loss = 1 - torch.mean(iou)

        return iou_loss
