import torch
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
from kornia.losses import dice_loss
from einops import rearrange

class DICELoss(nn.Module):
    def __init__(self):
        super(DICELoss, self).__init__()

    def forward(self, input, target):
        target = target.squeeze(1)
        loss = dice_loss(input, target)

        return loss
    

###kwan - added for class imbalance

class WeightedDiceLoss(nn.Module):
    def __init__(self, class_weights=[0.2, 0.8], epsilon=1e-6):
        super(WeightedDiceLoss, self).__init__()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.epsilon = epsilon

    def forward(self, input, target):
        # input: [B, C, H, W], target: [B, H, W]
        input = rearrange(input, 'b c h w -> b c (h w)')
        target = rearrange(target, 'b h w -> b (h w)')

        target_one_hot = F.one_hot(target, num_classes=input.size(1))  # [B, HW, C]
        target_one_hot = target_one_hot.permute(0, 2, 1).float()       # [B, C, HW]

        input = F.softmax(input, dim=1)

        # Compute per-class dice
        intersection = (input * target_one_hot).sum(dim=2)
        union = input.sum(dim=2) + target_one_hot.sum(dim=2)
        dice_score = (2 * intersection + self.epsilon) / (union + self.epsilon)

        weights = self.class_weights.to(input.device)
        dice_loss = (1 - dice_score) * weights.unsqueeze(0)  # broadcast
        return dice_loss.sum(dim=1).mean()



### version 2
"""
class DICELoss(nn.Module):
    def __init__(self, eps=1e-5):
        super(DICELoss, self).__init__()
        self.eps = eps

    def to_one_hot(self, target):
        N, C, H, W = target.size()
        assert C == 1
        target = torch.zeros(N, 2, H, W).to(target.device).scatter_(1, target, 1)
        return target

    def forward(self, input, target):
        N, C, _, _ = input.size()
        input = F.softmax(input, dim=1)

        #target = self.to_one_hot(target)
        target = torch.eye(2)[target.squeeze(1)]
        target = target.permute(0, 3, 1, 2).type_as(input)

        dims = tuple(range(1, target.ndimension()))
        inter = torch.sum(input * target, dims)
        cardinality = torch.sum(input + target, dims)
        loss = ((2. * inter) / (cardinality + self.eps)).mean()

        return 1 - loss
"""
