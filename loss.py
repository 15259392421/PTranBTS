from monai.losses.dice import DiceLoss
from monai.losses import FocalLoss
import torch
import torch.nn as nn

class MyCriterion(nn.Module):
    def __init__(self):
        super(MyCriterion, self).__init__()
        self.Dice_criterion = DiceLoss(smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True)
        self.Focal_criterion = FocalLoss(to_onehot_y=False)

    def forward(self, x, y):
        loss = self.Dice_criterion(x, y) + self.Focal_criterion(x, y)
        return loss
