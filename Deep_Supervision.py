import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from loss import MyCriterion
def downsample_seg_for_ds_transform2(x ,seg, ds_scales=((1, 1, 1),(0.25, 0.25, 0.25), (0.125, 0.125, 0.125))):
    ds_scales = []
    for layerout in x:
        layerout_shape = layerout.shape[3]
        y_shape = seg.shape[3]
        mul = layerout_shape/y_shape
        ds_scales.append((mul,mul,mul))


    output = []
    for s in ds_scales:
        if all([i == 1 for i in s]):
            output.append(seg)
        else:
            new_shape = list(seg.shape[2:])
            for i in range(len(new_shape)):
                new_shape[i] = int(new_shape[i]*s[i])
            out_seg = F.interpolate(seg, size=tuple(new_shape), mode='trilinear', align_corners=True)
            output.append(out_seg)
    return output



class MultipleOutputLoss2(nn.Module):
    def __init__(self, deep):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        weights = np.array([1 / (2 ** i) for i in range(deep)])
        weights = weights / weights.sum()
        self.weight_factors = weights
        self.loss = MyCriterion()

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        return l

class DeepCriterion(nn.Module):
    def __init__(self,Deepth = 5):
        super(DeepCriterion, self).__init__()
        self.loss = MultipleOutputLoss2(Deepth)

    def forward(self, x, y):
        y = downsample_seg_for_ds_transform2(x, y)
        loss = self.loss(x, y)
        return loss
if __name__ == "__main__":
    seg = torch.randn(size=(1, 3, 128, 128, 128))
    pr = (torch.randn(size=(1, 3, 128, 128, 128)),torch.randn(size=(1, 3, 64, 64, 64)),
         torch.randn(size=(1, 3, 32,32,32)),torch.randn(size=(1, 3, 16,16,16)),torch.randn(size=(1, 3, 8,8,8)))

    cr = DeepCriterion()
    loss = cr(pr,seg)