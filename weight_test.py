import time

import torch
import yaml
from easydict import EasyDict

from PHtransBTS import PHtransBTS


def weight_test(model, x):
    start_time = time.time()
    _ = model(x)
    end_time = time.time()
    need_time = end_time - start_time
    from thop import profile

    flops, params = profile(model, inputs=(x,))
    throughout = round(x.shape[0] / (need_time / 1), 3)
    return flops, params, throughout


def Unitconversion(flops, params, throughout):
    print("params : {} M".format(round(params / (1000**2), 2)))
    print("flop : {} G".format(round(flops / (1000**3), 2)))
    print("throughout: {} Images/Min".format(throughout * 60))


if __name__ == "__main__":
    device = torch.device('cuda:0')


    x = torch.rand(1, 4, 128, 128, 128).to(device)
    with torch.no_grad():
        model = PHtransBTS(channels=(24, 80, 160, 320, 400, 400),
                           blocks=(1, 1, 1, 1, 1),
                           heads=(1, 2, 2, 4, 8),
                           r=(4, 2, 1, 1, 1),
                           deep_supervision=True,
                           branch_in=4, branch_out=3,
                           AgN=24, conv_proportion=0.8).to(device)
        model.eval()
        flops, param, throughout = weight_test(model, x)
    Unitconversion(flops, param, throughout)
