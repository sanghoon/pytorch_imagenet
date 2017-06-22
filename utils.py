import torch.nn as nn
import torch
import numpy as np
import math

def initvars(modules):
    # Copied from vision/torchvision/models/resnet.py
    for m in modules:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class MultiCropEnsemble(nn.Module):
    def __init__(self, module, cropsize, act=nn.functional.softmax, flipping=True):
        super(MultiCropEnsemble, self).__init__()
        self.cropsize = cropsize
        self.flipping = flipping
        self.internal_module = module
        self.act = act

    # Naive code
    def forward(self, x):
        # H, W >= cropsize
        assert(x.size()[2] >= self.cropsize)
        assert(x.size()[3] >= self.cropsize)

        cs = self.cropsize
        x1 = 0
        x2 = x.size()[2] - self.cropsize
        cx = x.size()[2] // 2 - self.cropsize // 2
        y1 = 0
        y2 = x.size()[3] - self.cropsize
        cy = x.size()[3] // 2 - self.cropsize // 2

        get_output = lambda x: self.act(self.internal_module.forward(x))

        _y = get_output(x[:, :, x1:x1+cs, y1:y1+cs])
        _y = get_output(x[:, :, x1:x1+cs, y2:y2+cs]) + _y
        _y = get_output(x[:, :, x2:x2+cs, y1:y1+cs]) + _y
        _y = get_output(x[:, :, x2:x2+cs, y2:y2+cs]) + _y
        _y = get_output(x[:, :, cx:cx+cs, cy:cy+cs]) + _y

        if self.flipping == True:
            # Naive flipping

            arr = (x.data).cpu().numpy()                        # Bring back to cpu
            arr = arr[:,:,:, ::-1]                              # Flip
            x.data = type(x.data)(np.ascontiguousarray(arr))    # Store

            _y = get_output(x[:, :, x1:x1 + cs, y1:y1 + cs]) + _y
            _y = get_output(x[:, :, x1:x1 + cs, y2:y2 + cs]) + _y
            _y = get_output(x[:, :, x2:x2 + cs, y1:y1 + cs]) + _y
            _y = get_output(x[:, :, x2:x2 + cs, y2:y2 + cs]) + _y
            _y = get_output(x[:, :, cx:cx + cs, cy:cy + cs]) + _y

            _y = _y / 10.0
        else:
            _y = _y / 5.0

        return _y

