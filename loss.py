import torch
import torch.nn as nn
from pytorch_msssim import ssim

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        return self.l1(pred, target)

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
        self.l2 = nn.MSELoss()

    def forward(self, pred, target):
        return self.l2(pred, target)

class SSIMLoss(nn.Module):
    def __init__(self, data_range=1.0, size_average=True):
        super(SSIMLoss, self).__init__()
        self.data_range = data_range
        self.size_average = size_average

    def forward(self, pred, target):
        # 注意：pytorch_msssim 需要输入 shape = (batch, channel, height, width)
        ssim_val = ssim(pred, target, data_range=self.data_range, size_average=self.size_average)
        return 1 - ssim_val  # 因为 ssim 越高越好，但 loss 越低越好

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.8, data_range=1.0):
        super(CombinedLoss, self).__init__()
        self.l1 = L1Loss()
        self.ssim = SSIMLoss(data_range=data_range)
        self.alpha = alpha

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        ssim_loss = self.ssim(pred, target)
        return self.alpha * l1_loss + (1 - self.alpha) * ssim_loss
