import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def EntropyLoss(input_):
    mask = input_.ge(0.0000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = - (torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))




class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.sum(dim=0)

class Entropy(torch.nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, x):
        b = x*torch.log(x)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)


class MyBCE(torch.nn.Module):
    def __init__(self):
        super(MyBCE, self).__init__()

    def forward(self, p):
        return (-0.5*torch.log(p) - 0.5*torch.log(1-p)).mean()




class CORAL(nn.Module):
    def __init__(self):
        super(CORAL, self).__init__()

    def forward(self, source, target):
        d = source.size(1)

        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt

        # frobenius norm between source and target
        loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
        loss = loss / (4 * d * d)
        return loss