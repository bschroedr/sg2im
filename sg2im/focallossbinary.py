
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# https://github.com/mbsariyildiz/focal-loss.pytorch
# https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
#
# Usage:
# output0 = FocalLoss(gamma=2)(x, l)
# if gamma=0, same as standard CE loss

class FocalLossBinary(nn.Module):

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLossBinary, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        ##target = target.view(-1, 1)

        input = input.view(-1)
        target = target.view(-1)
        sig_pt = F.sigmoid(input)
        #logpt = -F.binary_cross_entropy(sig_pt, target) 
        logpt = target*torch.log(sig_pt) + (1-target)*torch.log(1-sig_pt)
 
        # multi-class cross entropy case
        ##logpt = F.log_softmax(input, dim=1)
        ##logpt = logpt.gather(1,target)
        ##logpt = logpt.view(-1)
        pt = Variable(logpt.exp())

        if self.alpha is not None:
            #if self.alpha.type() != input.data.type():
            #    self.alpha = self.alpha.type_as(input.data)
            #at = self.alpha.gather(0, target.data.view(-1))
            at = torch.full((input.shape),self.alpha[1])
            idx = torch.nonzero(target)
            at[idx] = self.alpha[0]
             
            logpt = logpt * Variable(at.to('cuda'))

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
