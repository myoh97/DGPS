from __future__ import absolute_import
from pydoc import visiblename

import torch
import torch.nn.functional as F
from torch import nn, autograd

class OIM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, 
                lut, fidelity, momentum):
        ctx.save_for_backward(inputs, targets, lut, momentum, fidelity)

        outputs_labeled = inputs.mm(lut.t())

        return torch.cat([outputs_labeled], dim=1)

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, lut, momentum, fidelity = ctx.saved_tensors

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(
                torch.cat([lut], dim=0))
            if grad_inputs.dtype == torch.float16:
                grad_inputs = grad_inputs.to(torch.float32)

        for idx, (x, y) in enumerate(zip(inputs, targets)):
            if (0 <= y) and (y < len(lut)):
                lut[y] = momentum * lut[y] + (1-momentum)* x * fidelity[idx]
                lut[y] /= lut[y].norm()

        return grad_inputs, None, None, None, None

def oim(inputs, targets, lut, fidelity, momentum=0.5):
    return OIM.apply(inputs, targets, lut, fidelity, torch.tensor(momentum))

class OIMLoss(nn.Module):
    """docstring for OIMLoss"""

    def __init__(self, num_features, num_pids,
                 oim_momentum, oim_scalar, 
                 ):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar
        self.register_buffer('lut', torch.zeros(self.num_pids, self.num_features))

    def forward(self, inputs, roi_label, cls_scores, fidelity):
        # merge into one batch, background label = 0
        targets = torch.cat(roi_label)
        label = targets - 1  # background label = -1
        
        projected = oim(inputs, label, self.lut, fidelity, momentum=self.momentum)
        
        labeled_projected = projected[label>-1]
        if len(labeled_projected)<1:
            return torch.zeros(1).cuda(inputs.device, non_blocking = True)

        labeled_label = label[label>-1]
        labeled_gate = cls_scores[label>-1]
        
        val = labeled_projected * labeled_gate.unsqueeze(-1) * self.oim_scalar
        loss_oim = F.cross_entropy(val, labeled_label.detach())
        
        return loss_oim 

