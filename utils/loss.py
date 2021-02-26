from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np

def cal_cross_entropy(pred, target, weight, crossEntropy):
    loss = 0
    for i, predi in enumerate(pred):
        for j, predj in enumerate(predi):
            loss += crossEntropy(predj.view(1, -1), target[i][j]) * weight[i][j]
    return loss * 0.01

class HeirarchicalCrossEntropyLoss(object):
    def __init__(self, weights, device):
        super(HeirarchicalCrossEntropyLoss, self).__init__()
        self.weights = [torch.tensor(w, dtype=torch.float, device=device) for w in weights]
        """Loss functions for semantic segmentation"""
        self.CrossEntropyLosses = [nn.CrossEntropyLoss(weight=w) for w in self.weights]
    def __call__(self, pred, target, level=0):
        pred = pred.view(-1, len(self.weights[level]))
        target=target.view(-1)
        return self.CrossEntropyLosses[level](pred, target)

class ConsistencyLoss:
    def __init__(self, CM, CLW, device="cpu"):
        super(ConsistencyLoss, self).__init__()
        self.gather_id = [np.argmax(m, axis=0)for m in CM]
        self.weights = torch.Tensor(CLW).to(device)
    def __call__(self, preds):
        probs = [nn.functional.softmax(pred.permute(0, 2, 1), dim=2) for pred in preds]
        loss = 0
        for i, gid in enumerate(self.gather_id):
            probs_ = probs[i + 1] - probs[i][..., gid]
            loss += nn.functional.relu(probs_).sum() * self.weights[i]
        return loss * 0.01
            
def cal_consistency_loss(CM, preds, CLW):
    probs = [nn.functional.softmax(pred.permute(0, 2, 1), dim=2) for pred in preds]
    for i, matrix in enumerate(CM):
        m = torch.Tensor(matrix).cuda()
        prob_ = torch.matmul(m, probs[i + 1].permute(0, 2, 1)).permute(0, 2, 1)
        CLoss = CLW[i] * torch.nn.functional.relu(prob_- probs[i]).sum()
    return CLoss
