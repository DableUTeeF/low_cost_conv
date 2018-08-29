import torch
from torch import nn


class Lowcost(nn.Module):
    def __init__(self, c, m, w, h):
        super(Lowcost, self).__init__()
        self.m = m
        self.w = w
        self.c = c
        self.h = h
        self.weights = torch.empty(m, c, requires_grad=True).cuda()
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x):
        x = x.view(x.size(0), self.c, self.w * self.h)
        w = self.weights.unsqueeze(0).expand(x.size(0), self.m, self.c)
        x = torch.bmm(w, x)
        x = x.view(x.size(0), self.m, self.w, self.h)
        return x
