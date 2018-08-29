from lowcost_resnet import *
import torch
from viz import make_dot, make_dot_from_trace

input_ = torch.randn((1, 3, 32, 32))

model = ResNet(BasicBlock, [2, 2, 2], activation='relu')
# model = torch.nn.DataParallel(model).cuda()
print(model)
y = model(input_)
make_dot(y.mean(), params=dict(model.named_parameters())).view()
