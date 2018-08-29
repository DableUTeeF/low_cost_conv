import torch.nn as nn
import torch.nn.functional as F
from pytorch_lowcost_conv import Lowcost


class LowcostConv(nn.Module):
    def __init__(self, in_planes, planes, w, h, stride=1, padding=1, multiply=4):
        super().__init__()
        self.conv = nn.Conv2d(in_planes,
                              int(planes / multiply),
                              kernel_size=3,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.squeeze = Lowcost(int(planes / multiply), planes, w, h)
        # self.expand = Lowcost(in_planes*multiply, out_planes, w, h)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.squeeze(x)
        # x = self.expand(x)
        x = self.bn(x)
        return F.relu(x)


class LowcostDWConv(nn.Module):
    def __init__(self, in_planes, planes, w, h, stride=1, padding=1, multiply=4):
        super().__init__()
        self.conv = nn.Conv2d(in_planes,
                              int(planes / multiply),
                              kernel_size=1,
                              stride=stride,
                              padding=0,
                              bias=False)
        self.dw = nn.Conv2d(int(planes / multiply),
                            int(planes / multiply),
                            groups=int(planes / multiply),
                            kernel_size=3,
                            stride=1,
                            padding=padding,
                            bias=False)
        self.squeeze = Lowcost(int(planes / multiply), planes, w, h)
        # self.expand = Lowcost(in_planes*multiply, out_planes, w, h)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.dw(x)
        x = self.squeeze(x)
        # x = self.expand(x)
        x = self.bn(x)
        return F.relu(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, w, h, stride=1, activation='relu'):
        super(BasicBlock, self).__init__()
        self.activation = activation
        self.conv1 = LowcostDWConv(in_planes, planes, w, h, stride=stride)
        self.conv2 = LowcostDWConv(planes, planes, w, h, stride=1)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, activation='relu'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.activation = activation
        self.conv1 = LowcostConv(3, 64, 32, 32, stride=1, padding=1, multiply=4)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 32, 32, stride=1, activation=activation)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 16, 16, stride=2, activation=activation)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 8, 8, stride=2, activation=activation)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layer(self, block, planes, num_blocks, w, h, stride, activation='relu'):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, w, h, stride, activation))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
