import torch.nn as nn
import torch.nn.functional as F
from pytorch_lowcost_conv import Lowcost


def swish(x):
    return F.sigmoid(x) * x


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


class Lowcost2BN(nn.Module):
    def __init__(self, in_planes, planes, w, h, stride=1, padding=1, multiply=4):
        super().__init__()
        self.conv = nn.Conv2d(in_planes,
                              int(planes / multiply),
                              kernel_size=3,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn1 = nn.BatchNorm2d(int(planes / multiply))
        self.squeeze = Lowcost(int(planes / multiply), planes, w, h)
        # self.expand = Lowcost(in_planes*multiply, out_planes, w, h)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.squeeze(x)
        # x = self.expand(x)
        x = self.bn2(x)
        return F.relu(x)


class LowcostSqueeze(nn.Module):
    def __init__(self, in_planes, planes, w, h, stride=1, padding=1, multiply=4):
        super().__init__()
        # self.squeeze = Lowcost(in_planes, int(planes / multiply), w*stride, h*stride)
        self.conv = nn.Conv2d(in_planes,
                              int(planes / multiply),
                              kernel_size=3,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.expand = Lowcost(int(planes / multiply), planes, w, h)
        self.bn = nn.BatchNorm2d(int(planes / multiply))

    def forward(self, x):
        # x = self.squeeze(x)
        x = self.conv(x)
        x = self.bn(x)
        x = swish(x)
        x = self.expand(x)
        return x


class LowcostDW2BN(nn.Module):
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
        self.bn1 = nn.BatchNorm2d(int(planes / multiply))
        self.squeeze = Lowcost(int(planes / multiply), planes, w, h)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.dw(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.squeeze(x)
        # x = self.expand(x)
        x = self.bn2(x)
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
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.dw(x)
        x = self.squeeze(x)
        x = self.bn(x)
        return F.relu(x)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation='relu'):
        super(ResBlock, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        out += self.shortcut(x)
        out = out
        return out


#
lowcost = LowcostDWConv
#


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, w, h, stride=1, activation='relu'):
        super(BasicBlock, self).__init__()
        self.activation = activation
        self.conv1 = lowcost(in_planes, planes, w, h, stride=stride)
        self.conv2 = lowcost(planes, planes, w, h, stride=1)

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


class DWBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, w, h, stride=1, activation='relu', multiply=4):
        super(DWBlock, self).__init__()
        self.activation = activation
        self.dw1 = LowcostDWConv(in_planes, planes, w, h, stride=stride, multiply=multiply)
        self.dw2 = LowcostDWConv(planes, planes, w, h, stride=1, multiply=multiply)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.dw1(x)
        out = self.dw2(out)
        out += self.shortcut(x)
        return out


class ConvBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, w, h, stride=1, activation='relu', multiply=4):
        super(ConvBlock, self).__init__()
        self.activation = activation
        self.conv1 = LowcostSqueeze(in_planes, planes, w, h, stride=stride, multiply=multiply)
        self.conv2 = LowcostSqueeze(planes, planes, w, h, stride=1, multiply=multiply)

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
        self.conv1 = LowcostConv(3, 64, 32, 32, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 32, 32, stride=1, activation=activation)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 16, 16, stride=2, activation=activation)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 8, 8, stride=2, activation=activation)
        # self.conv2 = LowcostConv(2048, 256, 32, 32, stride=1, padding=1)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layer(self, block, planes, num_blocks, w, h, stride, activation='relu'):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            if block == ResBlock:
                layers.append(block(self.in_planes, planes, stride, activation))
            else:
                layers.append(block(self.in_planes, planes, w, h, stride, activation))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.conv2(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class LowcostNetA(nn.Module):
    m = 4

    def __init__(self, block, num_blocks, num_classes=10):
        super(LowcostNetA, self).__init__()
        self.in_planes = 64
        self.conv1 = LowcostConv(3, 128, 32, 32, stride=1, multiply=self.m)
        self.dw1 = DWBlock(128, 128, 32, 32, stride=1, multiply=self.m)
        self.conv2 = ConvBlock(128, 256, 16, 16, stride=2, multiply=self.m)
        self.dw2 = DWBlock(256, 256, 16, 16, stride=1, multiply=self.m)
        self.conv3 = ConvBlock(256, 512, 8, 8, stride=2, multiply=self.m)
        self.dw3 = DWBlock(512, 512, 8, 8, stride=1, multiply=self.m)
        # self.conv2 = LowcostConv(2048, 256, 32, 32, stride=1, padding=1)
        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.dw1(out)
        out = self.conv2(out)
        out = self.dw2(out)
        out = self.conv3(out)
        out = self.dw3(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class LowcostNetC(nn.Module):
    m = 2

    def __init__(self, block, num_blocks, num_classes=10):
        super(LowcostNetC, self).__init__()
        self.in_planes = 64
        self.conv1 = LowcostConv(3, 64, 32, 32, stride=1, multiply=self.m)
        self.dw1 = ConvBlock(64, 64, 32, 32, stride=1, multiply=self.m)
        self.conv2 = ConvBlock(64, 128, 16, 16, stride=2, multiply=self.m)
        self.dw2 = ConvBlock(128, 128, 16, 16, stride=1, multiply=self.m)
        self.conv3 = ConvBlock(128, 256, 8, 8, stride=2, multiply=self.m)
        self.dw3 = ConvBlock(256, 256, 8, 8, stride=1, multiply=self.m)
        # self.conv2 = LowcostConv(2048, 256, 32, 32, stride=1, padding=1)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu6(out)
        out = self.dw1(out)
        out = self.conv2(out)
        out = self.dw2(out)
        out = self.conv3(out)
        out = self.dw3(out)
        out = F.relu6(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class LowcostNetD(nn.Module):
    m = 2

    def __init__(self, block, num_blocks, num_classes=10):
        super(LowcostNetD, self).__init__()
        self.in_planes = 64
        self.conv1 = LowcostConv(3, 64, 32, 32, stride=1, multiply=self.m)
        self.block1a = ConvBlock(64, 64, 32, 32, stride=1, multiply=self.m)
        self.block1b = ConvBlock(64, 64, 32, 32, stride=1, multiply=self.m)
        self.block2a = ConvBlock(64, 128, 16, 16, stride=2, multiply=self.m)
        self.block2b = ConvBlock(128, 128, 16, 16, stride=1, multiply=self.m)
        self.block3a = ConvBlock(128, 256, 8, 8, stride=2, multiply=self.m)
        self.block3b = ConvBlock(256, 256, 8, 8, stride=1, multiply=self.m)
        # self.conv2 = LowcostConv(2048, 256, 32, 32, stride=1, padding=1)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.block1a(out)
        out = self.block1b(out)
        out = self.block2a(out)
        out = self.block2b(out)
        out = self.block3a(out)
        out = self.block3b(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class LowcostNetE(nn.Module):
    m = 2

    def __init__(self, block, num_blocks, num_classes=10):
        super(LowcostNetE, self).__init__()
        self.in_planes = 64
        self.conv1 = LowcostSqueeze(3, 64, 32, 32, stride=1, multiply=self.m)
        self.block1a = ConvBlock(64, 64, 32, 32, stride=1, multiply=self.m)
        self.block1b = ConvBlock(64, 64, 32, 32, stride=1, multiply=self.m)
        self.block2a = ConvBlock(64, 128, 16, 16, stride=2, multiply=self.m)
        self.block2b = ConvBlock(128, 128, 16, 16, stride=1, multiply=self.m)
        self.block3a = ConvBlock(128, 256, 8, 8, stride=2, multiply=self.m)
        self.block3b = ConvBlock(256, 256, 8, 8, stride=1, multiply=self.m)
        # self.conv2 = LowcostConv(2048, 256, 32, 32, stride=1, padding=1)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.block1a(out)
        out = self.block1b(out)
        out = self.block2a(out)
        out = self.block2b(out)
        out = self.block3a(out)
        out = self.block3b(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class LowcostNetB(nn.Module):
    m = 2

    def __init__(self, block, num_blocks, num_classes=10):
        super(LowcostNetB, self).__init__()
        self.in_planes = 64
        self.conv1 = LowcostConv(3, 128, 32, 32, stride=1, multiply=self.m)
        self.dw1 = DWBlock(64, 64, 32, 32, stride=1, multiply=self.m)
        self.conv2 = DWBlock(64, 128, 16, 16, stride=2, multiply=self.m)
        self.dw2 = DWBlock(128, 128, 16, 16, stride=1, multiply=self.m)
        self.conv3 = DWBlock(128, 256, 8, 8, stride=2, multiply=self.m)
        self.dw3 = DWBlock(256, 256, 8, 8, stride=1, multiply=self.m)
        # self.conv2 = LowcostConv(2048, 256, 32, 32, stride=1, padding=1)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.dw1(out)
        out = self.conv2(out)
        out = self.dw2(out)
        out = self.conv3(out)
        out = self.dw3(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
