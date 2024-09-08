import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

# __all__ = ['Res2Net', 'res2net18', 'res2net101_v1b']

# model_urls = {
#     'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
#     'res2net101_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth',
# }


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, scale=4):
        """ Constructor
               Args:
                   inplanes: input channel dimensionality
                   planes: output channel dimensionality
                   stride: conv stride. Replaces pooling layer.
                   downsample: None when stride = 1
                   baseWidth: basic width of conv3x3
                   scale: number of scale --- 控制每个分支的宽度
                   type: 'normal': normal set. 'stage': first block of a new stage.
               """
        super(Bottle2neck, self).__init__()
        # 每个分支的宽度
        width = int(planes * (scale / 4))
        #1*1的卷积核
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        #3*3的卷积核
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes * Bottle2neck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes,
                          planes * Bottle2neck.expansion,
                          stride=stride,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(planes * Bottle2neck.expansion))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class Res2Net(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super(Res2Net, self).__init__()
        self.in_channel = 64
        # 输入图像的通道数为3（即RGB图像），输出通道数为64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channel, num_blocks, stride):
        strides = [stride] + (num_blocks - 1) * [1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, out_channel, stride))
            self.in_channel = out_channel * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

def res2net18():
    """Constructs a Res2Net-18_v1b model.
    Res2Net-18 refers to the Res2Net-50_v1b_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [2, 2, 2, 2])
    return model

# if __name__ == '__main__':
#     images = torch.rand(1, 3, 224, 224).cuda(0)
#     model = res2net50_v1b_26w_4s(pretrained=True)
#     model = model.cuda(0)
#     print(model(images).size())