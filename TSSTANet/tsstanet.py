import torch
import torch.nn as nn

from TSSTANet.basic_block import SpatialBlock, TemporalBlock, SpatiotemporalBlock, AttentionSpatiotemporalBlock

__all__ = ['tanet', 'sanet', 'stanet', 'stanet_af']


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, module, stride=1, padding=1, k=2, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        self.conv2 = module(in_channels=width, inner_channels=width, kernel_size=3,
                            stride=stride, padding=padding, dilation=dilation, k=k)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)

        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, module, layers, in_channels, num_classes, features=64,
                 groups=1, k=2, width_per_group=64, norm_layer=None, drop_prob=0.0):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.k = k
        self.module = module
        self.downsample_stride = (1, 2, 2)

        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.conv_1 = nn.Conv3d(in_channels, self.inplanes, kernel_size=7, stride=self.downsample_stride, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=self.downsample_stride, padding=1)

        self.layer_1 = self._make_layer(Bottleneck, features, layers[0] * k, stride=1)
        self.layer_2 = self._make_layer(Bottleneck, features * 2, layers[1] * k, stride=2)
        self.layer_3 = self._make_layer(Bottleneck, features * 4, layers[2] * k, stride=2)
        self.layer_4 = self._make_layer(Bottleneck, features * 8, layers[3] * k, stride=2)
        self.fc = nn.Linear(features * 8 * Bottleneck.expansion, num_classes)
        """
            tiny network
        """
        # self.layer_1 = self._make_layer(Bottleneck, 8, layers[0] * k, stride=1)
        # self.layer_2 = self._make_layer(Bottleneck, 16, layers[1] * k, stride=self.downsample_stride)
        # self.layer_3 = self._make_layer(Bottleneck, 32, layers[2] * k, stride=self.downsample_stride)
        # self.layer_4 = self._make_layer(Bottleneck, 64, layers[3] * k, stride=self.downsample_stride)
        # self.fc = nn.Linear(64 * Bottleneck.expansion, num_classes)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(drop_prob)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv3x3(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, self.module, stride=stride, k=self.k,
                        downsample=downsample, groups=self.groups,
                        base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.module, k=self.k, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)

        x = self.avgpool(x).view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def tanet(layers, in_channels, num_classes, **kwargs):
    return ResNet(TemporalBlock,
                  layers=layers,
                  in_channels=in_channels,
                  num_classes=num_classes,
                  **kwargs)


def sanet(layers, in_channels, num_classes, **kwargs):
    return ResNet(SpatialBlock,
                  layers=layers,
                  in_channels=in_channels,
                  num_classes=num_classes,
                  **kwargs)


def stanet(layers, in_channels, num_classes, **kwargs):
    return ResNet(SpatiotemporalBlock,
                  layers=layers,
                  in_channels=in_channels,
                  num_classes=num_classes,
                  **kwargs)


def stanet_af(layers, in_channels, num_classes, **kwargs):
    return ResNet(AttentionSpatiotemporalBlock,
                  layers=layers,
                  in_channels=in_channels,
                  num_classes=num_classes,
                  **kwargs)


if __name__ == '__main__':
    # net = tanet(in_channels=3, num_classes=101).to('cuda')
    # train = torch.rand(2, 3, 32, 64, 64).to('cuda')
    # print(net(train).shape)

    from torchkeras import summary

    # net = sanet(in_channels=3, num_classes=101, k=8)
    # net = tanet(in_channels=3, num_classes=101, k=2)
    net = stanet_af(layers=[2, 2, 2, 2], in_channels=1, num_classes=101, k=2)
    summary(net, (1, 64, 128, 128))
