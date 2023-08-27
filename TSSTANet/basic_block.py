import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SpatialBlock',
           'TemporalBlock',
           'SpatiotemporalBlock',
           'AttentionSpatiotemporalBlock']


class SpatialBlock(nn.Module):
    """Spatial attention block

        Arguments:
            in_channels (int): the dim of the whole tensor flowing into SpatialBlock
            inner_channels (int): the dim of total tensor among k-group
            k (int): inner groups
        """

    def __init__(self, in_channels, inner_channels, kernel_size, stride, padding, dilation, k=2):
        super(SpatialBlock, self).__init__()
        self.k = k

        self.group_conv = nn.Conv3d(in_channels, inner_channels, kernel_size,
                                    stride, padding, dilation, groups=k)
        self.bn = nn.BatchNorm3d(inner_channels)
        self.relu = nn.ReLU(inplace=True)

        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.fc = nn.Conv3d(inner_channels, inner_channels, 1, groups=k)
        self.softmax = SoftMax(k)

    def forward(self, x):
        x = self.group_conv(x)
        x = self.bn(x)
        x = self.relu(x)

        att = self.adaptive_pool(x)
        att = self.fc(att).view(x.size(0), self.k, x.size(1) // self.k, 1, x.size(3), x.size(4))
        att = self.softmax(att).view(x.size(0), -1, 1, x.size(3), x.size(4))

        x = x * att
        return x


class TemporalBlock(nn.Module):
    """Temporal attention block

    Arguments:
        in_channels (int): the dim of the whole tensor flowing into TemporalBlock
        inner_channels (int): the dim of total tensor among k-group
        k (int): inner groups
    """
    def __init__(self, in_channels, inner_channels, kernel_size, stride, padding, dilation, k=2):
        super(TemporalBlock, self).__init__()
        self.k = k

        self.group_conv = nn.Conv3d(in_channels, inner_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, groups=k)
        self.bn = nn.BatchNorm3d(inner_channels)
        self.relu = nn.ReLU(inplace=True)

        self.adaptive_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        self.fc = nn.Conv3d(inner_channels, inner_channels, kernel_size=1, groups=k)
        self.softmax = SoftMax(k)

    def forward(self, x):
        x = self.group_conv(x)
        x = self.bn(x)
        x = self.relu(x)

        att = self.adaptive_pool(x)
        att = self.fc(att).view(x.size(0), self.k, x.size(1) // self.k, x.size(2), 1, 1)
        att = self.softmax(att).view(x.size(0), x.size(1), x.size(2), 1, 1)

        x = x * att

        return x


class SpatiotemporalBlock(nn.Module):
    """Spatio-Temporal Fusing attention block

    Arguments:
        in_channels (int): the dim of the whole tensor flowing into SpatiotemporalBlock
        inner_channels (int): the dim of total tensor among k-group in temporalBlock
        k (int): inner groups of each block
    """
    def __init__(self, in_channels, inner_channels, kernel_size, stride, padding, dilation, k=2):
        super(SpatiotemporalBlock, self).__init__()
        self.k = k
        self.temporal_block = TemporalBlock(in_channels=in_channels, inner_channels=inner_channels,
                                            kernel_size=kernel_size, stride=stride,
                                            padding=padding, dilation=dilation, k=k)
        self.spatial_block = SpatialBlock(in_channels=in_channels, inner_channels=inner_channels,
                                          kernel_size=kernel_size, stride=stride,
                                          padding=padding, dilation=dilation, k=k)

    def forward(self, x):
        x_t = self.temporal_block(x)
        x_s = self.spatial_block(x)
        x_t = x_t.view(x_t.size(0), self.k, x_t.size(1) // self.k, x_t.size(2), x_t.size(3), x_t.size(4))
        x_s = x_s.view(x_s.size(0), self.k, x_s.size(1) // self.k, x_s.size(2), x_s.size(3), x_s.size(4))
        x = x_s + x_t
        x = x.view(x_t.size(0), x_t.size(2) * self.k, x_t.size(3), x_t.size(4), x_t.size(5))
        return x


class AttentionSpatiotemporalBlock(nn.Module):
    """Spatio-Temporal attention vector fusing attention block

    Arguments:
        in_channels (int): the dim of the whole tensor flowing into this block
        inner_channels (int): the dim of total tensor among k-group in two block
        k (int): inner groups of each block
    """

    def __init__(self, in_channels, inner_channels, kernel_size, stride, padding, dilation, k=2):
        super(AttentionSpatiotemporalBlock, self).__init__()
        self.k = k

        # temporal block before attention vector
        self.group_conv_temporal = nn.Conv3d(in_channels, inner_channels, kernel_size,
                                             stride, padding, dilation, groups=k)
        self.bn_temporal = nn.BatchNorm3d(inner_channels)
        self.adaptive_pool_temporal = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.fc_temporal = nn.Conv3d(inner_channels, inner_channels, kernel_size=1, groups=k)

        # spatial block before attention vector
        self.group_conv_spatial = nn.Conv3d(in_channels, inner_channels, kernel_size,
                                            stride, padding, dilation, groups=k)
        self.bn_spatial = nn.BatchNorm3d(inner_channels)
        self.adaptive_pool_spatial = nn.AdaptiveAvgPool3d((1, None, None))
        self.fc_spatial = nn.Conv3d(inner_channels, inner_channels, 1, groups=k)

        self.relu = nn.ReLU(inplace=True)
        self.softmax = SoftMax(k)

    def forward(self, x):
        # temporal block before attention vector
        x_t = self.group_conv_temporal(x)
        x_t = self.bn_temporal(x_t)
        x_t = self.relu(x_t)
        attention_t = self.adaptive_pool_temporal(x_t)
        attention_t = self.fc_temporal(attention_t).view(x_t.size(0), self.k, x_t.size(1) // self.k,
                                                         x_t.size(2), 1, 1)
        attention_t = self.softmax(attention_t)

        # spatial block before attention vector
        x_s = self.group_conv_spatial(x)
        x_s = self.bn_spatial(x_s)
        x_s = self.relu(x_s)
        attention_s = self.adaptive_pool_spatial(x_s)
        attention_s = self.fc_spatial(attention_s).view(x_s.size(0), self.k, x_s.size(1) // self.k,
                                                        1, x_s.size(3), x_s.size(4))
        attention_s = self.softmax(attention_s)

        x = attention_s * attention_s
        x = x.view(x.size(0), x.size(2) * self.k, x.size(3), x.size(4), x.size(5))
        x = x * (x_s + x_t)

        return x


class SoftMax(nn.Module):
    def __init__(self, k):
        super(SoftMax, self).__init__()
        self.k = k

    def forward(self, x):
        # batch = x.size(0)
        if self.k > 1:
            # x = x.view(batch, self.k, -1)
            x = F.softmax(x, dim=1)
            # x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


if __name__ == '__main__':

    # net = TemporalBlock(in_channels=32, inner_channels=64, kernel_size=3,
    #                     stride=2, padding=1, dilation=1, k=2).to('cuda')
    #
    # net = SpatialBlock(in_channels=32, inner_channels=32, kernel_size=3,
    #                    stride=2, padding=1, dilation=1, k=2).to('cuda')

    net = SpatiotemporalBlock(in_channels=32, inner_channels=64, kernel_size=3,
                                       stride=2, padding=1, dilation=1, k=2).to('cuda')

    train = torch.rand(2, 32, 64, 128, 128).to('cuda')

    print(net(train).shape)

    # x = torch.rand(2, 4)
    # print(x)
    # softmax = SoftMax(2)
    # print(softmax(x))
