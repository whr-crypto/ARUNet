import torch
import torch.nn as nn
from nets.residual import RESD

from functools import partial



class Residuals(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class Channel_Attention(nn.Module):
    def __init__(self, channel, ratio = 16):
        super(Channel_Attention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        avg = self.avgpool(x).view([b, c])
        fc = self.fc(avg).view(b, c, 1, 1)
        return x * fc
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.up(x)

        return x

class up_conv_1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv_1, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.atten = self_attention(ch_out)
    def forward(self, x):
        x = self.up(x)
        x = self.atten(x)
        return x
class SEConv(nn.Module):
    def __init__(self, inch, outch):
        super(SEConv, self).__init__()
        self.attention = Channel_Attention(channel=inch)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inch, out_channels=outch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        #x = self.attention(x)
        x = self.conv(x)
        return x
class self_attention(nn.Module):
    def __init__(self, channel):
        super(self_attention, self).__init__()
        self.channel = channel
        self.pointwiseConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.ordinaryConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.dilationConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=2, stride=1, dilation=2, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.voteConv = nn.Sequential(
            nn.Conv2d(self.channel * 2, self.channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.channel),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.pointwiseConv(x)
        x2 = self.ordinaryConv(x)
        x3 = self.dilationConv(x)
        _x = self.relu(torch.cat((x1, x2), dim=1))
        _x = self.voteConv(_x)
        x = x + x3 * _x
        return x


class ARU_Net(nn.Module):
    def __init__(self, num_classes=2, pretrained=False, backbone = 'rau'):
        super(RAU_Net, self).__init__()
        if backbone == 'rau':
            self.RA = RESD(pretrained=pretrained)
        self.sp = SpConvMixerBlock()
        self.up_1 = up_conv(1024, 512)
        self.up_2 = up_conv_1(512, 256)
        self.up_3 = up_conv_1(256, 128)
        self.up_4 = up_conv_1(128, 64)
        self.se_1 = SEConv(1024, 512)
        self.se_2 = SEConv(512, 256)
        self.se_3 = SEConv(256, 128)
        self.se_4 = SEConv(128, 64)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.att = space_attention()
        self.backbone = backbone
    def forward(self, inputs):
        if self.backbone == "rau":
            [feat1, feat2, feat3, feat4, feat5] = self.RA.forward(inputs)
        #feat5 = self.att(feat5)
        #feat5 = self.sp(feat5)
        x = self.up_1(feat5)
        x = torch.cat((x, feat4), dim=1)
        x = self.se_1(x)
        x = self.up_2(x)
        x = torch.cat((x, feat3), dim=1)
        x = self.se_2(x)
        x = self.up_3(x)
        x = torch.cat((x, feat2), dim=1)
        x = self.se_3(x)
        x = self.up_4(x)
        x = torch.cat((x, feat1), dim=1)
        x = self.se_4(x)
        x = self.conv1(x)
        #print(f"layer1 output shape: {d1.shape}")
        return x
    def freeze_backbone(self):
        if self.backbone == "rau":
            for param in self.RA.parameters():
                param.requires_grad = False


    def unfreeze_backbone(self):
        if self.backbone == "rau":
            for param in self.RA.parameters():
                param.requires_grad = True
'''def BR():
    model = RAU_Net()
    return model
def test():
    net = BR()
    x = torch.randn(1, 3, 256, 256)
    y = net(x)
    print(y)
test()'''

class SpConvMixerBlock(nn.Module):
    def __init__(self, dim=1024, depth=5, k=7):
        super(SpConvMixerBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residuals(nn.Sequential(
                    # deep wise
                    nn.Conv2d(dim, dim, kernel_size=(k, k), groups=dim, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                    nn.Conv2d(dim, dim, kernel_size=(1, 1)),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)))
                ) for i in range(depth)]
            )

    def forward(self, x):

        x = self.block(x)
        #print(f"layer2 output shape: {x.shape}")

        #print(f"layer1 output shape: {x.shape}")
        return x

class space_attention(nn.Module):
    def __init__(self, kernel_size=5):
        super(space_attention, self).__init__()
        padding = kernel_size // 2
        # 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        # sigmoid函数
        self.sigmoid = nn.Sigmoid()

        # 前向传播

    def forward(self, x):
        # 在通道维度上最大池化 [b,1,h,w]  keepdim保留原有深度
        # 返回值是在某维度的最大值和对应的索引
        x_maxpool, _ = torch.max(x, dim=1, keepdim=True)

        # 在通道维度上平均池化 [b,1,h,w]
        x_avgpool = torch.mean(x, dim=1, keepdim=True)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]
        x1 = torch.cat([x_maxpool, x_avgpool], dim=1)

        # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        x1 = self.conv(x1)
        # 空间权重归一化
        x1 = self.sigmoid(x1)
        # 输入特征图和空间权重相乘
        x = x * x1
        #print(f"layer2 output shape: {outputs.shape}")
        return x



















