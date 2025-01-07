import torch.nn as nn
import torch.utils.model_zoo as model_zoo
class Residual(nn.Module):
    def __init__(self, img_ch, out_ch, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(img_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv2d(img_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, x):
        x1 = self.conv1(x)
        x1_5 = self.conv3(x)
        x = x1 + x1_5
        x2 = self.conv2(x)
        x = x + x2
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels=3 , output_ch = 2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = Residual(img_ch=64, out_ch=128)
        self.conv3 = Residual(img_ch=128, out_ch=256)
        self.conv4 = Residual(img_ch=256, out_ch=512)
        self.conv5 = Residual(img_ch=512, out_ch=1024)
        self.MAX = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x = self.conv_1(x1)
        x = self.bn2(x)
        x = self.relu(x)
        feat1 = x + x1
        x = self.MAX(feat1)
        feat2 = self.conv2(x)
        x = self.MAX(feat2)
        feat3 = self.conv3(x)
        x = self.MAX(feat3)
        feat4 = self.conv4(x)
        x = self.MAX(feat4)
        feat5 = self.conv5(x)
        #print(f"layer1 output shape: {feat5.shape}")
        return [feat1, feat2, feat3, feat4, feat5]

def RESD(pretrained=False, **kwargs):
    model = ResidualBlock(**kwargs)
    if pretrained:
        model.load_state_dict(
            model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth', model_dir='model_data'),
            strict=False)


    return model