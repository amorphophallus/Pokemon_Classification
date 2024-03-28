import torch
import torch.nn as nn
from torchinfo import summary


class BasicBlock(nn.Module):
    """
    ResNet 的基础模块，也就是 residual 结构，分为降维和不降维两种
    """
    def __init__(self, in_channel, out_channel, is_downsample):
        super(BasicBlock, self).__init__()

        # 参数
        self.is_downsample = is_downsample
        self.stride = 2 if is_downsample else 1

        # 模块
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=self.stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.identity = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=self.stride)

    def forward(self, x):
        identity =self.identity(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.relu(x)
        return x


class ResNet18(nn.Module):
    """
    用 BasicBlock 组成 ResNet
    """
    def __init__(self, in_channel, num_classes):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicBlock(in_channel=64, out_channel=64, is_downsample=False),
            BasicBlock(in_channel=64, out_channel=64, is_downsample=False)
        )
        self.conv3 = nn.Sequential(
            BasicBlock(in_channel=64, out_channel=128, is_downsample=True),
            BasicBlock(in_channel=128, out_channel=128, is_downsample=False)
        )
        self.conv4 = nn.Sequential(
            BasicBlock(in_channel=128, out_channel=256, is_downsample=True),
            BasicBlock(in_channel=256, out_channel=256, is_downsample=False)
        )
        self.conv5 = nn.Sequential(
            BasicBlock(in_channel=256, out_channel=512, is_downsample=True),
            BasicBlock(in_channel=512, out_channel=512, is_downsample=False)
        )
        self.predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.predictor(x)
        return x


class Lenet5(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(Lenet5, self).__init__()  # 利用参数初始化父类
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16 * 54 * 54, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=num_classes),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.classifier(self.feature(x))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet18(in_channel=3, num_classes=159).to(device)
    # model = Lenet5(in_channel=3, num_classes=159).to(device)
    print(
        summary(model=model, input_size=(32, 3, 224, 224), col_width=20,
                col_names=['input_size', 'output_size'], row_settings=['var_names'],
                verbose=0)
    )