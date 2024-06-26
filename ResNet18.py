from torch import nn

class Block(nn.Module):
    def __init__(self, input_channel, output_channel, stride=1, use1x1=False):
        super(Block, self).__init__()
        # 3*3卷积层_1
        self.conv1 = nn.Conv2d(
            in_channels=input_channel,
            out_channels=output_channel,
            kernel_size=3,
            padding=1,
            stride=stride,
        )

        # 3*3卷积层_2
        self.conv2 = nn.Conv2d(
            in_channels=output_channel,
            out_channels=output_channel,
            kernel_size=3,
            padding=1,
            stride=1,
        )

        # 批量规范化层

        self.bn1 = nn.BatchNorm2d(output_channel)
        self.bn2 = nn.BatchNorm2d(output_channel)

        self.relu = nn.ReLU()

        self.conv1x1 = None
        # 1*1 卷积层
        if use1x1 == True:
            self.conv1x1 = nn.Conv2d(
                in_channels=input_channel,
                out_channels=output_channel,
                kernel_size=1,
                stride=stride,
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.conv1x1:
            x = self.conv1x1(x)

        out += x
        out = self.relu(out)
        return out


def block_generate(input_channel, output_channel, num_block):
    blocks = []
    for i in range(num_block):
        if i == 0:
            blocks.append(Block(input_channel, output_channel, stride=2, use1x1=True))
        else:
            blocks.append(Block(output_channel, output_channel))
    return blocks

# ResNet
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.net = nn.Sequential(
            # 考虑到MNIST数据集的图片尺寸太小，ResNet18网络的7x7降采样卷积和池化操作容易丢失一部分信息
            # 所以我们将7x7的降采样层和最大池化层去掉，替换为一个3x3的降采样卷积，同时减小该卷积层的步长和填充大小，
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            # *block_generate(64, 64, 2),
            Block(64, 64),
            Block(64, 64),
            *block_generate(64, 128, 2),
            *block_generate(128, 256, 2),
            *block_generate(256, 512, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)
