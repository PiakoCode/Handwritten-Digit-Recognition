import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler

dataset_path = "../dataset"

lr = 0.001
batch_size = 64
epoch = 5


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

now_time = datetime.datetime.now().strftime("%m月%d日%H:%M")

trans = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# 数据集: MNIST
train_data = torchvision.datasets.MNIST(
    root=dataset_path,
    train=True,
    transform=trans,
    download=True,
)
test_data = torchvision.datasets.MNIST(
    root=dataset_path,
    train=False,
    transform=trans,
    download=True,
)

# length 长度


train_data_size = len(train_data)
test_data_size = len(test_data)

# # 如果train_data_size=10, 训练数据集的长度为：10
# print("训练数据集的长度为：{}".format(train_data_size))
# print("测试数据集的长度为：{}".format(test_data_size))


train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

image_size = (1, 28, 28)


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
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
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

if __name__ == "__main__":

    model = ResNet()
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
    )
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 20, 30, 40, 50, 60, 70, 80, 90], gamma=0.8
    )

    loss_func = nn.CrossEntropyLoss()
    loss_func = loss_func.to(device)

    # ====== Train ===== #

    # 设置训练网络的一些参数
    # 记录训练的次数
    total_train_step = 0
    # 记录测试的次数
    total_test_step = 0

    train_loss_list = []
    test_loss_list = []
    accuracy_list = []

    for i in range(epoch):
        print("-------第 {} 轮训练开始-------".format(i + 1))
        # 训练步骤开始
        model.train()
        for data in train_dataloader:

            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)

            loss = loss_func(outputs, targets)

            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:
                print(
                    "训练次数: {}, Loss: {}".format(total_train_step, loss.item()),
                    flush=True,
                )
                train_loss_list.append(loss.item())
        scheduler.step()
        for param_group in optimizer.param_groups:
            print("当前学习率: ", param_group["lr"])
        # 测试步骤开始
        model.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data in test_dataloader:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)
                loss = loss_func(outputs, targets)
                total_test_loss = total_test_loss + loss.item()
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy = total_accuracy + accuracy

        print("整体测试集上的Loss: {}".format(total_test_loss))
        print(
            "整体测试集上的正确率: {}".format(total_accuracy / test_data_size), flush=True
        )
        test_loss_list.append(total_test_loss)
        accuracy_list.append(total_accuracy.cpu().numpy() / test_data_size)
        total_test_step = total_test_step + 1


    torch.save(model, f"model_{now_time}.pth")
    print("模型已保存")

    # 绘图
    x_train = np.linspace(1, len(train_loss_list), num=len(train_loss_list))
    x_test = np.linspace(1, len(test_loss_list), num=len(test_loss_list))

    plt.figure(figsize=(20, 30))

    plt.subplot(3, 1, 1)
    plt.plot(x_train, train_loss_list, color="r", label="train_loss")
    plt.grid(True)
    plt.legend()

    print(test_loss_list)
    print(accuracy_list)
    plt.subplot(3, 1, 2)
    plt.plot(x_test, test_loss_list, color="g", label="test_loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(x_test, accuracy_list, color="b", label="accuracy")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"result_{now_time}.png")
