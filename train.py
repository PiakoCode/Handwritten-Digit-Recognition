import datetime
import torch
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from ResNet18 import ResNet
from data import train_dataloader, test_dataloader, train_data_size, test_data_size


torch.manual_seed(89)

lr = 0.01
epoch = 60


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("now using device: ", device)

start_time = datetime.datetime.now().strftime("%m月%d日%H:%M")

image_size = (1, 28, 28)


if __name__ == "__main__":

    model = ResNet()
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
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

    loss_min = np.Inf

    for num_epochs in range(epoch):
        print("-------第 {} 轮训练开始-------".format(num_epochs + 1))
        # 训练步骤开始
        model.train()
        for batch_idx, (data) in enumerate(train_dataloader):

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
            if not batch_idx % 100:
                print(
                    f"Epoch: {num_epochs+1:03d}/{epoch:03d} | Batch {batch_idx:04d}/{len(train_dataloader):04d} | Loss: {loss:.4f}"
                )
                train_loss_list.append(loss.item())
            # if total_train_step % 100 == 0:
            #     print(
            #         "训练次数: {}, Loss: {}".format(total_train_step, loss.item()),
            #         flush=True,
            #     )
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
            "整体测试集上的正确率: {}".format(total_accuracy / test_data_size),
            flush=True,
        )
        test_loss_list.append(total_test_loss)
        accuracy_list.append(total_accuracy.cpu().numpy() / test_data_size)
        total_test_step = total_test_step + 1
        scheduler.step(total_test_loss)

        if total_test_loss < loss_min:
            loss_min = total_test_loss
            torch.save(model, f"model_{start_time}.pth")
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
    plt.savefig(f"result_{start_time}.png")
