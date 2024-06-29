from matplotlib import pyplot as plt
import numpy as np
import torch


def plot_training_metrics(log_dict, num_epochs):
    loss_list = log_dict["train_loss_per_batch"]
    train_acc = log_dict["train_acc_per_epoch"]
    test_acc = log_dict["test_acc_per_epoch"]
    model_name = log_dict["model_name"]
    train_loss_per_epoch = log_dict["train_loss_per_epoch"]
    test_loss_per_epoch = log_dict["test_loss_per_epoch"]

    running_avg_loss = np.convolve(loss_list, np.ones(200) / 200, mode="valid")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].plot(train_loss_per_epoch, label="Training Loss")
    axs[0].plot(test_loss_per_epoch, label="Test Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title(f"Loss on {model_name}")
    axs[0].legend(loc="best")
    axs[0].grid(True)

    # plot training accuracy
    axs[1].plot(
        np.arange(1, len(train_acc) + 1),
        train_acc,
        label="Training Accuracy",
    )
    axs[1].plot(
        np.arange(1, len(test_acc) + 1),
        test_acc,
        label="test Accuracy",
    )
    axs[1].xlim = (0, num_epochs + 1)
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].set_title(f"Accuracy on {model_name}")
    axs[1].legend(loc="best")
    axs[1].grid(True)

    fig.savefig(f"{model_name}_training_performance.svg", format="svg")
    fig.show()

    plt.figure()
    plt.plot(loss_list, label="Minibatch Loss")
    plt.plot(running_avg_loss, label="Running Average Loss", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Training Loss on {model_name}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.yscale("log")

    plt.savefig(f"{model_name}_training_loss.svg", format="svg")
    plt.show()


def plot_images_with_predictions(model, data_loader, model_name, device):
    # step1: get 10 sample images from the data loader
    images, labels = next(iter(data_loader))
    images, labels = images[:10], labels[:10]

    images = images.to(device)
    labels = labels.to(device)

    # step2: get model predictions and calculate accuracy
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    correct_count = (predicted == labels).sum().item()
    accuracy = correct_count / len(labels) * 100

    # step3: plot the images with the predicted labels
    images = images.cpu()
    labels = labels.cpu()
    predicted = predicted.cpu()

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(
        f"10 MNIST samples on Test Dataset using {model_name}\nAccuracy: {accuracy:.2f}%",
        fontsize=16,
        fontweight=600,
    )

    for i in range(10):
        ax = axes[i // 5, i % 5]
        img = images[i].numpy().squeeze()  # 使用 squeeze 去除单维度

        ax.imshow(img, cmap='gray')  # 使用灰度色彩映射
        ax.axis("off")

        color = "blue" if predicted[i] == labels[i] else "red"
        ax.set_title(
            f"True: {labels[i]}\nPred: {predicted[i]}",
            fontsize=12,
            color=color,
            y=-0.25,
        )

    plt.savefig(f"{model_name}_mnist_predictions.svg", format="svg")
    plt.show()


def plot_compare(log_dicts):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    color_list = ["#6495ED", "#4EEE94", "#EEC900", "#FF6347", "#BA55D3", "#00808C"]
    ind_color = 0
    for log_dict in log_dicts:
        test_acc = log_dict["test_acc_per_epoch"]
        model_name = log_dict["model_name"]
        test_loss_per_epoch = log_dict["test_loss_per_epoch"]

        axs[0].plot(
            np.arange(1, len(test_loss_per_epoch) + 1),
            test_loss_per_epoch,
            ".--",
            color=color_list[ind_color],
            label=f"{model_name}",
        )

        axs[1].plot(
            np.arange(1, len(test_acc) + 1),
            test_acc,
            ".--",
            color=color_list[ind_color],
            label=f"{model_name}",
        )
        ind_color += 1

    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("test Loss")
    axs[0].set_title(f"MNIST test Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].set_title(f"MNIST test Accuracy")
    axs[0].legend(loc="best")
    axs[1].legend(loc="best")
    axs[0].grid(True)
    axs[1].grid(True)
    fig.savefig("MNIST_training_performance.svg", format="svg")
    fig.show()
