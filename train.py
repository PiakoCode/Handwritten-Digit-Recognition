import datetime
import torch
from torch import nn
import numpy as np
from torch import optim
import torch.nn.functional as F
from data import train_dataloader, test_dataloader
import time

from model import ResNet, ResNet_with_Dropout, ResNet_with_Smaller
from utils import plot_compare, plot_images_with_predictions, plot_training_metrics

torch.manual_seed(89)

lr = 0.01
epoch = 10
num_classes = 10

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("now using device: ", device)

start_time = datetime.datetime.now().strftime("%m月%d日%H:%M")

image_size = (1, 28, 28)


def eval_model(model, data_loader):
    model.eval()
    class_total = [0.0 for _ in range(num_classes)]
    class_correct = [0.0 for _ in range(num_classes)]
    sum_loss, num_correct, num_examples = 0.0, 0.0, 0

    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            # compute the model output
            outputs = model(features)
            _, predicted_labels = torch.max(outputs, 1)
            # compute the loss
            loss = F.cross_entropy(outputs, targets, reduction="sum")
            sum_loss += loss.item()

            # compute the correct radix
            num_examples += targets.size(0)
            num_correct += (predicted_labels == targets).sum().item()

            # compute each class 's correct count
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_correct[label] += (predicted_labels[i] == label).item()
                class_total[label] += 1

    accuracy = num_correct / num_examples * 100
    avg_loss = sum_loss / num_examples

    return {
        "accuracy": accuracy,
        "avg_loss": avg_loss,
        "class_correct": class_correct,
        "class_total": class_total,
    }


# train model
def train(
    model,
    train_loader,
    test_loader,
    num_epochs=25,
    model_name="model",
    optimizer=None,
    loss_fn=None,
    scheduler=None,
) -> dict:
    log_dict = {
        "train_loss_per_batch": [],
        "train_acc_per_epoch": [],
        "test_acc_per_epoch": [],
        "train_loss_per_epoch": [],
        "test_loss_per_epoch": [],
        "test_loss_min": np.Inf,
        "learning_rates": [],
        "model_name": model_name,
    }
    start_time = time.time()
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        log_dict["learning_rates"].append(current_lr)
        print(
            f"Epoch: {epoch+1:03d}/{num_epochs:03d} | Current Learning Rate: {current_lr:.6f}"
        )
        ###################
        # 训练 #
        ###################
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader, 0):
            features = features.to(device)
            targets = targets.to(device)

            # step1: predict the output
            outputs = model(features)

            # step2: loss backpropagation
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()

            # step3: update model parameters
            optimizer.step()

            log_dict["train_loss_per_batch"].append(loss.item())
            if not batch_idx % 50:
                print(
                    f"Epoch: {epoch+1:03d}/{num_epochs:03d} | Batch {batch_idx:04d}/{len(train_loader):04d} | Loss: {loss:.4f}"
                )

        #! each epoch, evaluate the model
        ######################
        # 验证 #
        ######################
        model.eval()
        with torch.set_grad_enabled(False):
            train_eval_res = eval_model(model, train_loader)
            train_acc = train_eval_res["accuracy"]
            train_loss = train_eval_res["avg_loss"]
            print(
                f"**Epoch: {epoch+1:03d}/{num_epochs:03d} | Train. Acc.: {train_acc:.3f}% | Loss: {train_loss:.4f}"
            )
            log_dict["train_loss_per_epoch"].append(train_loss)
            log_dict["train_acc_per_epoch"].append(train_acc)

            # * each epoch, evaluate the model on the test dataset which is not used for training
            test_eval_res = eval_model(model, test_loader)
            test_acc = test_eval_res["accuracy"]
            test_loss = test_eval_res["avg_loss"]
            log_dict["test_loss_per_epoch"].append(test_loss)
            log_dict["test_acc_per_epoch"].append(test_acc)
            print(
                f"**Epoch: {epoch+1:03d}/{num_epochs:03d} | Test. Acc.: {test_acc:.3f}% | Loss: {test_loss:.4f}"
            )
            # * save the model if the test loss is decreased
            if test_loss <= log_dict["test_loss_min"]:
                print(
                    f"**Test loss decreased ({log_dict['test_loss_min']:.6f} --> {test_loss:.6f}). Saving model ..."
                )
                torch.save(model.state_dict(), f"{model_name}_mnist.pt")
                log_dict["test_loss_min"] = test_loss

        if scheduler is not None:
            # scheduler.step()
            scheduler.step(test_loss)
        print(f"Time elapsed: {(time.time() - start_time) / 60:.2f} min")

    print(f"Total Training Time: {(time.time() - start_time)/ 60:.2f} min")
    return log_dict


def train_all_resnet_models():
    resnet_models = {
        "ResNet18": (
            ResNet().to(device),
            20,
            0.01,
        ),
        "ResNet18_With_Dropout": (ResNet_with_Dropout().to(device), 20, 0.01),
        "ResNet_with_Smaller_Conv": (
            ResNet_with_Smaller().to(device),
            20,
            0.01,
        ),
    }

    log_dicts = []

    for model_name, (model, num_epochs, initial_lr) in resnet_models.items():
        print(
            f"Training {model_name} for {num_epochs} epochs with initial learning rate {initial_lr}..."
        )
        # optimizer = optim.SGD(
        #     model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4
        # )
        # loss_fn = nn.CrossEntropyLoss()
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
        optimizer = optim.SGD(
            model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        loss_fn = nn.CrossEntropyLoss()
        log_dict = train(
            model,
            train_dataloader,
            test_dataloader,
            num_epochs=num_epochs,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            model_name=model_name,
        )
        log_dicts.append(log_dict)
        model.load_state_dict(torch.load(f"{model_name}_mnist.pt"))
        plot_training_metrics(log_dict, num_epochs)
        plot_images_with_predictions(model, test_dataloader, model_name, device)

    return log_dicts

if __name__ == "__main__":

    # 开始训练所有模型
    log_dicts = train_all_resnet_models()
    plot_compare(log_dicts)
