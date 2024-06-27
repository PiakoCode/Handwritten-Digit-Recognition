from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

dataset_path = "../dataset"
batch_size = 128

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(28, padding=4),  # 随机裁剪，填充4个像素
        transforms.RandomRotation(15),  # 随机旋转
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

train_data = torchvision.datasets.MNIST(
    root=dataset_path,
    train=True,
    transform=transform_train,
    download=True,
)
test_data = torchvision.datasets.MNIST(
    root=dataset_path,
    train=False,
    transform=transform_test,
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


def check_dataset(loader, set_name):
    print(f"{set_name} Set:")
    images, labels = next(iter(loader))
    print("batch count", len(loader))
    print("image size per batch", images.size())
    print("label size per batch", labels.size())

if __name__ == "__main__":
    check_dataset(train_dataloader, "Training")
    check_dataset(test_dataloader, "Testing")
