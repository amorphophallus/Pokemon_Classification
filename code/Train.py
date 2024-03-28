from torch.utils.tensorboard import SummaryWriter
import torch
import pandas as pd
import os
from torch.utils.data import DataLoader, ConcatDataset
from DataPreprocess import CustomDataset
from ResNet18 import ResNet18, Lenet5


EPOCHS = 10
BATCH_SIZE = 32
eps = 1e-3  # 如果判断某个属性的可能性小于 eps 就认为没有这个属性
TEST_LEAK = False  # 是否存在测试集泄露
LOAD_MODEL = True  # 是否加载之前训练过的模型
MODEL_DIR = r'./models/'
MODEL_PATH = os.path.join(MODEL_DIR, 'ResNet18_PokemonType.pth')


def get_result(x):
    """
    用于把模型输出的概率变成唯一答案
    输入 x 是一个 (BATCH_SIZE, CLASS_SIZE) 的张量
    对他每个 batch 求一个 argmax，得到一个 (BATCH_SIZE, 2) 的张量 ret
    """
    _, idx = torch.sort(x, descending=True)
    for i in range(x.shape[0]):
        if _[i, 1] < eps:
            idx[i, 1] = -1
    return idx[:, :2]


def compare_result(x, y):
    """
    输入 x, y 是 2 个 (BATCH_SIZE, 2) 的张量
    默认 x 是模型输出的数据，y 是标签
    输出一个数字表示有多少判断是正确的
    """
    cnt = 0
    for i in range(y.shape[0]):
        if y[i, 1] == -1:
            if x[i, 0] == y[i, 0] or x[i, 1] == y[i, 0]:
                cnt = cnt + 1
        else:
            if x[i, 0] == y[i, 0] and x[i, 1] == y[i, 1] or x[i, 0] == y[i, 1] and x[i, 1] == y[i, 0]:
                cnt = cnt + 1
    return cnt


if __name__ == '__main__':
    """
    准备数据集
    """
    origin_df = pd.read_csv('pokemon/pokemon.csv')  # load original data
    train_df = pd.read_csv('pokemon/train.csv').merge(origin_df, on="Name")
    test_df = pd.read_csv('pokemon/test.csv').merge(origin_df, on="Name")
    dataset_train = CustomDataset('pokemon/images', train_df)
    dataset_test = CustomDataset('pokemon/images', test_df)
    if TEST_LEAK:  # 如果有测试集泄露的话就把两个数据集合并起来就行了
        dataset_train = ConcatDataset([dataset_train, dataset_test])
    dataLoader_train = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    dataLoader_test = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=False)
    num_classes = dataset_test.get_num_classes()
    device = torch.device("cuda" if torch.cuda.is_available() else "")
    print("Number of classes: ", num_classes)
    print("Size of train dataset: ", len(dataset_train))
    print("Size of test dataset: ", len(dataset_test))
    print("device: ", device)

    """
    模型、损失函数、优化器实例化
    """
    model = ResNet18(in_channel=3, num_classes=num_classes).to(device)
    if os.path.exists(MODEL_PATH) and LOAD_MODEL:  # 在上次训练的基础上继续训练
        model.load_state_dict(torch.load(MODEL_PATH))
        print("Load model Successfully!")
    # model = Lenet5(in_channel=3, num_classes=num_classes).to(device)
    criterion = torch.nn.MultiLabelSoftMarginLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    writer = SummaryWriter()

    """
    开始训练
    """
    model.train()
    for epoch in range(EPOCHS):
        train_loss, train_acc = 0.0, 0.0
        batch_cnt = 0
        for images, labels in dataLoader_train:
            # 1. Get Training Data
            images = images.to(device)  # 注意张量在 cpu 和 gpu 之间切换不是 in-place 操作，所以还是需要赋值的
            labels = labels.to(device)

            # 2. Forward pass: Compute predicted y by passing x to the model
            y_pred = model(images)

            # 3. Compute and print loss
            loss = criterion(y_pred, labels)

            # 4. Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 5. accumulate accuracy and loss
            train_loss += loss.item()
            y_pred_res = get_result(y_pred)
            labels_res = get_result(labels)
            train_acc += compare_result(y_pred_res, labels_res)

            batch_cnt = batch_cnt+1
            print("finish " + str(batch_cnt) + " " + str(train_acc))

        # compute average loss and accuracy and draw on tensorboard
        train_loss /= len(dataLoader_train)
        train_acc /= len(dataLoader_train) * BATCH_SIZE
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        print(f"Epoch: {epoch}| Train loss: {train_loss: .5f}| Train acc: {train_acc: .5f}")
    writer.flush()

    """
    保存模型
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    torch.save(obj=model.state_dict(), f=MODEL_PATH)
    print("Save model Successfully!")

    """
    开始测试
    """
    test_loss, test_acc = 0.0, 0.0
    with torch.no_grad():
        for images, labels in dataLoader_test:
            images = images.to(device)
            labels = labels.to(device)
            y_pred = model(images)
            loss = criterion(y_pred, labels)
            test_loss += loss.item()
            y_pred_res = get_result(y_pred)
            labels_res = get_result(labels)
            test_acc += compare_result(y_pred_res, labels_res)
    test_loss /= len(dataLoader_test)
    test_acc /= len(dataLoader_test) * BATCH_SIZE
    print('test_loss: {:.4f}, test_acc: {:.4f}'.format(test_loss, test_acc))
