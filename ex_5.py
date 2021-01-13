import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import datasets
import numpy as np
import matplotlib as mpl
from tqdm import tqdm
from data_loader import train_loader, val_loader, test_loader
from gcommand_loader import find_classes
import os

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                    stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=30):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0], 1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=3)
        self.layer2 = self.make_layer(block, 32, layers[0], 1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = self.make_layer(block, 64, layers[1], 1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        # self.conv_bn = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(13 * 8 * 64, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, blocks, stride):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.layer1(out)
        out = self.pool1(out)
        out = self.layer2(out)
        out = self.pool2(out)
        out = self.layer3(out)
        out = self.pool3(out)
        # out = self.conv_bn(out)
        out = out.view(out.size(0), -1)
        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        return F.log_softmax(out)



net_args = {
    "block": ResidualBlock,
    "layers": [2, 2, 2, 2]
}
cnn = ResNet(**net_args)




cpu_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(cpu_gpu)
total_step = len(train_loader)


def final_train(model,epoch, num_epochs):
    loss_list = []
    acc_list = []
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.999, 0.9), eps=1e-08, weight_decay=0.001, amsgrad=True)
    start = int(np.sum((np.random.random(2))))
    if start == 0:
        for i, (images, labels) in enumerate(train_loader):
            # Run the forward pass
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = float(labels.size(0))
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(100. * (correct / total))

            if (i + 1) % 20 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100.))
        for i, (images, labels) in enumerate(val_loader):
            print(i)
            # Run the forward pass
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = float(labels.size(0))
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(100. * (correct / total))

            if (i + 1) % 20 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100.))
    else:
        for i, (images, labels) in enumerate(val_loader):
            print(i)
            # Run the forward pass
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = float(labels.size(0))
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(100. * (correct / total))

            if (i + 1) % 20 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100.))
        for i, (images, labels) in enumerate(train_loader):
            print(i)
            # Run the forward pass
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = float(labels.size(0))
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(100. * (correct / total))

            if (i + 1) % 20 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100.))

def train(model,epoch, num_epochs,device):
    model.to(device)
    model.train()
    loss_list = []
    acc_list = []
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.0004, betas=(0.999, 0.999), eps=1e-08,
                           weight_decay=0, amsgrad=True)
    # optimizer = optim.Adam(model.parameters(), lr=0.0004, betas=(0.999, 0.9), eps=1e-08, weight_decay=0, amsgrad=True)
    for i, (images, labels) in enumerate(train_loader):
        # Run the forward pass
        images, labels = images.to(device), labels.to(device)
        outputs = cnn(images)
        loss = F.nll_loss(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = float(labels.size(0))
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(100. * (correct / total))

        if (i + 1) % 20 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100.))
    acc_list=np.array(acc_list)
    loss_list = np.array(loss_list)
    return np.mean(acc_list), np.mean(loss_list)

def validation(model, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        loss_val = 0
        correct_val = 0
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_val += F.nll_loss(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct_val += pred.eq(target.view_as(pred)).cpu().sum()
        loss_val /= int(len(val_loader))
        acc_val = 100. * float(correct_val) / int(len(val_loader.dataset))
        print('Validation set: Average loss: {:.4f}, Accuracy: {}/{}'
              ' ({:.0f}%)\n'.format(loss_val, correct_val, int(len(val_loader.dataset)), int(acc_val)))
    return acc_val, loss_val

def test(model, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        pre_test = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, sample_pre = torch.max(output.data, 1)  # get the index of the max log-probability
            for sample in sample_pre:
                new_sample = int(sample)
                pre_test.append(new_sample)
    return pre_test


def graph_maker(num_epochs, device):
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    for epoch in range(num_epochs):
        acc_train, loss_train = train(cnn, epoch, num_epochs,device)
        train_acc_list.append(acc_train)
        train_loss_list.append(loss_train)
        acc_val, loss_val = validation(cnn, device)
        val_acc_list.append(acc_val)
        val_loss_list.append(loss_val)
    pre_test = test(cnn, device = cpu_gpu)
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 20
    mpl.rcParams['axes.labelsize'] = 16
    epoch_exis = range(1, 25)
    plt.figure(1)
    plt.title('Loss Per Epoch, ResNet Model')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(epoch_exis, train_loss_list, '-ok', color='red', label='train')
    plt.plot(epoch_exis, val_loss_list, '-ok', color='blue', label='validation')
    plt.legend()
    plt.show()
    plt.figure(2)
    plt.title('Accuracy Per Epoch, ResNet Model')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(epoch_exis, train_acc_list, '-ok', color='red', label='train')
    plt.plot(epoch_exis, val_acc_list, '-ok', color='blue', label='validation')
    plt.legend()
    plt.show()
    print(train_loss_list)
    print(train_acc_list)
    print(val_loss_list)
    print(val_acc_list)
    return pre_test


if __name__ == "__main__":
    np.random.seed(0)
    pre_test = graph_maker(num_epochs=24, device=cpu_gpu)
    file = open('test_y', 'w')
    paths = [os.path.basename(i[0]) for i in test_loader.dataset.spects]
    outputs = []
    for path in paths:
        outputs.append((path))
    for i in range(len(pre_test)-1):
        file.write(f"{paths[i]},{find_classes('train')[0][pre_test[i]]}\n")
    file.write(f"{paths[len(pre_test)-1]},{find_classes('train')[0][pre_test[len(pre_test)-1]]}")
