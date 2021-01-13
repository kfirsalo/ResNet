from gcommand_loader import GCommandLoader
import torch
import numpy as np
from torchvision import transforms
from gcommand_loader import find_classes

np.random.seed(0)
dataset_train = GCommandLoader('C:/Users/kfirs/PycharmProjects/ex_5/train')
dataset_validation = GCommandLoader('C:/Users/kfirs/PycharmProjects/ex_5/valid')
dataset_test = GCommandLoader('C:/Users/kfirs/PycharmProjects/ex_5/test')
print(find_classes('C:/Users/kfirs/PycharmProjects/ex_5/train')[0])
# transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = torch.utils.data.DataLoader(
         dataset_train, batch_size=100, shuffle=True, pin_memory=True, sampler=None)
val_loader = torch.utils.data.DataLoader(
         dataset_validation, batch_size=100, shuffle=True, pin_memory=True, sampler=None)
test_loader = torch.utils.data.DataLoader(
         dataset_test, batch_size=100, shuffle=True, pin_memory=True, sampler=None)


for k, (input, label) in enumerate(test_loader):
    print(input ,label[0] ,k)


print(len(train_loader))

