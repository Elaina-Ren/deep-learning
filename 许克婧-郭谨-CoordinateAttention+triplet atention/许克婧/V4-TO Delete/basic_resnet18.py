import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


batch_size =8
# lr=0.002:还行 效果也不错
lr=0.001
momentum=0.9
epochs = 10#可调整--这里可以变成train函数


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)


#1.load data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


#这是已经划分好训练和测试数据集了的
trainset = torchvision.datasets.CIFAR100(root='/content/drive/MyDrive/DeepLearnning/FinalPro/cifar-100-python', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='/content/drive/MyDrive/DeepLearnning/FinalPro/cifar-100-python', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = {19: 'cattle', 29: 'dinosaur', 0: 'apple', 11: 'boy', 1: 'aquarium_fish', 86: 'telephone',
           90: 'train', 28: 'cup', 23: 'cloud', 31: 'elephant', 39: 'keyboard', 96: 'willow_tree', 
           82: 'sunflower', 17: 'castle', 71: 'sea', 8: 'bicycle', 97: 'wolf', 80: 'squirrel', 
           74: 'shrew', 59: 'pine_tree', 70: 'rose', 87: 'television', 84: 'table', 64: 'possum', 
           52: 'oak_tree', 42: 'leopard', 47: 'maple_tree', 65: 'rabbit', 21: 'chimpanzee', 
           22: 'clock', 81: 'streetcar', 24: 'cockroach', 78: 'snake', 45: 'lobster', 49: 'mountain', 
           56: 'palm_tree', 76: 'skyscraper', 89: 'tractor', 73: 'shark', 14: 'butterfly', 9: 'bottle', 
           6: 'bee', 20: 'chair', 98: 'woman', 36: 'hamster', 55: 'otter', 72: 'seal', 43: 'lion', 51: 'mushroom', 
           35: 'girl', 83: 'sweet_pepper', 33: 'forest', 27: 'crocodile', 53: 'orange', 92: 'tulip', 50: 'mouse', 
           15: 'camel', 18: 'caterpillar', 46: 'man', 75: 'skunk', 38: 'kangaroo', 66: 'raccoon', 77: 'snail', 
           69: 'rocket', 95: 'whale', 99: 'worm', 93: 'turtle', 4: 'beaver', 61: 'plate', 94: 'wardrobe', 68: 'road', 
           34: 'fox', 32: 'flatfish', 88: 'tiger', 67: 'ray', 30: 'dolphin', 62: 'poppy', 63: 'porcupine', 40: 'lamp', 
           26: 'crab', 48: 'motorcycle', 79: 'spider', 85: 'tank', 54: 'orchid', 44: 'lizard', 7: 'beetle', 12: 'bridge', 
           2: 'baby', 41: 'lawn_mower', 37: 'house', 13: 'bus', 25: 'couch', 10: 'bowl', 57: 'pear', 5: 'bed', 
           60: 'plain', 91: 'trout', 3: 'bear', 58: 'pickup_truck', 16: 'can'}


#残差块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channel != self.expansion * out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, self.expansion * out_channel,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channel)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class Net(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(Net, self).__init__()
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block, out_channel, num_blocks, stride):
        strides = [stride] + (num_blocks - 1) * [1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, out_channel, stride))
            self.in_channel = out_channel * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet():
    return Net(BasicBlock, [2, 2, 2,2])#4个残差块


net = ResNet().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

def train(epoch):
  net.train()
  running_loss = 0.0
  for i, data in enumerate(trainloader, 0):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data
      inputs = inputs.to(device)
      labels = labels.to(device)

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.item()
      if i % 2000 == 1999:    # print every 2000 mini-batches
          print(f'[{epoch+1}, {i+1:5d}] loss: {running_loss / 2000:.3f}')
          running_loss = 0.0


def validate(loss_vector, accuracy_vector):
    net.eval()
    total = 0
    val_loss, correct = 0, 0
    for data, target in testloader:
        data = data.to(device)
        target = target.to(device)
        output = net(data)
        val_loss += criterion(output, target).data.item()
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == target).sum().item()
        total += target.size(0)#total就是一个 总数...

    val_loss /= total
    loss_vector.append(val_loss)

    accuracy = 100 * correct / total
    accuracy_vector.append(accuracy)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, total, accuracy))
    
#开始训练 
lossv, accv = [], []

#train
for epoch in range(epochs):  # loop over the dataset multiple times
  train(epoch)
  validate(lossv,accv)

print("training finished！")


plt.figure(figsize=(5,3))
plt.plot(np.arange(1,epochs+1), lossv)
plt.title('validation loss,epoch=%s'%epochs)

plt.figure(figsize=(5,3))
plt.plot(np.arange(1,epochs+1), accv)
plt.title('validation accuracy,epoch=%s'%epochs)