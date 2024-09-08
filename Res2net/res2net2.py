import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math
import os
batch_size = 8
# lr=0.002:还行 效果也不错
lr = 0.001
momentum = 0.9
epochs = 10
print_every=2000

DATASET_PATH='/kaggle/working/input'
RESULT_FOLDER='/kaggle/working/Results'
if not os.path.exists(RESULT_FOLDER):
    # 使用os.makedirs()函数创建文件夹
    os.makedirs(RESULT_FOLDER)
resFilename=RESULT_FOLDER+'GC+CA-10.txt'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', device)
# 1.load data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 这是已经划分好训练和测试数据集了的
trainset = torchvision.datasets.CIFAR100(root=DATASET_PATH, train=True,
                                         download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root=DATASET_PATH, train=False,
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
           26: 'crab', 48: 'motorcycle', 79: 'spider', 85: 'tank', 54: 'orchid', 44: 'lizard', 7: 'beetle',
           12: 'bridge',
           2: 'baby', 41: 'lawn_mower', 37: 'house', 13: 'bus', 25: 'couch', 10: 'bowl', 57: 'pear', 5: 'bed',
           60: 'plain', 91: 'trout', 3: 'bear', 58: 'pickup_truck', 16: 'can'}







class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, scale=4):
        """ Constructor
               Args:
                   inplanes: input channel dimensionality
                   planes: output channel dimensionality
                   stride: conv stride. Replaces pooling layer.
                   downsample: None when stride = 1
                   baseWidth: basic width of conv3x3
                   scale: number of scale --- 控制每个分支的宽度
                   type: 'normal': normal set. 'stage': first block of a new stage.
               """
        super(Bottle2neck, self).__init__()
        # 每个分支的宽度
        width = int(planes * (scale / 4))
        #1*1的卷积核
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        #3*3的卷积核
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes * Bottle2neck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes,
                          planes * Bottle2neck.expansion,
                          stride=stride,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(planes * Bottle2neck.expansion))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class Res2Net(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super(Res2Net, self).__init__()
        self.in_channel = 64
        # 输入图像的通道数为3（即RGB图像），输出通道数为64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)#新增
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)#新增
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channel, num_blocks, stride):
        strides = [stride] + (num_blocks - 1) * [1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, out_channel, stride))
            self.in_channel = out_channel * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
#         x = self.relu(x)
        x=F.relu(x)
#         x = self.maxpool(x)#新增

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

#         x = self.avgpool(x)#改变
        x=F.avg_pool2d(x,4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

def res2net18():
    """Constructs a Res2Net-18_v1b model.
    Res2Net-18 refers to the Res2Net-50_v1b_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [2, 2, 2, 2])
    return model



net = res2net18().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

# 开始训练
loss_vector, accuracy_vector = [], []
train_loss_vector = []
train_accuracy_vector = []
with open(resFilename, "w") as f:
    for epoch in range(epochs):
        # 训练
        net.train()
        running_loss = 0.0
        total = 0
        all = 0
        running_loss = 0.0
        train_loss = 0.0
        train_accuracy = 0.0
        correct = 0
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
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            all += labels.size(0)
            correct += predicted.eq(labels.data).sum()
            train_accuracy += predicted.eq(labels.data).cpu().sum()
            if i % print_every == print_every-1:  # print every 2000 mini-batches
                print(
                    f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f},Train accuracy:{100 * correct / total:.2f}')
                f.write('[%03d  %05d] |Loss: %.03f |Train accuracy: %.02f  '
                        % (epoch + 1, i + 1, running_loss / 2000, 100 * correct / total))
                f.write('\n')
                f.flush()
                total = 0
                running_loss = 0.0
                correct = 0
        train_loss /= all
        train_accuracy = 100 * train_accuracy / all
        train_loss_vector.append(train_loss)
        train_accuracy_vector.append(train_accuracy)

        # 每训练完一个epoch测试一下准确率
        # 测试不需要反向计算梯度：提高效率
        with torch.no_grad():
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
                total += target.size(0)  # total就是一个 总数...

            val_loss /= total
            loss_vector.append(val_loss)

            accuracy = 100 * correct / total
            accuracy_vector.append(accuracy)

            print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                val_loss, correct, total, accuracy))
            f.write('\nValidation set: Average loss:%4f,Accuracy:%d/%d = %2f \n'
                    % (val_loss, correct, total, accuracy))
            f.flush()

print("training finished！")
plt.figure(figsize=(5, 3))
plt.plot(np.arange(1, epochs + 1), loss_vector, label='Vali_Loss')
plt.plot(np.arange(1, epochs + 1), train_loss_vector, label='Train_Loss')
plt.title('loss,epoch=%s' % epochs)
# 显示图例
plt.legend()
plt.show()

plt.figure(figsize=(5, 3))
plt.plot(np.arange(1, epochs + 1), accuracy_vector, label='Vali_accura')
plt.plot(np.arange(1, epochs + 1), train_accuracy_vector, label='train_accura')
plt.title('accuracy,epoch=%s' % epochs)
plt.legend()
plt.show()