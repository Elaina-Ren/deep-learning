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

#先用实验2的参数跑跑......
batch_size =8
# lr=0.002:还行 效果也不错
lr=0.001
momentum=0.9
epochs =20
print_every=2000

DATASET_PATH='/kaggle/working/input'
RESULT_FOLDER='/kaggle/working/Results'
if not os.path.exists(RESULT_FOLDER):
    # 使用os.makedirs()函数创建文件夹
    os.makedirs(RESULT_FOLDER)
resFilename=RESULT_FOLDER+'res18.txt'


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
           26: 'crab', 48: 'motorcycle', 79: 'spider', 85: 'tank', 54: 'orchid', 44: 'lizard', 7: 'beetle', 12: 'bridge', 
           2: 'baby', 41: 'lawn_mower', 37: 'house', 13: 'bus', 25: 'couch', 10: 'bowl', 57: 'pear', 5: 'bed', 
           60: 'plain', 91: 'trout', 3: 'bear', 58: 'pickup_truck', 16: 'can'}


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1 / 3) * (x_out + x_out11 + x_out21)
        else:
            x_out = (1 / 2) * (x_out11 + x_out21)
        return x_out


class Bottleneck(nn.Module):
    expansion = 4 #通道数变为原来的4倍

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        #1*1卷积
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        #3*3卷积 用于特征提取
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        
       
        self.bn2 = nn.BatchNorm2d(planes)

        #1*1卷积 通道数增加
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        #引入TA
        self.triplet_attention = TripletAttention(inplanes)


        # self.downsample = downsample
        self.stride = stride
        self.shortcut = nn.Sequential()
        # #CA初始化 V1
        # self.ca = CoordAtt(planes * self.expansion, planes * self.expansion)
      
        


        if stride != 1 or inplanes != planes * Bottleneck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes,
                          planes * Bottleneck.expansion,
                          stride=stride,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion))

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)#conv2是特征提取层 将CA加在这里蛮合适
        out = self.bn2(out)
        out = self.relu(out)
        

        out = self.conv3(out)
        out = self.bn3(out)


        #加入TA
        out=self.triplet_attention(out)
        out+=self.shortcut(x)#前面层的特征直连
        out = self.relu(out)

        return out


class Net(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(Net, self).__init__()
        self.in_channel = 64

        #输入图像的通道数为3（即RGB图像），输出通道数为64
        #（不同卷积核权重不同，会提取到不同的信息）
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
    return Net(Bottleneck, [2, 2, 2,2])#resNet18：用瓶颈网


net = ResNet().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)  

#开始训练 
loss_vector, accuracy_vector = [], []
train_loss_vector=[]
train_accuracy_vector=[]
with open(resFilename, "w") as f:
  for epoch in range(epochs):
    #训练
    net.train()
    running_loss = 0.0
    total=0
    all=0
    running_loss = 0.0
    train_loss=0.0
    train_accuracy=0.0
    correct=0
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
        train_loss+=loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        all+=labels.size(0)
        correct += predicted.eq(labels.data).sum()
        train_accuracy+=predicted.eq(labels.data).cpu().sum()
        if i % print_every == print_every-1:    # print every 2000 mini-batches
            print(f'[{epoch+1}, {i+1:5d}] loss: {running_loss / 2000:.3f},Train accuracy:{100*correct/total:.2f}')
            f.write('[%03d  %05d] |Loss: %.03f |Train accuracy: %.02f  '
              % (epoch + 1, i + 1, running_loss / 2000,100*correct/total))
            f.write('\n')
            f.flush()
            total=0;
            running_loss = 0.0
            correct=0
    train_loss/=all
    train_accuracy=100*train_accuracy/all
    train_loss_vector.append(train_loss)
    train_accuracy_vector.append(train_accuracy)
        
    # 每训练完一个epoch测试一下准确率
    #测试不需要反向计算梯度：提高效率
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
          total += target.size(0)#total就是一个 总数...

      val_loss /= total
      loss_vector.append(val_loss)

      accuracy = 100 * correct / total
      accuracy_vector.append(accuracy)

      print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
          val_loss, correct, total, accuracy))
      f.write('\nValidation set: Average loss:%4f,Accuracy:%d/%d = %2f \n'
      %(val_loss, correct, total, accuracy))  
      f.flush()          


print("training finished！")
plt.figure(figsize=(5,3))
plt.plot(np.arange(1,epochs+1), loss_vector,label='Vali_Loss')
plt.plot(np.arange(1, epochs + 1), train_loss_vector, label='Train_Loss')
plt.title('loss,epoch=%s'%epochs)
# 显示图例
plt.legend()
plt.show()

plt.figure(figsize=(5,3))
plt.plot(np.arange(1,epochs+1), accuracy_vector,label='Vali_accura')
plt.plot(np.arange(1,epochs+1), train_accuracy_vector,label='train_accura')
plt.title('accuracy,epoch=%s'%epochs)
plt.legend()
plt.show()
