import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math

batch_size =24
lr=0.01
momentum=0.9
epochs =30



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
trainset = torchvision.datasets.CIFAR100(root='./data/CIFAR100', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data/CIFAR100', train=False,
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


import torch
import torch.nn as nn
import math
import torch.nn.functional as F



class GCLayer(nn.Module):
    def __init__(self, in_channels, scale, sr):
        super(GCLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels//scale

        self.sr=sr

        self.conv1 = nn.Conv2d(self.in_channels, 1, 1)
        self.sm = nn.Softmax(dim=1)

        self.transform = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1),
            nn.LayerNorm([self.out_channels, 1, 1]),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.in_channels, 1),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        out=self.conv1(x)
        out=out.view(b,1,-1)#[b, 1, H, W]
        out=out.permute(0,2,1)#[b, 1, H*W]
        out=out.view(b,-1,1)#[b, H*W, 1]
        out=out.contiguous()
        out=self.sm(out)
        x2 = x.view(b, c, h*w)# [b, c, h*w]
        out = torch.matmul(x2, out)
        out = out.view(b, c, 1, 1).contiguous()
        out = self.transform(out)
        return x+self.sr*out, out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.shortcut = nn.Sequential()
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
        # residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # if self.downsample is not None:
        #     residual = self.downsample(x)

        # out += residual
        #加上CA模块 v1
        # out=self.ca(out)
        out+=self.shortcut(x)
        out = self.relu(out)

        return out


class Net(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100,use_gcblock=True):
        super(Net, self).__init__()
        self.use_gcblock=use_gcblock

        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)

        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.gcblock1=GCLayer(256,16,0.1)
        self.gcblock2=GCLayer(512,16,0.2)
        self.gcblock3=GCLayer(1024,16,0.3)
        self.gcblock4=GCLayer(2048,16,0.4)

        self.fc10=nn.Linear(256,256)
        self.fc11=nn.Linear(256,512)
        self.bn1=nn.BatchNorm2d(512)

        self.fc20=nn.Linear(512,512)
        self.fc21=nn.Linear(512,1024)
        self.bn2=nn.BatchNorm2d(1024)

        self.fc30=nn.Linear(1024,1024)
        self.fc31=nn.Linear(1024,2048)
        self.bn3=nn.BatchNorm2d(2048)

        self.relu = nn.ReLU(inplace=True)

        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block, out_channel, num_blocks, stride):
        strides = [stride] + (num_blocks - 1) * [1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, out_channel, stride))
            self.in_channel = out_channel * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn(self.conv1(x)))
        out = self.layer1(out)
        if(self.use_gcblock):
            out, gcparams1 = self.gcblock1(out)
            b,c,h,w=gcparams1.shape[:]
            gcparams1=gcparams1.view(b, c)
        out = self.layer2(out)
        if(self.use_gcblock):
            out, gcparams2 = self.gcblock2(out)
            b,c,h,w=gcparams2.shape[:]
            gcparams2=gcparams2.view(b, c)
            #===============================================
            gcparamsLinear12=self.fc10(gcparams1)
            gcparamsLinear12=self.fc11(gcparamsLinear12)
            
            gcparamsLinear12=self.relu(gcparamsLinear12)
            gcparamsLinear12=gcparamsLinear12.view(b,c,h,w)
            gcparamsLinear12=self.bn1(gcparamsLinear12)
            out+=0.3*gcparamsLinear12
            #===============================================
        out = self.layer3(out)
        if(self.use_gcblock):
            out, gcparams3 = self.gcblock3(out)
            b,c,h,w=gcparams3.shape[:]
            gcparams3=gcparams3.view(b, c)
            #===============================================
            gcparamsLinear23=self.fc20(gcparams2)
            gcparamsLinear23=self.fc21(gcparamsLinear23)
            gcparamsLinear23=self.relu(gcparamsLinear23)
            gcparamsLinear23=gcparamsLinear23.view(b,c,h,w)
            gcparamsLinear23=self.bn2(gcparamsLinear23)
            out+=0.3*gcparamsLinear23
            #===============================================
        out = self.layer4(out)
        if(self.use_gcblock):
            out, gcparams4 = self.gcblock4(out)
            b,c,h,w=gcparams4.shape[:]
            gcparams4=gcparams4.view(b, c)
            #===============================================
            gcparamsLinear34=self.fc30(gcparams3)
            gcparamsLinear34=self.fc31(gcparamsLinear34)
            
            gcparamsLinear34=self.relu(gcparamsLinear34)
            gcparamsLinear34=gcparamsLinear34.view(b,c,h,w)
            gcparamsLinear34=self.bn3(gcparamsLinear34)

            out+=0.3*gcparamsLinear34
            #===============================================
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
with open("./results/res18bottle-accu.txt", "w") as f:
  for epoch in range(epochs):
    #训练
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
        if i % 3000 == 2999:    # print every 2000 mini-batches
            print(f'[{epoch+1}, {i+1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
            f.write('[%03d  %05d] |Loss: %.03f  '
              % (epoch + 1, i + 1, running_loss / 2000))
            f.write('\n')
            f.flush()

        
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

      print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
          val_loss, correct, total, accuracy))
      f.write('\nValidation set: Average loss:%4f,Accuracy:%d/%d = %2f \n'
      %(val_loss, correct, total, accuracy))  
      f.flush()          


print("training finished！")
plt.figure(figsize=(5,3))
plt.plot(np.arange(1,epochs+1), loss_vector)
plt.title('validation loss,epoch=%s'%epochs)

plt.figure(figsize=(5,3))
plt.plot(np.arange(1,epochs+1), accuracy_vector)
plt.title('validation accuracy,epoch=%s'%epochs)



