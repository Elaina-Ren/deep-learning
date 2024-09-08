"""
@Project ：base
@File ：data.py
@Author ：
@Date ：2023/5/11 14:27
"""
import torch
import torchvision
from torchvision import transforms

mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
num_workers = 2

# 直接返回trainloader,testloader导入模型
def cifar100_dataset(args):
    '''
    RandomCrop：以输入图的随机位置为中心做指定size的裁剪操作
    RandomHorizontalFlip：以0.5概率水平翻转给定的PIL图像
    RandomRotation(15)：随机进行图像的旋转，而且旋转的角度在【-15~15】度之间
    ToTensor()：将原始PILImage格式或者numpy.array格式的数据格式化为可被pytorch快速处理的张量类型
    Normalize()：逐channel的对图像进行标准化（均值变为0，标准差变为1）
    '''
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    cifar100_train = torchvision.datasets.CIFAR100(root = args.data_path, train = True, download = True, transform = transform_train)
    train_loader = torch.utils.data.DataLoader(cifar100_train, batch_size = args.batch_size, shuffle = True, num_workers = num_workers)

    cifar100_test = torchvision.datasets.CIFAR100(root = args.data_path, train = False, download = True, transform = transform_test)
    test_loader = torch.utils.data.DataLoader(cifar100_test, batch_size = 100, shuffle = False, num_workers = num_workers)

    return train_loader, test_loader, cifar100_train.classes
