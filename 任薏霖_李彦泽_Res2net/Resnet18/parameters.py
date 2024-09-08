# -*- coding:utf-8 -*-
# @Time : 2023-05-15 20:25
# @Author : Renyilin + Liyanze
# @File : Resnet18
# @software: PyCharm


# 训练的次数
epoch = 50

# 训练的批次大小
batch_size = 1024

# 数据集的分类类别数量
CIFAR100_class = 100

# 模型训练时候的学习率大小
resnet_lr = 0.002

# 保存模型权重的路径 保存xml文件的路径
resnet_save_path_CIFAR100 = './res/'
resnet_save_model = './res/best_model.pth'
