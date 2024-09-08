# DeepLearningDemo

#### 介绍
深度学习大作业

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request


#### 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)


#### 研究对象

在这个项目中，我们选择了四个常用的神经网络模型进行研究：ResNet18、Res2Net、Triplet Attention和Coordinate Attention。

# 项目介绍

这是一个关于计算机视觉领域的研究项目，旨在探讨如何提高神经网络在图像分类任务中的性能。

## 研究对象

在这个项目中，我们选择了四个常用的神经网络模型进行研究：ResNet18、Res2Net、Triplet Attention和Coordinate Attention。这些模型都是目前比较流行的深度学习模型，被广泛应用于图像分类、目标检测等领域。

## 实验设计

我们将这四个模型应用到CIFAR-100数据集上进行实验，比较它们在不同超参数组合下的性能表现。具体来说，我们采用了SGD优化器，学习率为0.001，动量为0.9，并在不同批量大小下进行训练。我们还尝试了一些改进方法，如在ResNet18基础上加入GC Block和在Res2Net基础上加入Attention机制，以提高模型性能。

## 结果分析

通过对这四个模型的研究和实验，得出了以下结论，包括：

- GC Block是一种有效的模型改进方法；
- Attention机制的效果不如预期；
- 批量大小和学习率的组合对模型性能有很大影响。

在未来的工作中，我们可以进一步探究其他模型和改进方法，以提高神经网络在图像分类任务中的性能。

#### 结论

通过对这些模型的分析和比较，我们得出了以下几个结论：

- 在ResNet18基础上增加全局上下文信息能够提高模型性能；
- 在Res2Net中增加多个尺度特征能够提高模型性能；
- 在Triplet Attention中加入通道注意力能够提高模型性能；
- 在Coordinate Attention中加入全局上下文信息能够提高模型性能。

#### 改进方法

基于以上结论，我们提出了两种改进网络的方法：GC Net和GC Net plus。这两种方法都能够显著提高模型性能，但是GC Net plus相对于GC Net来说更加复杂，需要更多的计算资源。在未来的研究中，我们将继续探索更复杂的网络结构和更具创新性的网络架构，以期在图像分类任务中取得更好的性能。

