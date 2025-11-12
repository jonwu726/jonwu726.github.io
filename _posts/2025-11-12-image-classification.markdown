---
layout:     post
title:      "图像分类经典论文复现 [01]"
subtitle:   " \"CNN开山之作LeNet-5\""
date:       2025-11-12
author:     "吴优"
header-img: "img/post-bg-2015.jpg"
tags:
    - Machine Learning
    - Computer Vision
    - Image Classification
---

> 以前为什么没去卷机器学习


## 目录

1. [前言](#前言)
2. [文献梳理](#文献梳理)
    1. [基本情况](#基本情况)
    2. [研究背景](#研究背景)
    3. [核心内容及创新](#核心内容及创新)
3. [复现工作](#复现工作)


## 前言

回头再看，上一篇只起了个头的博客，已经是23年的事情了，而我也早已离开游戏行业，令人感概。

24年不堪无尽的加班从网易裸辞后，进入了学校工作，回归躺平生活。今年又幸运地跳槽到另一所高校，离家的距离从5分钟变成了10分钟。

学校的事情也不是很多，但人不能闲着啊，所以开坑机器学习，目前研究的兴趣还是在计算机视觉这块。正所谓不积跬步无以至千里，目前不急着接触前沿的东西，先着手阅读和复现几篇经典论文，建立一下研究基础。

这篇文章所要探讨的，是卷积神经网络(CNN)的开山之作，2018年图灵奖得主LeCun的著名文章：Gradient-Based Learning Applied  to Document Recognition.

---

## 文献梳理

在正式复现前，先对文献做一个基本的概述。

### 基本情况

| 标题 | Gradient-Based Learning Applied  to Document Recognition |
| --- | --- |
| 期刊 | Proceedings of the IEEE |
| 分区 | 一区 |
| 时间 | 1998.11 |
| DOI | 10.1109/5.726791 |

该文献首次完整构建并成功训练了名为LeNet-5的卷积神经网络，用于手写数字的识别。其核心在于直接根据像素输入，通过卷积、池化层等自动提取特征，并采用反向传播进行端到端学习。

其在MNIST数据集上取得了极佳的识别效果，证明了CNN在真实世界模式识别任务中的巨大潜力。

### 研究背景

// TODO: 传统的图像分类做法

// TODO: 多层感知机

// TODO: 生物视觉的研究

### 核心内容及创新

// TODO: CNN架构

// TODO: 梯度下降

## 复现工作

我的机器配置是 ultra7-265k + 24g 6000c28 x 2 + 3080 12g，项目跑在 win11 系统下的 wsl 里，安装的是 ubuntu 22.04 版本，在前期也做了一定的调研，说是 Linux 下跑机器学习/深度学习无论是配环境的便捷程度还是性能上都更优，个人目前还没体会到这一点，可能因为我的 Linux 水平还比较低吧，哈哈。

整体工程基于 pytorch 实现，对于神经网络的训练，它几乎提供了保姆级的服务，不需要自己手搓任何轮子。而且直接提供了 MNIST 这个数据集类，可以很方便地拉取数据。

主要可以分为数据处理、训练、验证、可视化这四部分。

### 数据处理

数据处理想着是尽量复用代码，于是建了一个基类如下，希望一般情况下仅通过更改 `DATASET_NAME` 来优雅地复用（后来复现 AlexNet 时发现事情并没有这么简单）

```python
class DataHandler:
    """数据处理单元"""

    DATASET_NAME = ""

    def __init__(self):
        self.data = None
        self.transform = None
        self.data_loader = None
        self.data_loader_split = None

    def load_data(self, save_path, b_train=True):
        """
        读取数据
        :param save_path: 数据本地存储路径
        :param b_train: True则为训练集
        """

        loader = getattr(torchvision.datasets, self.DATASET_NAME)
        if callable(loader):
            # 判断是否已有本地数据文件
            raw_dir = os.path.join(save_path, self.DATASET_NAME, "raw")
            b_download = os.path.exists(raw_dir)
            # 载入数据
            self.data = loader(
                root=save_path,
                train=b_train,
                transform=self.transform,
                download=b_download
            )
        else:
            raise FileNotFoundError("No dataset named " + self.DATASET_NAME)

    def config_preprocess(self, resize_shape=None, b_tensor=True, mean=None, std=None):
        """
        数据预处理配置
        :param resize_shape: tuple类型，或int
        :param b_tensor: PIL图像转Tensor
        :param mean: 正态分布均值，支持数组
        :param std: 正态分布标准差，支持数组
        """

        transforms = []
        if isinstance(resize_shape, (tuple, int)):
            # 如为单个数字，会保持比例，tuple则改为对应长宽高
            transforms.append(tsf.Resize(resize_shape))

        if b_tensor:
            transforms.append(tsf.ToTensor())
            # Normalize的前提是做了Tensor转化
            if (mean is not None) and (std is not None):
                transforms.append(tsf.Normalize(mean, std))

        self.transform = torchvision.transforms.Compose(transforms)

    def create_dataloader(self, batch_size=32, b_shuffle=True):
        """
        用于迭代获取批次数据的加载器
        :param batch_size: 批次大小
        :param b_shuffle: 是否随机
        """

        assert self.data
        self.data_loader = DataLoader(
            dataset=self.data,
            batch_size=batch_size,
            shuffle=b_shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=True  # 加速GPU传输
        )
        return self.data_loader

    def create_dataloader_split(self, split_ratio=0.8, batch_size=32, b_shuffle=True, random_seed=42):
        """
        加载并拆分数据
        :param split_ratio: train_loader占的比例
        :param batch_size: 批次大小
        :param b_shuffle: 是否随机
        :param random_seed: 随机种子
        :return: train_loader, val_loader
        """

        assert self.data
        total_size = len(self.data)
        train_size = int(total_size * split_ratio)
        val_size = total_size - train_size

        generator = torch.Generator().manual_seed(random_seed)
        subset_train, subset_val = random_split(
            self.data,
            [train_size, val_size],
            generator=generator
        )

        train_loader = DataLoader(
            subset_train,
            batch_size=batch_size,
            shuffle=b_shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )

        val_loader = DataLoader(
            subset_val,
            batch_size=batch_size,
            shuffle=b_shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )

        self.data_loader_split = [train_loader, val_loader]
        return train_loader, val_loader
```

### 训练

### 验证

### 可视化
