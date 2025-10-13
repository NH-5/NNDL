# 概述

本项目是一个全连接神经网络的项目,实现了手写数字识别的功能.

# 项目文件结构

代码见'mnist.py'文件,数据集来自[这里](https://github.com/RethinkFun/DeepLearning/blob/master/chapter8/data/mnist.zip)

可以保存每一次训练的时间和对应的损失函数及正确率的值在logs文件夹下的txt文件中.

把加载数据集的行为转移到额外的代码实现.

2025.10.13:用ChatGPT对代码做了优化.

# 依赖

项目基于python=3.15.5,pandas=2.3.3,pytorch=2.8.0+cu129实现.

# 模型效果

在测试数据集上最终能达到接近95%的正确率.