# 概述

本项目是一个全连接神经网络的项目,实现了手写数字识别的功能.

所使用的损失函数为均方误差函数或交叉熵损失函数.

# 项目文件结构

代码见`mnist.py`文件,数据集来自[这里](https://github.com/RethinkFun/DeepLearning/blob/master/chapter8/data/mnist.zip)

可以保存每一次训练的时间和对应的损失函数及正确率的值在logs文件夹下的txt文件中.

将网络,数据集加载,训练结束后通知,config,训练,记录数据等模块分开实现

# 依赖

项目主要基于:`python=3.13.5` `pandas=2.3.3` `requests=2.32.5` `torch=2.8.0+cu129`

详细信息参考`env.yml`,文件来自`conda env export > env.yml`.

# 模型效果

训练30轮,batch_size=64的情况:

    当使用均方误差函数,学习率设置3左右.在测试数据集上最终能达到接近95%的正确率.

    当使用交叉熵损失函数时,学习率设置0.01.在测试数据集上最终能达到88.97%的正确率.


# 结果复现

- `git clone`代码
- 新建`.env`文件,根据`.env.example`所示格式写入`bark_key`
- 根据实际修改`train.py`中各项参数,运行`train.py`开始训练


# 注意事项

- 当前仅对所提供链接处下载的CSV数据集做过测试,确认代码能够运行