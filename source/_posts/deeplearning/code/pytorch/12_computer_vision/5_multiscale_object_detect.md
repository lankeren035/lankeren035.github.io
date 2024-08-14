---
title: 12.5 多尺度目标检测
date: 2024-8-13 12:00:00
tags: [机器学习,pytorch]
categories: [机器学习]
comment: true
toc: true
---

#### 

<!--more-->

# 5 多尺度目标检测

- 上一节我们以输入图像的每个像素为中心，生成了多个锚框。然而，如果为每个像素都生成的锚框，我们最终可能会得到太多需要计算的锚框。

## 5.1 多尺度锚框

- 我们可以在输入图像中均匀采样一小部分像素，并以它们为中心生成锚框。此外，在不同尺度下，我们可以生成不同数量和不同大小的锚框。

- 直观地说，比起较大的目标，较小的目标在图像上出现的可能性更多样。例如，1×1、1×2和2×2的目标可以分别以4、2和1种可能的方式出现在2×2图像上。因此，当使用较小的锚框检测较小的物体时，我们可以采样更多的区域，而对于较大的物体，我们可以采样较少的区域。

- 先读取一张图像。


```python
%matplotlib inline
import torch
from d2l import torch as d2l
img = d2l.plt.imread('./img/catdog.jpg')
h, w = img.shape[:2]
h, w
```




    (718, 931)



- 我们将卷积图层的二维数组输出称为特征图。通过定义特征图的形状，我们可以确定任何图像上均匀采样锚框的中心。

- display_anchors函数定义如下。我们在特征图（fmap）上生成锚框（anchors），每个单位（像素）作为锚框的中心。由于锚框中的(x,y)轴坐标值（anchors）已经被除以特征图（fmap）的宽度和高度，因此这些值介于0和1之间，表示特征图中锚框的相对位置

- 由于锚框（anchors）的中心分布于特征图（fmap）上的所有单位，因此这些中心必须根据其相对空间位置在任何输入图像上均匀分布。更具体地说，给定特征图的宽度和高度fmap_w和fmap_h，以下函数将均匀地对任何输入图像中fmap_h行和fmap_w列中的像素进行采样。以这些均匀采样的像素为中心，将会生成大小为s（假设列表s的长度为1）且宽高比（ratios）不同的锚框。


```python
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # 前两个维度上的值不影响输出
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes, anchors[0] * bbox_scale)
```

- 首先，让我们考虑探测小目标。为了在显示时更容易分辨，在这里具有不同中心的锚框不会重叠：锚框的尺度设置为0.15，特征图的高度和宽度设置为4。我们可以看到，图像上4行和4列的锚框的中心是均匀分布的。


```python
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
```

    c:\Users\admin\miniconda3\envs\d2l\lib\site-packages\torch\functional.py:513: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\native\TensorShape.cpp:3610.)
      return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
    


    
![svg](img/deeplearning/code/pytorch/12_computer_vision/5_multiscale_object_detect_files/5_multiscale_object_detect_5_1.svg)
    


- 我们将特征图的高度和宽度减小一半，然后使用较大的锚框来检测较大的目标。当尺度设置为0.4时，一些锚框将彼此重叠。


```python
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
```


    
![svg](img/deeplearning/code/pytorch/12_computer_vision/5_multiscale_object_detect_files/5_multiscale_object_detect_7_0.svg)
    


- 进一步将特征图的高度和宽度减小一半，然后将锚框的尺度增加到0.8。此时，锚框的中心即是图像的中心。


```python
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
```


    
![svg](img/deeplearning/code/pytorch/12_computer_vision/5_multiscale_object_detect_files/5_multiscale_object_detect_9_0.svg)
    


## 5.2 多尺度检测

- 一种基于CNN的多尺度目标检测方法:

    - 在某种规模上, 假设我们有c张形状为h x w的特征图, 用上面的方法生成了hw组锚框, 每组锚框中的锚框个数为a, 那么一共有c x h x w x a个锚框. 接下来，每个锚框都根据真实值边界框来标记了类和偏移量。在当前尺度下，目标检测模型需要预测输入图像上hw组锚框类别和偏移量，其中不同组锚框具有不同的中心。

    - 假设此处的c张特征图是CNN基于输入图像的正向传播算法获得的中间输出。既然每张特征图上都有hw个不同的空间位置，那么相同空间位置可以看作含有c个单元。根据对感受野的定义，特征图在相同空间位置的c个单元在输入图像上的感受野相同：它们表征了同一感受野内的输入图像信息。因此，我们可以将特征图在同一空间位置的c个单元变换为使用此空间位置生成的a个锚框类别和偏移量。本质上，我们用输入图像在某个感受野区域内的信息，来预测输入图像上与该区域位置相近的锚框类别和偏移量。

    - 当不同层的特征图在输入图像上分别拥有不同大小的感受野时，它们可以用于检测不同大小的目标。例如，我们可以设计一个神经网络，其中靠近输出层的特征图单元具有更宽的感受野，这样它们就可以从输入图像中检测到较大的目标。

- 简言之，我们可以利用深层神经网络在多个层次上对图像进行分层表示，从而实现多尺度目标检测。
