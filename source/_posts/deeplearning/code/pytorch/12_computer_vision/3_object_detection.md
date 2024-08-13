---
title: 12.3 目标检测和边界框
date: 2024-8-12 16:00:00
tags: [机器学习,pytorch]
categories: [机器学习]
comment: true
toc: true
---

#### 

<!--more-->

# 3 目标检测和边界框



```python
%matplotlib inline
import torch
from d2l import torch as d2l
d2l.set_figsize()
img = d2l.plt.imread('./img/catdog.jpg')
d2l.plt.imshow(img);
```


    
![svg](img/deeplearning/code/pytorch/12_computer_vision/3_object_detection_files/3_object_detection_1_0.svg)
    


## 3.1 边界框（boundingbox）

- 边界框是矩形的，由矩形左上角的以及右下角的x和y坐标决定。另一种常用的边界框表示方法是边界框中心的(x,y)轴坐标以及框的宽度和高度。


```python
#定义在这两种表示法之间进行转换的函数
#@save
def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    #boxes可以是长度为4的张量，也可以是形状为（n，4）的张量, n是锚框的数量
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

#@save
def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    #boxes可以是长度为4的张量，也可以是形状为（n，4）的张量, n是锚框的数量
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes
```

- 根据坐标信息定义图像中狗和猫的边界框。图像中坐标的原点是图像的左上角，向右的方向为x轴的正方向，向下的方向为y轴的正方向。


```python
dog_bbox, cat_bbox = [60.0, 45.0, 500.0, 650.0], [500.0, 112.0, 900.0, 600.0]

#通过转换两次来验证边界框转换函数的正确性。
boxes = torch.tensor((dog_bbox, cat_bbox))
box_center_to_corner(box_corner_to_center(boxes)) == boxes
```




    tensor([[True, True, True, True],
            [True, True, True, True]])



- 将边界框在图中画出

- 定义一个辅助函数bbox_to_rect。它将边界框表示成matplotlib的边界框格式。


```python
#@save
def bbox_to_rect(bbox, color):
    """将边界框（左上x，左上y，右下x，右下y）转换为matplotlib格式。"""
    # ((左上x,左上y),宽,高)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)

fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
```


    
![svg](img/deeplearning/code/pytorch/12_computer_vision/3_object_detection_files/3_object_detection_7_0.svg)
    

