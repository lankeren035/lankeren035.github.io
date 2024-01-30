---
title: SRDiff论文讲解
date: 2024-01-27 00:00:00
tags: [深度学习,论文,超分]
categories: [深度学习]
comment: true
toc: true


---

######
<!--more-->

- 论文标题：[Srdiff: Single image super-resolution with diffusion probabilistic models](https://www.sciencedirect.com/science/article/pii/S0925231222000522)

- 来源： Neurocomputing 2022  
- 贡献：
  - 首次将diffusion用于图像超分，与SR3不同的是：SR3直接预测HR图像，而SRDiff预测LR和HR图像之间的差值，这使得DM能够专注于残差细节，加快收敛速度，稳定训练。其次，SRDiff将LR通过encoder后作为条件输入Unet。

# 1. 问题

- 以往的方案基于PSNR，GAN，flow，会出现过于平滑，模式崩溃，模型开销大等问题

# 2. 解决方案

![](D:\blog\themes\yilia\source\img\deeplearning\paper\SR\SRDiff\1.png)

![](img/deeplearning/paper/SR/SRDiff/1.png)

- 将残差图像拿来做扩散
- 将LR通过encoder后作为条件

# 3. 实验

 ![](D:\blog\themes\yilia\source\img\deeplearning\paper\SR\SRDiff\2.png) 

 ![](img/deeplearning/paper/SR/SRDiff/2.png) 

 ![](D:\blog\themes\yilia\source\img\deeplearning\paper\SR\SRDiff\3.png) 

 ![](img/deeplearning/paper/SR/SRDiff/3.png) 