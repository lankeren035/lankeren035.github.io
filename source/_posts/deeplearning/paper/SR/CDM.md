---
title: CDM论文讲解
date: 2024-01-26 00:00:00
tags: [深度学习,论文,超分]
categories: [深度学习]
comment: true
toc: true

---

######
<!--more-->

- 论文标题：[Cascaded diffusion models for high fidelity image generation](https://dl.acm.org/doi/abs/10.5555/3586589.3586636)

- 来源：  JMLR 2022  
- 贡献：
  - Cascaded Diffusion Models(CDM)产生的高保真样本在FID评分和分类准确性评分方面优于BigGAN-deep和VQ-VAE-2。
  - 为超分辨率模型引入了条件增强，并发现它对实现高样本保真度至关重要。



# 1. 方法

- 整体思想： Classifier Diffusion Models + SR3 + Tricks的串联模型，应用多个不同分辨率的扩散模型实现超分效果。 

![](D:\blog\themes\yilia\source\img\deeplearning\paper\SR\CDM\1.png)
![](img/deeplearning/paper/SR/CDM/1.png)
![](D:\blog\themes\yilia\source\img\deeplearning\paper\SR\CDM\2.png)
![](img/deeplearning/paper/SR/CDM/2.png)
- SR3中也提到可以用级联的SR3做生成，本文在此基础上提出条件增强以提高生成质量

# 1.1 条件增强

- 截断条件采样：逆向过程中的中间图片Xt输入下一个超分模型（而不是XT，即逆向过程只做一部分）
- 非截断条件增强：生成XT之后再施加高斯噪声后输入下一个模型



# 2. 实验

![](D:\blog\themes\yilia\source\img\deeplearning\paper\SR\CDM\3.png)

![](img/deeplearning/paper/SR/CDM/3.png)

- 效果比直接级联SR3好