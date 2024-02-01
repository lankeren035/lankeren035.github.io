---
title: Resshift论文讲解
date: 2024-01-27 00:00:00
tags: [深度学习,论文,超分]
categories: [深度学习]
comment: true
toc: true



---

#
<!--more-->

- 论文标题：[Resshift: Efficient diffusion model for image super-resolution by residual shifting](https://arxiv.org/abs/2307.12348)

- 来源：NeurIPS. 2023 
- 贡献：
  - 提出了一种残差移动方法，加快了扩散模型推理速度，同时保留高性能。
  - 提出一个噪声设计方案，能够有效控制扩散过程中的噪声强度和转换速度，也可有效控制保真度-真实性之间的权衡。  

# 1. 问题

- 基于diffusion的超分辨率采样速度慢，现有的加速采样技术不可避免地会在一定程度上牺牲性能，导致超分结果过于模糊。



# 2. 方法

