---
title: 1. DDPM论文理解
date: 2023-10-24 10:28:00
tags: [深度学习,DDPM,生成模型]
categories: [深度学习]
comment: false
toc: true
---
#
<!--more-->

- 文章：[去噪扩散概率模型](https://arxiv.org/pdf/2006.11239.pdf)

- 代码：https://github.com/hojonathanho/diffusion
# 一. 论文摘要

- 提出了扩散概率模型
- 潜在变量模型来源于非平衡热力学
- 在加权变分界限上进行训练（变分界限是根据“扩散概率模型” 和 “采用了去噪分数匹配与朗之万动力学的训练方法“之间的联系
- 模型允许渐进的有损解压（可以解释为自回归解码的推广）
- 在无条件CIFAR10上Inception=9.46，FID=3.17。在256*256 LSUN上与Progressive GAV相似
