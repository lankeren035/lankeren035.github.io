---
title:  "Zero-Shot Video Editing Using Off-The-Shelf  Image Diffusion Models论文理解"

date:  2025-2-14 11:28:00

tags:  [视频编辑,论文]

categories:  [论文,视频]

comment:  false

toc:  true


---

#

<!--more-->

- arxiv 2023
- [论文地址](https://arxiv.org/abs/2303.17599)

- [项目地址](https://github.com/baaivision/vid2vid-zero)



# 0. Abstract

大规模文本到图像扩散模型在图像生成和编辑方面取得了前所未有的成功。然而，如何将这样的成功扩展到视频编辑尚不清楚。最近对视频编辑的初步尝试需要大量的文本到视频数据和计算资源用于训练，这通常是不可访问的。在这项工作中，我们提出了vid2vid-zero，一种简单而有效的零镜头视频编辑方法。我们的vid2vidzero利用现成的图像扩散模型，不需要对任何视频进行培训。我们方法的核心是用于文本到视频对齐的空文本反转模块、用于时间一致性的跨帧建模模块和用于原始视频保真度的空间正则化模块。在没有任何训练的情况下，我们利用注意力机制的动态性质来实现测试时的双向时间建模。实验和分析显示了在编辑属性、主题、地点等方面有希望的结果。，在现实世界的视频中。



> TAV + VIdeo p2p ?