---
title: 1. A Neural Algorithm of Artistic Style 2015
date: 2024-9-16 14:00:00
tags: [论文, 风格迁移]
categories: [风格迁移, 论文]
comment: true
toc: true
---

#### 

<!--more-->

# A Neural Algorithm of Artistic Style 2015




$$l_{\text {content }}(\vec{p}, \vec{x}, l)=\frac{1}{2} \sum_{i, j}\left(F_{i j}^{l}-P_{i j}^{l}\right)^{2}$$

- 这里$\frac{1}{2}$是为了方便求导

$$l_{\text {style }}(\vec{a}, \vec{x})=\sum_{l=0}^{L} w_{l} E_{l}$$
$$E_{l}=\frac{1}{4 N_{l}^{2} M_{l}^{2}} \sum_{i, j}\left(G_{i j}^{l}-A_{i j}^{l}\right)^{2}$$

- $N_{l}$是第$l$层的通道数，$M_{l}$是第$l$层的特征图的高和宽的乘积
- 这里$\frac{1}{4}$是为了方便求导, $\frac{ \partial l_{\text {style }} }{ \partial G_{i j}^{l} }$有个$\frac{1}{2}$, $\frac{ \partial G_{i j}^{l} }{ \partial F_{i j}^{l} }$有个$\frac{1}{2}$, 所以$\frac{1}{4}$. G是Gram矩阵.