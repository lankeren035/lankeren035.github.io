---
title: "Gen-L-Video: Multi-Text to Long Video Generation via Temporal Co-Denoising"
date: 2025-5-13 10:28:00
tags: [diffusion,论文]
categories: [diffusion]
comment: false
toc: true



---

#
<!--more-->

- 本文将文生视频/视频编辑分为三类：
  - 预训练方法
  - zero shot方法
  - one shot方法

## 方法

### 时间协同去噪

- 就是将长视频生成变成多个clip生成，每个clip之间有重叠，重叠部分取平均

### 将Gen-L-视频与主流范例相结合

- 预训练方法

  - 使用本文提出的co-denoising

- zero shot方法

  - co-denoising
  - 这种方法通常使用跨帧注意力，将跨帧注意力的锚帧换成每个clip的中间帧

- one shot方法

  - 当不同clip使用同一个提示词的时候，无法区分不同clip，本文提出学习一个clip标识作为网络条件，然后推理的时候使用cfg：

    $$\hat \epsilon_ \theta (v^i_ t , t, c ^ i, e^ i ) = (1 + w ) \epsilon _ \theta (v_ t ^ i , t, c^ i ,e^ i) - w \epsilon_ \theta ( v^ i _ t , t, \varnothing, \varnothing)$$