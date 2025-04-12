---
title: "UNIVST: A UNIFIED FRAMEWORK FOR TRAINING-FREE LOCALIZED VIDEO STYLE TRANSFER"

date: 2025-3-25

tags: [论文]

categories: [论文]

comment: true

toc: true



---

#

<!--more-->





# 方法

## 4.1 点匹配帧传播

- 核心思想：对每个潜在特征使用掩码，掩码内的使用编辑分支，掩码外的使用内容分支

- 点匹配：类似tokenflow的方式，对于第一帧的点A，从第二帧遍历所有的点，找到与之距离最近的k个点作为对应点。根据这个位置映射关系（使用inversion过程中upblock的第二个block的特征）来传播掩码。

  

## 4.2 AdaIN 引导的风格迁移

- 在t0-t1时间对 编辑分支和风格分支的zt执行adain（有权重变化）
- styleid直接将kv替换为风格分支的，q使用编辑分支和内容分支的加权；本文把k和v用编辑分支与风格分支做adain（在t2,t3做，动态权重）



## 4.3 滑动窗口平滑

- 对一步预测的zt->0，解码得到视频，对于解码视频选择2m大小的窗口，以i位置的帧为中心，所有帧向他warp取均值替换第i帧。