---
title: "training_setup"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/scene.gaussian_model.py.GaussianModel.training_setup"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| training_args | - | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| - | - |

## 3. 操作逻辑

1. 初始化高斯的如下属性：
	- densify尺度（用于后面判断高斯算大还是小，决定走split还是clone）
	- densify梯度累计器（后面 `add_densification_stats()` 里会把可见高斯的 2D view-space 梯度范数累加进来）
	- 计数器/归一化分母（后面每次某个高斯参与统计时，`self.denom[update_filter] += 1`；最终 `densify_and_prune()` 里会算 `grads = self.xyz_gradient_accum / self.denom`，得到平均梯度强度）
2. 初始化高斯核心属性的学习率
3. 创建高斯核心参数优化器、曝光优化器
4. 创建[[3dgs_get_expon_lr_func|位置学习率调度器]](传参数初始值和最终值乘以场景归一化尺度)、[[3dgs_get_expon_lr_func|曝光学习率调度器]]

## 4. 信息

| 字段   | 内容                         |     |
| ---- | -------------------------- | --- |
| 所属文件 | [[3dgs_gaussian_model.py]] |     |
| 所属类  | [[3dgs_GaussianModel]]     |     |
| 命名空间 | -                          |     |
| 类型   | -                          |     |
