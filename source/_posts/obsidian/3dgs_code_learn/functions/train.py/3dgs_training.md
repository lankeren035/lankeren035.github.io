---
title: "training"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/train.py.training"
hexo-path:
---

## 1. 输入

| 参数名                   | 类型  | 解释                                    |
| --------------------- | --- | ------------------------------------- |
| dataset               | -   | 模型超参数，包含数据集相关信息：数据集路径、分辨率、输出模型路径、sh阶数 |
| opt                   | -   | 优化超参数，包括学习率、损失权重、阈值                   |
| pipe                  | -   | pipeline超参数，包括：使用python计算，调试模式、抗锯齿    |
| testing_iterations    | -   | -                                     |
| saving_iterations     | -   | -                                     |
| checkpoint_iterations | -   | -                                     |
| checkpoint            | -   | -                                     |
| debug_from            | -   | -                                     |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| - | - |

## 3. 操作逻辑

1. 初始化设置
	- 初始化类：高斯模型类( [[3dgs_GaussianModel]] )，[[source/_posts/obsidian/3dgs_code_learn/functions/scene/gaussian_model.py/GaussianModel/3dgs___init__|构造方法]]
	- 初始化类：场景类（传入高斯模型类实例）( [[3dgs_Scene]] )，[[source/_posts/obsidian/3dgs_code_learn/functions/scene/__init__.py/Scene/3dgs___init__|构造方法]]

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_train.py|train.py]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | - |
