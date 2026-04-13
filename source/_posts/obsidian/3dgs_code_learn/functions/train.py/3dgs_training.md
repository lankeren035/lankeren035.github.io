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
	- 初始化高斯模型的训练参数[[3dgs_training_setup|training_setup(opt)]]
	- 加载预训练模型
	- 背景颜色 \[0,0,0]
	- 深度损失权重进行指数衰减调度
2. 读取训练相机列表（无放回采样）[[3dgs_getTrainCameras]]
3. 连接GUI（这个 GUI 一般是远程可视化调试界面，用来：查看当前渲染结果、手动切换视角、控制是否继续训练、看实时结果）##todo
4. 训练循环：
	1) 更新学习率[[3dgs_update_learning_rate]]
	2) 每 1000 次迭代，把球谐函数 SH 的阶数提高一级
	3) 若视角池空了，就重新装满，采样
	4) 当前步以后是否启动debug模式
	5) **渲染**[[source/_posts/obsidian/3dgs_code_learn/functions/gaussian_renderer/__init__.py/3dgs_render|render(当前训练视角信息，场景高斯，pipeline配置，背景颜色，...)]]
	

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_train.py|train.py]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | - |
