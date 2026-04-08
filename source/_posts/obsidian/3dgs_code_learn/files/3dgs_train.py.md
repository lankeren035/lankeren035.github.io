---
title: "train.py"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/train.py"
hexo-path:
---

## 1. 流程

1. 读取如下配置：
	- 模型参数lp，[[3dgs_ModelParams]]
	- 训练参数op，[[3dgs_OptimizationParams]]
	- pipline参数pp，[[3dgs_PipelineParams]]
2. 接收外部参数：
	- --ip
	- --port
	- --debug_from
	-   --detect_anomaly
	- --test_iterations
	- --save_iterations
	- --quiet
	- --disable_viewer
	- --checkpoint_iterations
	- --start_checkpoint
3. 开启训练：[[3dgs_training]]
## 2. 包含的类

| 类名 | 解释 |
|---|---|
| - | - |

## 3. 包含的函数

| 函数名 | 返回 | 解释 |
|---|---|---|
| [[3dgs_training\|training]] | - | - |
| [[3dgs_prepare_output_and_logger\|prepare_output_and_logger]] | - | - |
| [[3dgs_training_report\|training_report]] | - | - |
| [[source/_posts/obsidian/3dgs_code_learn/functions/train.py/3dgs_0_main\|main]] | - | 由 if __name__ == '__main__' 入口块生成的入口笔记 |
