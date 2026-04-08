---
title: "Scene"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/scene.__init__.py.Scene"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| args | ModelParams | - |
| gaussians | GaussianModel | - |
| load_iteration | default=None | - |
| shuffle | default=True | - |
| resolution_scales | default=[1.0] | - |

## 2. 属性

| 属性名 | 类型 | 来源函数 | 解释 |
|---|---|---|---|
| gaussians | GaussianModel | <class> | - |
| cameras_extent | - | __init__ | - |
| gaussians | - | __init__ | - |
| loaded_iter | - | __init__ | - |
| model_path | - | __init__ | - |
| test_cameras | - | __init__ | - |
| train_cameras | - | __init__ | - |

## 3. 方法

| 方法名 | 解释 |
|---|---|
| [[source/_posts/obsidian/3dgs_code_learn/functions/scene/__init__.py/Scene/3dgs___init__\|__init__]] | b |
| [[3dgs_save\|save]] | - |
| [[3dgs_getTrainCameras\|getTrainCameras]] | - |
| [[3dgs_getTestCameras\|getTestCameras]] | - |
