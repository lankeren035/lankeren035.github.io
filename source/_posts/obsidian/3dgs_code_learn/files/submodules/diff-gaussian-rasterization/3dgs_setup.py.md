---
title: "setup.py"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.setup.py"
hexo-path:
---

## 1. 流程

1. 将如下C++代码编译成python包，命名为`_C`
	- cuda_rasterizer/rasterizer_impl.cu
	- cuda_rasterizer/forward.cu
	- cuda_rasterizer/backward.cu
	- rasterize_points.cu
2. 为该python包打包3个接口[[source/_posts/obsidian/3dgs_code_learn/files/submodules/diff-gaussian-rasterization/3dgs_ext.cpp|3dgs_ext.cpp]]

## 2. 包含的类

| 类名 | 解释 |
|---|---|
| - | - |

## 3. 包含的函数

| 函数名 | 返回 | 解释 |
|---|---|---|
| - | - | - |
