---
title: "boxMinMax"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.simple-knn.simple_knn.cu.boxMinMax"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| P | uint32_t | - |
| points | float3* | - |
| indices | uint32_t* | - |
| boxes | MinMax* | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| __global__ void | - |

## 3. 操作逻辑

1. 初始化1024个线程，每个线程处理一个点
2. 创建block共享数组，一个block表示一个box
3. 归约循环：
	1) 写入：前1024个线程将当前点坐标作为当前线程的min和max
	2) 同步
	3) 比较：前512个线程负责进行比较：0对511，1对512 ......得到512个结果
		- 当前线程的min.x等于二者最小，min.y，min.z以及max类似
	4) 同步
	5) 进入下一轮循环，512个线程进行处理
4. 归约结束，剩余0号线程的min和max就表示这个box的min和max，用于表达盒子边界

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_simple_knn.cu\|simple_knn.cu]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | definition |
