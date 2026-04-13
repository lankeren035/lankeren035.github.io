---
title: "coord2Morton"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.simple-knn.simple_knn.cu.coord2Morton"
hexo-path:
---
- 对一个点（x,y,z)，该操作将这个坐标编码成一个数，先把坐标归一化到0-1024（用10个bit表示），因此该点用二进制表示为（x0x1x2...x9, y0y1y2...y9, z0z1z2...z9)，然后把这些bit交错起来，得到一个30bit的数：x0y0z0x1y1z1...x9y9z9，这样就得到了这个点的morton编码。如此编码之后，空间中相近的点在morton编码上也会相近，方便后续排序和分块。
## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| P | int | - |
| points | const float3* | - |
| minn | float3 | - |
| maxx | float3 | - |
| codes | uint32_t* | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| __global__ void | - |

## 3. 操作逻辑

1. 对x,y,z分别处理：
	- 将坐标值归一化到0-1023
	- 二进制扩展： b1 b2 b3 b4 -> b1 0 0 b2 0 0 b3 0 0 b4 0 0
2. 对xyz进行错位组合


- morton编码，例如对一个点（x,y,z)，该操作将这个坐标编码成一个数，先把坐标归一化到0-1024（用10个bit表示），因此该点用二进制表示为（x0x1x2...x9, y0y1y2...y9, z0z1z2...z9)，然后把这些bit交错起来，得到一个30bit的数：x0y0z0x1y1z1...x9y9z9，这样就得到了这个点的morton编码。如此编码之后，空间中相近的点在morton编码上也会相近，方便后续排序和分块。不过这里并不是空间中最近的三个点一定是严格在morton编码上最近的三个点，但通常是比较接近的。 它有点类似于ip地址的编码
## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_simple_knn.cu\|simple_knn.cu]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | definition |
