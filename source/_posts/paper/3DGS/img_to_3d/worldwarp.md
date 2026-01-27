---
title:  "WorldWarp: Propagating 3D Geometry with Asynchronous Video Diffusion"

date:  2025-12-27 15:28:00

tags:  [3DGS]

categories:  [3DGS]

comment:  false

toc:  true




---

#

<!--more-->



## Abstract

-  **长序列、强相机控制（给定轨迹）、几何一致** 的 单图生成视频， 并且能被重建成一致的 3D 结构 
- 现有方法的问题：
  -  **只用相机姿态编码（latent 条件）**：依赖数据集多样性，对分布外相机运动泛化差，而且“姿态”本身不提供场景的**像素级几何约束**。 
  -  **用显式 3D 先验（比如点云/mesh/3DGS）再投影**：投影会出现 **遮挡空洞（holes）**，还会有 **几何误差造成的扭曲/模糊**；直接 inpainting 或普通视频扩散很难同时“补洞 + 修扭曲”。 



## Method

>-  用“在线 3D 几何缓存（3DGS）”提供像素空间的结构锚点，再用一个专门训练过的时空扩散模型 ST-Diff 做“fill-and-revise”：空洞从纯噪声生成（fill），有几何但质量不好的区域只加部分噪声做修复（revise）。 