---
title: "create_from_pcd"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/scene.gaussian_model.py.GaussianModel.create_from_pcd"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| pcd | BasicPointCloud | - |
| cam_infos | int | - |
| spatial_lr_scale | float | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| - | - |

## 3. 操作逻辑

1. 初始化训练场景中所有高斯的如下属性：
	- 场景尺度
	- 从点云数据获得每个点的位置xyz
	- 从点云数据读取每个点的颜色RGB，然后转换成0阶sh: [[3dgs_RGB2SH]]
	- 初始化颜色：0-3阶，3通道，每通道16个颜色系数：[p,3,16]，初始用0阶，因此每个通道只有1个系数[p,3,1]，其他置为0
	- 初始化每个高斯的半径（各向同性初始化）
		- 计算点云中每个点与其最近邻的3个点的平方距离的平均值3knn：[[source/_posts/obsidian/3dgs_code_learn/functions/submodules/simple-knn/spatial.cu/3dgs_distCUDA2|3dgs_distCUDA2]]
		- 将$\sqrt{3knn}$ 作为高斯半径，同时将半径存储为log变成可训练的尺度参数（使用时会做exp激活），扩展到xyz三个方向
	- 初始化每个点的旋转四元数：(1,0,0,0)表示不旋转（初始各向同性）（也会做激活）
	- 初始化不透明度：0.1，同时也做反向激活存储
2. 注册可学习参数：
	
| 参数              | shape       | 解释                                     |
| --------------- | ----------- | -------------------------------------- |
| \_xyz           | (P,3)       | 高斯中心的 3D 坐标                            |
| \_features_dc   | (P,1,3)     | 0阶sh系数/当前训练球谐系数，每个通道各1个（转置存储）          |
| \_features_rest | (P,15,3)    | 高阶sh系数（转置存储，这种布局对后续 rasterizer 更顺手）    |
| \_scaling       | (P, 3)      | 尺度                                     |
| \_rotation      | (P, 4)      | 旋转                                     |
| \_opacity       | (P, 1)      | 不透明度                                   |
| \_exposure      | (训练图片数，3,4) | 每个训练视角都有自己独立的一组曝光/颜色校正参数，用于对每个像素颜色进行矫正 |
3. 其他参数初始化：
	- max_radii2D=0 ：【p】，每个高斯在屏幕空间投影半径的历史最大值缓存。主要用于：
		- densification / pruning 判断
		- 屏幕空间覆盖范围统计
		- 加速和可见性相关逻辑
	- exposure_mapping，每张图的曝光参数索引 {图1：曝光参数1，图2：曝光参数2}
	- pretrained_exposures=None，当前先不用预训练曝光
	- \_exposure=（3，4）单位矩阵，用于曝光颜色修改
## 4. 信息

| 字段   | 内容                         |     |
| ---- | -------------------------- | --- |
| 所属文件 | [[3dgs_gaussian_model.py]] |     |
| 所属类  | [[3dgs_GaussianModel]]     |     |
| 命名空间 | -                          |     |
| 类型   | -                          |     |
