---
title: "__init__"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/scene.gaussian_model.py.GaussianModel.__init__"
hexo-path:
---

## 1. 输入

| 参数名            | 类型                | 解释      |
| -------------| ----------------| ------|
| sh_degree      |                 | 球谐阶数（3） |
| optimizer_type | default='default' |       |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| | |

## 3. 操作逻辑

1. 初始化如下属性：
	```
	active_sh_degree = 0        #当前使用的最大球谐函数阶数   
	optimizer_type = optimizer_type
	max_sh_degree = sh_degree    #球谐系数
	_xyz = torch.empty(0)         # (P, 3), 每个点的xyz坐标
	_features_dc = torch.empty(0)  # 当前阶数的球谐系数
	_features_rest = torch.empty(0) # 剩余阶数的球谐系数
	_scaling = torch.empty(0)      # 协方差的缩放参数
	_rotation = torch.empty(0)     # 旋转
	_opacity = torch.empty(0)      # 不透明度
	max_radii2D = torch.empty(0)   # (P,), 渲染时的屏幕空间半径上界（占位/加速用）
	xyz_gradient_accum = torch.empty(0) # [P,1]，xyz坐标的梯度累积，用于自适应学习率？？？
	denom = torch.empty(0)       # [P,1]，xyz坐标的梯度累积的归一化因子？？？
	optimizer = None
	percent_dense = 0 
	spatial_lr_scale = 0
	```

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_gaussian_model.py\|gaussian_model.py]] |
| 所属类 | [[3dgs_GaussianModel\|GaussianModel]] |
| 命名空间 | |
| 类型 | |
