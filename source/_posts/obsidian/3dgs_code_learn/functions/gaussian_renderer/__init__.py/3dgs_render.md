---
title: "render"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/gaussian_renderer.__init__.py.render"
hexo-path:
---

## 1. 输入

| 参数名              | 类型            | 解释                                      |
| ---------------- | ------------- | --------------------------------------- |
| viewpoint_camera | -             | 当前训练的视角下的这一条相机信息                        |
| pc               | GaussianModel | 场景中的高斯                                  |
| pipe             | -             | pipeline配置                              |
| bg_color         | torch.Tensor  | 背景颜色                                    |
| scaling_modifier | default=1.0   | 对高斯尺度的一个整体缩放系数，调试或 GUI 控制时会改它           |
| separate_sh      | default=False | 是否分离SH相关处理，和稀疏Adam支持有关                  |
| override_color   | default=None  | 如果不为 `None`，说明你强行指定颜色，不再用高斯自己的 SH/特征算颜色 |
| use_trained_exp  | default=False | 是否使用训练得到的 exposure 参数对最终图像做曝光变换         |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| - | - |

## 3. 操作逻辑

1. 仿照高斯xyz的形状与数据类型创建一个屏幕空间占位张量【p,3】，为了让它在反向传播时能**接住屏幕空间 2D 均值相关的梯度**。后面 densify 就要用这些梯度做统计（这里有一个+0操作，这是一个很常见的小技巧，本质上不改变数值，仍然是 0。主要是为了生成一个新的张量表达式，兼容某些 autograd / 扩展算子的行为）
2. 计算tanfovx/tanfovy（注意这里用的是半视场角）
3. 打包光栅化参数：[[3dgs_GaussianRasterizationSettings|raster_settings]]
4. 创建光栅化器：[[source/_posts/obsidian/3dgs_code_learn/functions/submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py/GaussianRasterizer/3dgs___init__|rasterizer ( raster_settings )]]
5. 准备光栅化器输入：
	- means2D：2D坐标展位张量（第一步创建的）【p，3】
	- 高斯核心参数：
		- means3D：高斯中心【p，3】
		- opacity：不透明度【p，1】
		- scales：高斯尺度【p，3】
		- rotations：高斯旋转【p，4】
			- 如果启用python预计算则在python端使用scale和rotations先算出协方差
		- shs：球谐系数【p，16，3】
			- 如果启动颜色覆盖，则使用外部指定颜色
			- 如果启动python预计算则在python端使用当前激活的sh阶数、sh系数、观察方向计算出高斯RGB \[0,x]（后面会clamp到\[0,1]）
			- 如果启动分离sh模式，则将特征拆成两部分：
				- dc：0阶分量
				- shs：高阶sh分量
6. 光栅化/渲染[[source/_posts/obsidian/3dgs_code_learn/functions/submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py/GaussianRasterizer/3dgs_forward|3dgs_forward]]

## 4. 信息

| 字段   | 内容                                                                                |               |
| ---- | --------------------------------------------------------------------------------- | ------------- |
| 所属文件 | [[source/_posts/obsidian/3dgs_code_learn/files/gaussian_renderer/3dgs___init__.py | __init__.py]] |
| 所属类  | -                                                                                 |               |
| 命名空间 | -                                                                                 |               |
| 类型   | -                                                                                 |               |
