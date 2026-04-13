---
title: "GaussianRasterizationSettings"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.diff_gaussian_rasterization.__init__.py.GaussianRasterizationSettings"
hexo-path:
---
## 属性：

| 属性名            | 类型           | 来源函数    | 解释                                                        |
| -------------- | ------------ | ------- | --------------------------------------------------------- |
| antialiasing   | bool = false | <class> | 是否抗锯齿。这个选项直接传给底层 rasterizer                               |
| bg             | torch.Tensor | <class> | 背景颜色。当前像素如果没有高斯覆盖，就会落到背景上                                 |
| campos         | torch.Tensor | <class> | 相机中心位置。后面如果要从 SH 算颜色，需要知道“观察方向”，那就需要相机位                   |
| debug          | bool         | <class> | 是否用 debug 模式                                              |
| image_height   | int          | <class> | 输出图像高度                                                    |
| image_width    | int          | <class> |                                                           |
| prefiltered    | bool         | <class> | 启用预过滤                                                     |
| projmatrix     | torch.Tensor | <class> | 完整投影矩阵，把3D视锥投影到NDC，，x 和 y 轴的范围是 \[-1, 1]，z 轴范围通常是 \[0, 1] |
| scale_modifier | float=1      | <class> | 对所有高斯的尺度做一个统一调节。值大，高斯更“胖”；值小，高斯更“瘦”                       |
| sh_degree      | int          | <class> | 当前启用的球谐阶数                                                 |
| tanfovx        | float        | <class> | 半 FoV 正切                                                  |
| tanfovy        | float        | <class> | -                                                         |
| viewmatrix     | torch.Tensor | <class> | W2C                                                       |

