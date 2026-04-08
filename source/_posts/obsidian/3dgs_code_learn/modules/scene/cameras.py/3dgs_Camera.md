---
title: "Camera"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/scene.cameras.py.Camera"
hexo-path:
---

## 1. 输入

| 参数名             | 类型                                | 解释                   |
| --------------- | --------------------------------- | -------------------- |
| resolution      | -                                 | 最终图像分辨率              |
| colmap_id       | -                                 | 相机id                 |
| R               | -                                 | 旋转矩阵                 |
| T               | -                                 | 平移向量                 |
| FoVx            | -                                 | x视场角                 |
| FoVy            | -                                 | y视场角                 |
| depth_params    | -                                 | 深度参数                 |
| image           | -                                 | RGB图                 |
| invdepthmap     | -                                 | 逆深度图                 |
| image_name      | -                                 | 图片文件名                |
| uid             | -                                 | 相机信息序号               |
| trans           | default=np.array([0.0, 0.0, 0.0]) | -                    |
| scale           | default=1.0                       | -                    |
| data_device     | default='cuda'                    | -                    |
| train_test_exp  | default=False                     | 是否使用 train/test 实验设置 |
| is_test_dataset | default=False                     | 当前相机是不是来自测试集         |
| is_test_view    | default=False                     | 这个视角是不是被标记为测试视角      |
- **思考1**：逆深度图是啥？为什么要用逆深度图，而不是深度图？
	- 深度图：一个像素对应的3D点，沿着相机视线方向，离相机有多远。如1米，则该像素点的depth=1，10米则该像素点的depth=10
	- 逆深度：如1米，则该像素点的depth=1，10米则该像素点的depth=1/10
	- 好处：
		- 更适合表示“近处几何差异”，人眼和相机里，近处一点点距离变化，视觉影响很大
		- 透视关系里更自然
		- 数值范围更稳定，数值分布通常更集中一些，很多时候更利于学习或监督。
- **思考2**：is_test_dataset和is_test_view两个参数感觉一模一样啊？
	- 这取决于train_test_exp是否开启，如果不开启，那么这两个参数确实一模一样，如果开启了则代表：假设我的数据集一共100张图，然后从里面挑10张图作为测试。 最终90张图整图用于训练，10张图只训左半边（右边mask）。 10张图测试右半边。
- **思考3**：为什么要train_test_exp模式？
	- 这个模式里，模型会给每张图单独学习一个曝光参数。如果测试图完全不参与训练，那这张测试图对应的曝光参数根本学不到。
## 2. 属性

| 属性名                  | 类型                      | 来源函数     | 解释                                                                       |
| -------------------- | ----------------------- | -------- | ------------------------------------------------------------------------ |
| FoVx                 | -                       | __init__ | x视场角                                                                     |
| FoVy                 | -                       | __init__ | y视场角                                                                     |
| R                    | -                       | __init__ | 旋转矩阵                                                                     |
| T                    | -                       | __init__ | 平移向量                                                                     |
| alpha_mask           | -                       | __init__ | 用于曝光训练对一张图做左右mask                                                        |
| camera_center        | -                       | __init__ | -                                                                        |
| colmap_id            | -                       | __init__ | 相机id                                                                     |
| data_device          | -                       | __init__ | -                                                                        |
| depth_mask           | -                       | __init__ | 深度图的mask                                                                 |
| depth_reliable       | bool=False              | __init__ | 深度是否可靠                                                                   |
| full_proj_transform  | -                       | __init__ | -                                                                        |
| image_height         | -                       | __init__ | 最终图片高度                                                                   |
| image_name           | -                       | __init__ | 图片文件名                                                                    |
| image_width          | -                       | __init__ | 最终图片宽度                                                                   |
| invdepthmap          | default=None            | __init__ | 逆深度                                                                      |
| original_image       | -                       | __init__ | GT图[0,1]                                                                 |
| projection_matrix    | -                       | __init__ | -                                                                        |
| scale                | default=1               | __init__ | 场景归一化参数，这里采用默认值，按理来说应该调用的时候传入之前计算的场景尺度： 1/radius [[3dgs_getNerfppNorm]]  |
| trans                | default=[0.0, 0.0, 0.0] | __init__ | 场景归一化参数，这里采用默认值，按理来说应该调用的时候传入之前计算的场景位移： translate [[3dgs_getNerfppNorm]] |
| uid                  | -                       | __init__ | 相机信息序号                                                                   |
| world_view_transform | -                       | __init__ | -                                                                        |
| zfar                 | -                       | __init__ | -                                                                        |
| znear                | -                       | __init__ | -                                                                        |

## 3. 方法

| 方法名 | 解释 |
|---|---|
| [[source/_posts/obsidian/3dgs_code_learn/functions/scene/cameras.py/Camera/3dgs___init__\|__init__]] | - |
