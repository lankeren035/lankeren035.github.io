---
title: "__init__"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/scene.cameras.py.Camera.__init__"
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
## 2. 输出

| 返回类型 | 解释 |
|---|---|
| - | - |

## 3. 操作逻辑

1. 赋予属性
2. 确定数据设备
	- 这里决定图像、mask、深度图这些数据最终放在哪个设备上。
	- 但是后面变换矩阵部分是直接强制放到 CUDA 上的，并没有严格遵循 `data_device`这说明这个实现默认就是面向 GPU 训练写的。如果你真把 `data_device="cpu"`，图像可能在 CPU，但矩阵还是会去 CUDA，这里其实有潜在设备不一致问题!!
3. 图片从PIL转为torch格式[[3dgs_PILtoTorch]]
4. 构造alpha_mask:
	- 如果是RGBA图，提取alpha_mask
	- 否则用全1mask，整张图都是有效区域
5. 如果启用train_test_exp（假设我的数据集一共100张图，然后从里面挑10张图作为测试。 最终90张图整图用于训练，10张图只训左半边（右边mask）。 10张图测试右半边。）：
	- 测试集图片：alpha_mask左侧全0，表示用右侧左测试
	- 训练集中的测试集图片：alpha_mask右侧全0，表示用左侧左训练
6. 处理逆深度图以及其mask：##todo
7. 设置近平面（100）远平面（0.01）
8. 计算W2C（可自定义） [[3dgs_getWorld2View2]] （这里会将W2C进行转置保存，这是为了配合 CUDA / OpenGL 风格 / 行向量乘法约定，因此后面取数据的时候`self.camera_center = self.world_view_transform.inverse()[3, :3]` 取的是第四行而不是第四列，且这里代码写死的直接放到GPU上!!）
9. 计算透视投影矩阵projection_matrix [[3dgs_getProjectionMatrix]]

- **思考1**：在 8. 这里，之前在计算场景的缩放尺度和平移时，计算过W2C，为什么那个时候不直接保存到cam_info里，然后这里第二次调用直接赋值就行了？
	- 工程上是成立的不过没必要，这一步开销小，而且用cam_info只想保存原始描述，派生结果放到Camera，数据划分更清晰
- **思考2**：在 8. 这里计算W2C时，传入的平移和缩放不是之前计算的场景尺度和位移，而是用的默认值，按理来说应该调用的时候传入之前计算的场景尺度： 1/radius [[3dgs_getNerfppNorm]] 说明本代码对场景的不会做自定义缩放平移归一化。
## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_cameras.py\|cameras.py]] |
| 所属类 | [[source/_posts/obsidian/3dgs_code_learn/modules/scene/cameras.py/3dgs_Camera\|Camera]] |
| 命名空间 | - |
| 类型 | - |
