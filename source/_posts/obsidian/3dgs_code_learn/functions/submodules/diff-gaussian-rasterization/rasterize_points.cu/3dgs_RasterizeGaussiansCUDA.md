---
title: "RasterizeGaussiansCUDA"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.rasterize_points.cu.RasterizeGaussiansCUDA"
hexo-path:
---

## 1. 输入

| 参数名            | 类型                   | 解释                                                       |
| -------------- | -------------------- | -------------------------------------------------------- |
| background     | const torch::Tensor& | 背景颜色                                                     |
| means3D        | const torch::Tensor& | （P, 3)，每个点的3D坐标                                          |
| colors         | const torch::Tensor& | python预计算的高斯颜色【N, 3】（如有）                                 |
| opacity        | const torch::Tensor& | （P, 1)，每个点的不透明度，初始0.1                                    |
| scales         | const torch::Tensor& | 协方差缩放                                                    |
| rotations      | const torch::Tensor& | 协方差旋转                                                    |
| scale_modifier | const float          | 对所有高斯的尺度做一个统一调节。值大，高斯更“胖”；值小，高斯更“瘦”，代码中默认=1              |
| cov3D_precomp  | const torch::Tensor& | python预计算的3D协方差矩阵 【N, 3, 3】（如有）                          |
| viewmatrix     | const torch::Tensor& | 世界坐标系转换到相机坐标系                                            |
| projmatrix     | const torch::Tensor& | 完整投影矩阵，把3D视锥投影到NDC，x 和 y 轴的范围是 \[-1, 1]，z 轴范围通常是 \[0, 1] |
| tan_fovx       | const float          | 水平视场角（半角）的切线值                                            |
| tan_fovy       | const float          | 垂直视场角（半角）的切线值                                            |
| image_height   | const int            | 图像高度                                                     |
| image_width    | const int            | 图像宽度                                                     |
| sh             | const torch::Tensor& | 球谐系数【p, 16, 3】                                           |
| degree         | const int            | 当前启用的球谐阶数                                                |
| campos         | const torch::Tensor& | 相机中心在世界坐标系中的位置                                           |
| prefiltered    | const bool           | 是否启用预过滤                                                  |
| antialiasing   | const bool           | 是否启用抗锯齿                                                  |
| debug          | const bool           | 是否启用调试模式                                                 |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> | - |

## 3. 操作逻辑

1. 复制means3D的张量配置，改为float类型，用于后面新建的输出张量，确保和输入张量在**同一设备**上
2. 使用上面的张量配置初始化输出张量：
	- 渲染图输出张量out_color【C，H，W】
	- 逆深度图张量out_invdepthptr【1，H，W】
	- 高斯屏幕半径张量radii【P】，每个高斯在当前视角下的 **屏幕空间近似半径（外接矩形）**
3. 定义一个TensorOptions对象，数据类型为byte字节型，用于后面创建缓冲区，不打算把这些 buffer 当有固定结构的 Tensor 用，而是先申请一大块 `uint8/byte` 内存，后面再手动 reinterpret 成各种结构体数组
4. 使用上面的张量配置初始化缓冲区（但不分配实际容量，后面真正需要多少字节，再动态扩容。）：
	- `geomBuffer`：几何缓冲区（里面存了高斯预处理阶段的所有中间结果缓存）[[3dgs_GeometryState]]
	- `binningBuffer`：分桶排序缓冲区（存索引）[[3dgs_BinningState]]
	- `imgBuffer`：图像/像素阶段的辅助缓存（存tile范围索引）[[3dgs_ImageState]]
5. 回调定义缓冲分配函数
	- [[3dgs_resizeFunctional|geomFunc(geomBuffer)]] ：调用该函数时，会以geomBuffer作为一个输入参数
	- [[3dgs_resizeFunctional|binningFunc(binningBuffer)]]
	- [[3dgs_resizeFunctional|imgFunc(imgBuffer)]]
6. 调用CudaRasterizer命名空间下的Rasterizer类的静态成员函数[[source/_posts/obsidian/3dgs_code_learn/functions/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu/CudaRasterizer/Rasterizer/3dgs_forward|forward]]
	- 传递参数：
		- [[3dgs_resizeFunctional|geomFunc]]
		- [[3dgs_resizeFunctional|binningFunc]]
		- [[3dgs_resizeFunctional|imgFunc]]
		- P：点数
		- M：球谐系数总个数（3阶为16个）
		- out_color
		- out_invdepthptr
		- radii
		- 本函数的输入参数
		
		 

          

            W, H, //图像宽高

            means3D.contiguous().data<float>(), //每个点的3D位置

            sh.contiguous().data_ptr<float>(), //每个点的sh系数 [p, 16, 3]

            colors.contiguous().data<float>(), //每个点的预设颜色 [0]

            opacity.contiguous().data<float>(), //每个点的预设不透明度 [p, 1]

            scales.contiguous().data_ptr<float>(), //协方差的缩放因子 [p, 3]

            scale_modifier, //全局尺度系数（LOD/稳定性）？？？   1

            rotations.contiguous().data_ptr<float>(), //协方差的旋转 四元数 [p, 4]

            cov3D_precomp.contiguous().data<float>(), //每个点的协方差矩阵的预计算值 [0]

            viewmatrix.contiguous().data<float>(), //world to view 矩阵 [4,4]

            projmatrix.contiguous().data<float>(), //world到ndc的投影矩阵 [4,4]

            campos.contiguous().data<float>(), //相机在世界坐标系中的位置 [3]

            tan_fovx, //x方向的视场切线值

            tan_fovy, //y方向的视场切线值

            prefiltered, //是否预滤波 false

            out_color.contiguous().data<float>(), //输出的颜色图[C，H，W]

            out_invdepthptr,  //若为 nullptr 表示不写逆深度；否则写入 [1,H,W]

            antialiasing, //抗锯齿开关 false

            radii.contiguous().data<int>(), //每点屏幕半径 [p] 0

            debug);
## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_rasterize_points.cu\|rasterize_points.cu]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | definition |
