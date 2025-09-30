---
title: "3DGS代码运行"
date: 2025-8-28 10:28:00
tags: [3dgs]
categories: [3dgs]
comment: false
toc: true
---

#
<!--more-->

https://www.cnblogs.com/milton/p/18799695

## 1. 代码运行

```shell
git clone git@github.com:graphdeco-inria/gaussian-splatting.git --recursive
cd gaussian-splatting
conda env create --file environment.yml
conda activate gaussian_splatting
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

- 可能会出现环境问题，需要注意cuda与torch版本，我这里用的

  ```shell
  $ pip list
  Package                     Version
  --------------------------- ---------
  Brotli                      1.0.9
  certifi                     2024.8.30
  charset-normalizer          3.4.0
  colorama                    0.4.6
  diff_gaussian_rasterization 0.0.0
  fused_ssim                  0.0.0
  idna                        3.10
  joblib                      1.3.2
  numpy                       1.21.6
  opencv-python               4.12.0.88
  Pillow                      9.2.0
  pip                         22.3.1
  plyfile                     0.8.1
  PySocks                     1.7.1
  requests                    2.32.2
  setuptools                  69.0.3
  simple_knn                  0.0.0
  torch                       1.12.1
  torchaudio                  0.12.1
  torchvision                 0.13.1
  tqdm                        4.67.1
  typing_extensions           4.7.1
  urllib3                     2.2.1
  wheel                       0.42.0
  ```

  ```shell
  $ nvcc --version
  nvcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2022 NVIDIA Corporation
  Built on Thu_Feb_10_18:23:41_PST_2022
  Cuda compilation tools, release 11.6, V11.6.112
  Build cuda_11.6.r11.6/compiler.30978841_0
  ```

  ```shell
  $ nvidia-smi
  +---------------------------------------------------------------------------+
  | NVIDIA-SMI 535.183.01   Driver Version: 535.183.01  CUDA Version: 12.2    |
  ```

  ```shell
  >>> print("Torch CUDA:", torch.version.cuda)
  Torch CUDA: 11.3
  ```

## 2. 训练

- 需要先下载数据集


```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset>[~/code/gaussian-splatting/data/input/tandt_db/playroom] -m <./data/output>[~/code/gaussian-splatting/data/output/playroom]
# python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval # Train with train/test split
```

- **`iteration_7000/`, `iteration_30000/`**
   每隔一段迭代保存一次的快照。里面通常会有 `point_cloud.ply`（或其他名字），记录该时刻的高斯点云参数（位置、颜色、协方差、SH 系数等）。你可以加载这些文件做可视化，或者后续继续训练。

- **`cameras.json`**
   存放相机参数（内参、外参），训练时从 COLMAP / NeRF Synthetic 数据集解析出来的。后续渲染、指标计算都会用它。

- **`cfg_args`**
   保存训练时用的命令行参数/配置，比如学习率、批大小等，方便复现实验。

- **`exposure.json`**
   曝光相关的参数，训练过程中会对颜色/亮度做归一化或补偿。

- **`input.ply`**
   初始点云（通常来自 SfM/稀疏点云），是优化的起点。训练过程会 densify（增密）和 refine 出越来越多的高斯。

## 3. 测试

```shell
# 对于未训练的模型：用预训练模型（或把模型目录拷到另一台机器）时，模型里记住的 source_path 常常是作者/旧机器上的路径（在你机器上不存在）。这时必须用 -s 把你本地的数据集路径告诉 render.py，否则它找不到相机与帧列表，自然也就没法按 train/test 视角去渲染和算指标。
# python render.py -m <path to pre-trained model> -s <path to COLMAP dataset>
# 对于已训练的模型：渲染结果并保存图片
python render.py -m <./data/output>  #会在输出目录下生成选然后的图片（gt / renders)
# 评估
python metrics.py -m <path to pre-trained model> # Compute error metrics on renderings
```

```python
python full_eval.py \   # 一次性对下面三个数据集进行：训练，渲染，测试
  -m360 /path/to/mipnerf360 \
  -tat  /path/to/tanks_and_temples \
  -db   /path/to/deep_blending
```





## 4. 可视化

```shell
cd ~/code/tools/3dgs
git clone https://github.com/antimatter15/splat.git
cd splat
python3 convert.py ~/code/gaussian-splatting/data/output/drjohnson/point_cloud/iteration_30000/point_cloud.ply
cp ~/code/gaussian-splatting/data/output/drjohnson/point_cloud/iteration_30000/point_cloud.ply.splat ./drjohnson.splat
python3 -m http.server 8000
```

- 最后用vscode连接服务器的8000端口，输入`http://localhost:8000/index.html?url=http://localhost:8000/drjohnson.splat`



> - 输入的是多视角图片，3dgs通过colmap得到一个初始点云（ply）文件，然后基于这个ply文件训练3dgs，每个3d高斯按照均值点存储，也是得到一个点云文件（ply）
> - 初始的ply文件格式：`x，y，z，nx，ny，nz，R，G，B` 共9个参数
> - 输出的ply文件格式：`x，y，z，nx，ny，nz，f_dc_0，f_dc_1，f_dc_2，f_rest_0，...，f_rest_44，opacity，scale_0，scale_1，scale_2，rot_0，...，rot_3` 共62个参数