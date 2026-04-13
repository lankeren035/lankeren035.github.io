---
title: "readColmapCameras"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/scene.dataset_readers.py.readColmapCameras"
hexo-path:
---
- 形成数据：\[ [[3dgs_CameraInfo|CameroInfo1]] , [[3dgs_CameraInfo|CameroInfo2]] ]
## 1. 输入

| 参数名                 | 类型  | 解释      |
| ------------------- | --- | ------- |
| cam_extrinsics      | -   | 外参      |
| cam_intrinsics      | -   | 内参      |
| depths_params       | -   | 深度参数    |
| images_folder       | -   | 数据集图片路径 |
| depths_folder       | -   | 深度图图片路径 |
| test_cam_names_list | -   | 测试图片    |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| - | - |

## 3. 操作逻辑

1. 遍历外参条数（图片数量）：
	1) 取出这张图的外参
	2) 根据 `camera_id` 找到对应内参
	3) 把四元数转成旋转矩阵，并转置（对于正定矩阵转置等于求逆，因此这里注意旋转是C2W，而平移是W2C !!） [[source/_posts/obsidian/3dgs_code_learn/functions/scene/colmap_loader.py/3dgs_qvec2rotmat|3dgs_qvec2rotmat]]
	4) 把焦距换算成视场角 `FovX/FovY` [[3dgs_focal2fov]]
	5) 将一条图片信息封装成一个 `CameraInfo` [[3dgs_CameraInfo]]
	6) 放进 `cam_infos` 列表：
		- [CameraInfo1, CameroInfo2]

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_dataset_readers.py|dataset_readers.py]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | - |

## 其他
- 在for循环中可以这样让输出信息控制在一行：
	```
	sys.stdout.write('\r') # sys.stdout是标准输出设备，\r是回车符，表示光标回到当前行的开头
    sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics))) #覆盖掉当前行
	sys.stdout.flush() #刷新缓冲区
	```