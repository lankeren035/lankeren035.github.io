---
title: "computeCov3D"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.forward.cu.computeCov3D"
hexo-path:
---
- 根据旋转四元数和缩放向量得到3D协方差
## 1. 输入

| 参数名   | 类型              | 解释                                                         |
| ----- | --------------- | ---------------------------------------------------------- |
| scale | const glm::vec3 | 缩放                                                         |
| mod   | float           | scale_modifier，对所有高斯的尺度做一个统一调节。值大，高斯更“胖”；值小，高斯更“瘦”，代码中默认=1 |
| rot   | const glm::vec4 | 旋转                                                         |
| cov3D | float*          | 输出3D协方差                                                    |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| __device__ void | - |

## 3. 操作逻辑

1. ##todo
```c
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)

{

    // Create scaling matrix

    glm::mat3 S = glm::mat3(1.0f);

    S[0][0] = mod * scale.x;

    S[1][1] = mod * scale.y;

    S[2][2] = mod * scale.z;

  

    // Normalize quaternion to get valid rotation

    glm::vec4 q = rot;// / glm::length(rot);

    float r = q.x;

    float x = q.y;

    float y = q.z;

    float z = q.w;

  

    // Compute rotation matrix from quaternion

    glm::mat3 R = glm::mat3(

        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),

        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),

        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)

    );

  

    glm::mat3 M = S * R;

  

    // Compute 3D world covariance matrix Sigma

    glm::mat3 Sigma = glm::transpose(M) * M;

  

    // Covariance is symmetric, only store upper right

    cov3D[0] = Sigma[0][0];

    cov3D[1] = Sigma[0][1];

    cov3D[2] = Sigma[0][2];

    cov3D[3] = Sigma[1][1];

    cov3D[4] = Sigma[1][2];

    cov3D[5] = Sigma[2][2];

}
```

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_forward.cu|forward.cu]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | definition |
