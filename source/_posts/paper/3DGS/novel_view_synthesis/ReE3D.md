---
title:  "ReE3D: Boosting Novel View Synthesis for Monocular Images using Residual Encoders"

date:  2026-1-22 15:28:00

tags:  [NVS]

categories:  [NVS]

comment:  false

toc:  true



---

#

<!--more-->



## 概述

- 我有两个数据集：1）FFHQ 真实人脸数据集，只包含单张真实图 x（没有 GT 的 latent 和 pose）；2）合成数据集：随机采样一个身份 latent w，再采样多个视角 p1…p10，用**冻结的**预训练 EG3D 渲染得到 $x_{syn,k}=G_{3D}(w,p_k)$，因此这套合成数据天然带标签 $(x_{syn,k}, w, p_k)$，它和 FFHQ 无关。训练时的主路径用 FFHQ：先把真实图 x 送入 pose 提取器得到 $\phi=E_p(x)$，然后初始化 $w_1=w_{avg}、 y_1=x_{avg}$，接着做 n 轮迭代残差反演：每一轮把 x 与当前重建图 $hat y_t$ 拼成 6 通道输入给 residual encoder，得到残差 $\Delta_t$，更新  $w_{t+1}=\hat w_t+\Delta_t$，再用冻结的 EG3D 在同一个 $\phi$ 下渲染出新的重建图 $\hat y_{t+1}=G_{3D}(\hat w_{t+1},\phi)$——所以你问的 $\hat y$就是“上一轮由 EG3D 渲染出来的当前重建结果”，初始是平均图。用最终（或每步）的 $\hat y$ 原图 x 计算重建损失来训练 **residual encoder**（EG3D 始终冻结）。另外每隔 $\alpha$ 个 step，会从合成 multi-view 数据里取同一身份的不同视角样本做几何/多视角正则：一方面让合成图预测的 latent 回归到它的 GT latent w（feature/latent 回归），另一方面让同一身份不同视角得到的预测 latent 相互接近（view-invariant），从而促使 encoder 对同一人脸不同角度输出一致的 latent；这一步通常不做“预测 pose vs GT pose”的损失，合成数据的 GT pose 主要用来先训练/对齐 pose 提取器里的 mapper，使其输出能适配 EG3D。 

