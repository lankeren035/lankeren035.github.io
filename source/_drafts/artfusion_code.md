---
title: ArtFusion Controllable Arbitrary Style Transfer using Dual Conditional Latent Diffusion Models代码理解
date: 2024-9-19 10:28:00
tags: [风格迁移,diffusion,深度学习,代码]
categories: [风格迁移,diffusion]
comment: false
toc: true
---
#
<!--more-->

# ArtFusion Controllable Arbitrary Style Transfer using Dual Conditional Latent Diffusion Models代码理解

## 1. 环境配置
- 项目地址:https://github.com/ChenDarYen/ArtFusion

    ```bash
    git clone https://github.com/ChenDarYen/ArtFusion.git
    conda env create -f environment.yaml
    conda activate artfusion
    ```

- 下载模型

    - vae: https://ommer-lab.com/files/latent-diffusion/kl-f16.zip
        - 放到`./checkpoints/vae/kl-f16.ckpt`
    - artfusion: https://1drv.ms/u/s!AuZJlZC8oVPfgWC2O77TUlhIfELG?e=RoSa8a
        - 放到`./checkpoints/artfusion/`
    - 注意: artfusion下载过程容易中断,导致下载下来的模型大小不是3G, 注意检查

- 运行代码

    1. 运行`notebooks/style_transfer.ipynb`

        - 如果出现`numpy.core.multiarray failed to import`, 可能是下载的numpy版本不对(不知道为啥会下错), 重新安装:
            ```bash
            pip uninstall numpy
            conda install numpy=1.23.4 opencv=4.6.0 --override-channels -c defaults -c pytorch
            ```
        - 也有可能会出现torch安装成cpu版本.

## 2. 代码结构

- `notebooks/style_transfer.ipynb`: 推断

- `main.py` : 训练

## 3. 代码理解

### 3.1 推断部分(notebooks/style_transfer.ipynb)
```python
# 1. 参数设置
CFG_PATH = '../configs/kl16_content12.yaml' # 配置文件
CKPT_PATH = '../checkpoints/artfusion/artfusion_r12_step=317673.ckpt' # 模型路径
H = 256
W = 256
DDIM_STEPS = 250
ETA = 1.
SEED = 2023
DEVICE = 'cuda'


# 
import sys
sys.path.append('../')
import os
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from PIL import ImageDraw, ImageFont, Image
import matplotlib.pyplot as plt
from einops import rearrange
from omegaconf import OmegaConf
import albumentations
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
# 为了保证实验的「可复现性」，许多机器学习的代码都会有一个方法叫 seed_everything，这个方法尝试固定随机种子以让一些随机的过程在每一次的运行中产生相同的结果。
seed_everything(SEED)

config = OmegaConf.load(CFG_PATH) # 加载配置文件
config.model.params.ckpt_path = CKPT_PATH # 模型路径
config.model.params.first_stage_config.params.ckpt_path = None
model = instantiate_from_config(config.model) # 实例化模型
model = model.eval().to(DEVICE) # 模型加载到设备上, 并设置为eval模式
```

<details><summary>
- 跳转: instantiate_from_config: &#9660
</summary>

```python
def instantiate_from_config(config): #传入的是config.model
    if not "target" in config: # 检查配置文件中是否有target字段
        raise KeyError("Expected key `target` to instantiate.")
    # 返回两个参数, 第一个是模型, 第二个是模型的参数字典
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False): # 传入的是config.model.target
    module, cls = string.rsplit(".", 1) # 从右边开始分割, 分割一次
    # module = ldm.models.diffusion.dual_cond_ddpm
    # cls = DualCondDDPM
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    # 动态获取对象的属性值
    # importlib.import_module() 用于动态导入模块, 相当于 import 语句的动态版本
    # getattr() 函数用于返回一个对象属性值
    # 这里将ldm.models.diffusion.dual_cond_ddpm模块导入, 并获取这个文件的一个属性: DualCondDDPM类
    # 为什么要这么麻烦?
    return getattr(importlib.import_module(module, package=None), cls)
```
</details>

