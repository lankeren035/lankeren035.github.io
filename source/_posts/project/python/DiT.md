---
title: DiT

date: 2024-6-14

tags: [项目]

categories: [复现]

comment: false

toc: true

---

#

<!--more-->

#  DiT

- 论文：Scalable Diffusion Models with Transformers

- [项目地址]( [facebookresearch/DiT: Official PyTorch Implementation of "Scalable Diffusion Models with Transformers" (github.com)](https://github.com/facebookresearch/DiT) )

## 1. 环境

- 克隆项目

    ```bash
    git clone https://github.com/facebookresearch/DiT.git
    cd DiT
    ```

- 环境配置

  - 尝试（如果后面发现无法使用GPU，可能是你的CUDA版本与这里安装的包版本不对。你可以换一个CUDA版本，我选择重新配环境：将environment.yml里面的版本号去掉）

      ```bash
      conda env create -f environment.yml
      conda activate DiT
      ```

  - 尝试2
  
    ```bash
    conda create -n DiT python=3.9
    conda activate DiT
    conda install pytorch torchvision
    pip install timm diffusers accelerate
    ```
  
    

## 2. 运行

- 运行（会自动下载DiT权重，vae权重）：

  ```bash
  python sample.py --image-size 512 --seed 1
  ```

  - 如果出现`OSError: stabilityai/sd-vae-ft-mse does not appear to have a file named config.json.`，可能是hugging face在加载线上仓库的时候遇到了问题 。可能是你的服务器没开代理。如果服务器没有代理，你可以在本地运行，然后将`C/User/User_name/.cache/huggingface/`复制到服务器的`.cache`中。
  - seed设置了一个固定的种子，伪随机数生成器（PRNG）会根据这个种子值生成一个固定的随机数序列，所以同一个seed生成的图像是一样的。可以将seed改为任意值。
  - image-size也可以设为256

  - 如果save_image出现参数错误，可以将value_range改成range。

## 3 训练

