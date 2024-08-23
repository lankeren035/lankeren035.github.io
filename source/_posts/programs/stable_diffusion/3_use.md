---
title: 3-使用代码运行sd+lora

date: 2024-7-9

tags: [项目,stable diffusion]

categories: [项目]

comment: true

toc: true


---

#
<!--more-->



# 3- 使用代码运行sd+lora



## 3.1 下载配置文件/权重

- 方法一：手动下载（更快）
  
- 
  
- 方法二：代码下载（会下载很多没用的权重，会比较慢，不过可以学学用huggingface下载东西）
  - 需要用到的权重：
    - base stable diffusion
    - unet
    - vae
    - control net（可选，看你需不需要）

  

  - 创建download.py：

    ```python
    from diffusers import DiffusionPipeline,AutoencoderKL,StableDiffusionXLPipeline, UNet2DConditionModel
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_8step_unet.safetensors" # Use the correct ckpt for your step setting!
    
    # Load model.
    pipeline = DiffusionPipeline.from_pretrained(base) #下载base的sd
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16) #下载vae
    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16) #从base的unet部分加载json文件
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda")) #下载unet部分
    pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")
    
    
    from diffusers import ControlNetModel
    controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0-mid") #下载control net
    ```

    

  - 运行download.sh：(注意自己修改第一行)

    ```bash
    export HF_HOME="目标路径/diffuser/huggingface" #环境变量，不写则默认：user/cache/
    python test.py
    ```

    

## 3.2

