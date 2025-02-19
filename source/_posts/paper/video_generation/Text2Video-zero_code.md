---
title:  "Text2Video-Zero:  Text-to-Image Diffusion Models are Zero-Shot Video Generators代码理解"

date:  2025-2-17 13:00:00

tags:  [视频编辑,论文]

categories:  [论文,视频]

comment:  false

toc:  true




---

#

<!--more-->

- ICCV 2023
- [论文地址](https://openaccess.thecvf.com/content/ICCV2023/html/Khachatryan_Text2Video-Zero_Text-to-Image_Diffusion_Models_are_Zero-Shot_Video_Generators_ICCV_2023_paper.html)

- [项目地址](https://github.com/Picsart-AI-Research/Text2Video-Zero)

# 0. 启动

```python 
import torch
from model import Model
import os

#可用GPU为1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model = Model(device = "cuda:1", dtype = torch.float16)


prompt = "A horse galloping on a street"
params = {"t0": 44, "t1": 47 , "motion_field_strength_x" : 12, "motion_field_strength_y" : 12, "video_length": 8}

out_path, fps = f"./text2video_{prompt.replace(' ','_')}.mp4", 4
model.process_text2video(prompt, fps = fps, path = out_path, **params)
```

