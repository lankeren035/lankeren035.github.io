---
title:  "cannot import name cached_download from huggingface_hub "

date:  2025-2-10 1:00:00

tags:  [命令, conda]

categories:  [命令]

comment:  false

toc:  true



---

#

<!--more-->



#### 1. 问题

- 运行代码时出现：

```
ImportError: cannot import name 'cached_download' from 'huggingface_hub' 
```



#### 2. 原因

- 版本问题，项目中使用diffusers库下载了`huggingface_hub`，但是下载的最新版本，这个版本没有`cached_download`



#### 3. 解决

1. 查看项目的requirement.txt：

   ```tex
   torch==2.3.1
   torchvision==0.18.1
   diffusers==0.11.1
   transformers==4.25.1
   xformers==0.0.27
   imageio==2.27.0
   imageio-ffmpeg==0.4.9
   decord==0.6.0
   omegaconf==2.3.0
   gradio==3.36.1
   safetensors
   einops
   wandb
   ```

2. 根据diffusers库的版本去github找源码

   ![](../../../../themes/yilia/source/img/project/deepseek/local_install/1.png)

   ![](img/project/deepseek/local_install/1.png)

3. 查看他的`setup.py`找到相关字段：

   ```python
   "huggingface-hub>=0.10.0",
   ```

4. 重新安装这个包

5. 发现这个问题解决了，但是后面还有问题：

   ```shell
   snapshot_download() got an unexpected keyword argument 'local_dir'
   ```

   

6. 还是版本问题，刚刚安装0.10.0版本时发现有：

   ```shell
   ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
   gradio 3.36.1 requires huggingface-hub>=0.14.0, but you have huggingface-hub 0.10.0 which is incompatible.
   gradio-client 1.7.0 requires huggingface-hub>=0.19.3, but you have huggingface-hub 0.10.0 which is incompatible.
   ```

7. 换成0.19.3试试