---
title: 1 stable diffusion项目创建

date: 2024-7-1

tags: [项目,stable diffusion]

categories: [stable diffusion]

comment: true

toc: true
---

#
<!--more-->

# 1- stable diffusion项目创建

## 1. 下载项目


  ```bash
mkdir big_model
cd big_model
git clone https://github.com/lllyasviel/stable-diffusion-webui-forge.git
cd stable-diffusion-webui-forge
  ```



## 2. 安装虚拟环境

- 查看python环境（需要python3.10）：

  ```bash
  python --version
  ```

    - 如果你的python版本不对，可以先在conda安装一个python3.10：

      ```bash
      conda create --name python310 python=3.10
      conda activate python310
      ```
  
- 创建虚拟环境

  ```bash
  python -m venv ./venv
  conda deactivate
  ```

- 使用虚拟环境

   -  windows

      ```bash
      venv\Scripts\activate
      ```

   -  linux

      ```bash
      source ./venv/bin/activate
      ```



## 3. 运行

- 使用虚拟环境：先在stable-diffusion-webui-forge/webui.sh最前面加上：（这样以后就不用每次手动激活虚拟环境了）（windows类似，可能语法不同）

  ```bash
  . venv/bin/activate
  ```

- windows:

  ```bash
  .\webui.bat
  ```

- linux:

  ```bash
  bash webui.sh --xformers
  ```

- 运行后会自动下载一些模型权重，如果中途断了可能是网络问题，重新启动



## 4. 添加扩展

### 4.1 lora

### 4.1.1 下载

- 在big_model目录下载lora项目：

  ```bash
  cd ..
  git clone --recurse-submodules https://github.com/Akegarasu/lora-scripts
  cd lora-scripts
  
  ```

- 安装虚拟环境（确保你的python是3.10）

  ```bash
conda activate python310
  python -m venv ./venv
conda deactivate
  source ./venv/bin/activate
  
  ```

### 4.1.2 运行

- 同理使用虚拟环境，在run_gui.sh最前面加上：

  ```bash
  . venv/bin/activate
  ```

- windows:

  - 国内

    ```bash
    ./install-cn.ps1
    ./run_gui.ps1
    
    ```

  - 国外

    ```bash
    ./install.ps1
    ./run_gui.ps1
    
    ```

- linux:

  ```bash
  bash install.sh
  bash run_gui.sh
  
  ```

- 运行后会自动打开本地端口： [http://127.0.0.1:28000](http://127.0.0.1:28000/) 



## 5. 添加基础模型

- [模型网址](https://huggingface.co/ByteDance/SDXL-Lightning/tree/main)

### 5.1 Lora基础模型

```bash
cd ../stable-diffusion-webui-forge/models/Lora
wget https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_8step_lora.safetensors?download=true
```

- windows powershell下可用：

  ```bash
  Invoke-WebRequest -Uri "https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_8step_lora.safetensors?download=true" -OutFile "sdxl_lightning_8step_lora.safetensors"
  ```

- 在界面刷新即可看到：

  ![](../../../../themes/yilia/source/img/project/stable_diffusion/4.jpg)

  ![](img/project/stable_diffusion/4.jpg)

### 5.2 stable diffusion基础模型

```bash
cd ../stable-diffusion-webui-forge/models/Stable-diffusion
wget https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_8step.safetensors?download=true
```

- windows powershell下可用：

    ```bash
    Invoke-WebRequest -Uri "https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_8step.safetensors?download=true" -OutFile "sdxl_lightning_8step.safetensors"
    ```

