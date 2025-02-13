---
title:  "AnimateDiff代码复现 "

date:  2025-2-11 1:00:00

tags:  [paper, 复现]

categories:  [复现]

comment:  false

toc:  true

---

#

<!--more-->

## 1. 环境

### 1.1 项目拉取

- #### 过早的文件结束符（EOF）

  - 使用git clone时发现老是无法克隆项目：

    ```shell
    git clone https://github.com/guoyww/AnimateDiff.git
    正克隆到 'AnimateDiff'...
    remote: Enumerating objects: 718, done.
    remote: Counting objects: 100% (38/38), done.
    remote: Compressing objects: 100% (26/26), done.
    fatal: 远端意外挂断了/718), 46.53 MiB | 2.34 MiB/s  
    fatal: 过早的文件结束符（EOF）
    fatal: index-pack 
    ```

  - 网不好或者项目含有多个版本，一起下载空间较大，可以选择下载最新版本

    ```shell
    git clone --depth 1 https://github.com/guoyww/AnimateDiff.git
    ```

    

    

### 1.2 环境配置

- #### ImportError: cannot import name 'cached_download' from 'huggingface_hub'
  
- 运行: `python -m scripts.animate --config configs/prompts/1_animate/1_1_animate_RealisticVision.yaml`
  
- 出现错误：
  
    ```shell
    ImportError: cannot import name 'cached_download' from 'huggingface_hub' 
  ```
  
- 版本问题，项目中使用diffusers库下载了`huggingface_hub`，但是下载的最新版本，这个版本没有`cached_download`
  
- 解决
  
  1. 查看项目的requirement.txt：
  
       ```
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
  
     ![](../../../../themes/yilia/source/img/project/deepseek/local_install/1.png)![](../../../../img/project/deepseek/local_install/1.png)
  
  3. 查看他的`setup.py`找到相关字段：
  
       ```shell
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
  
    7. 换成0.19.3试试，问题解决



- ####  python: symbol lookup error: nvidia/cudnn/lib/libcudnn_cnn_infer.so.8: undefined symbol: _Z20traceback_iretf_implPKcRKN5cudnn16InternalStatus_tEb, version libcudnn_ops_infer.so.8

  - 应该是cudnn的版本兼容问题

  - 使用代码测试：

    ```python
    import torch
    print(torch.__version__)
    
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    ```

    ```shell
    
        raise RuntimeError(
    RuntimeError: cuDNN version incompatibility: PyTorch was compiled  against (8, 9, 2) but found runtime version (8, 4, 0). PyTorch already comes bundled with cuDNN. One option to resolving this error is to ensure PyTorch can find the bundled cuDNN. Looks like your LD_LIBRARY_PATH contains incompatible version of cudnn. Please either remove it from the path or install cudnn (8, 9, 2)
    ```

  - 解决：

    ```shell
    unset LD_LIBRARY_PATH
    ```

  - <details><summary>GPT解释:</summary>
        <p><code>LD_LIBRARY_PATH</code> 是一个环境变量，用于指定 Linux 系统在运行时查找共享库（动态链接库）的路径。这个环境变量的作用主要是告诉操作系统，在哪些目录下寻找需要的共享库文件。</p>
    <h3>主要作用：</h3>
    <ol>
      <li><strong>库文件搜索路径：</strong>
        <p><code>LD_LIBRARY_PATH</code> 指定了操作系统在执行程序时，搜索共享库（如 <code>.so</code> 文件）的位置。当一个程序运行时，它可能依赖于多个共享库，这些库可能存放在不同的文件夹中。<code>LD_LIBRARY_PATH</code> 就是让操作系统知道应该在哪里查找这些库。</p>
      </li>
      <li><strong>覆盖系统默认库路径：</strong>
        <p>如果 <code>LD_LIBRARY_PATH</code> 设置了特定的路径，操作系统就会优先在这些路径下查找所需的库，而不是系统默认的路径（如 <code>/lib</code>, <code>/usr/lib</code> 等）。</p>
      </li>
    </ol>
    <h3>举例说明：</h3>
    <p>假设你安装了 CUDA 和 cuDNN，它们通常会安装在 <code>/usr/local/cuda-11.6/lib64</code> 目录下。如果你有多个版本的 CUDA 和 cuDNN，可能需要用 <code>LD_LIBRARY_PATH</code> 来告诉操作系统使用哪个版本。例如：</p>
    <pre><code>-- 设置 <code>LD_LIBRARY_PATH</code>：
    export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH
    </code></pre>
    <p>这意味着，操作系统会首先在 <code>/usr/local/cuda-11.6/lib64</code> 目录下查找共享库文件。如果这个目录下没有找到，才会继续去其他默认路径查找。</p>
    <h3>为什么你遇到的问题会与 <code>LD_LIBRARY_PATH</code> 相关？</h3>
    <p>在你的情况中，PyTorch 是在 <strong>CUDA 12.1</strong> 和 <strong>cuDNN 8.9.2</strong> 上编译的，但是你的 <code>LD_LIBRARY_PATH</code> 环境变量指向了一个不兼容的 <strong>CUDA 11.6</strong> 目录，这导致系统加载了 <strong>cuDNN 8.4.0</strong>，与 PyTorch 所需的 <strong>cuDNN 8.9.2</strong> 版本不兼容，从而导致了运行时错误。</p>
    <p>当你 <code>unset LD_LIBRARY_PATH</code> 后，系统就不会加载指定的路径，而是使用 <strong>PyTorch 自带的 cuDNN 版本</strong>，解决了版本不匹配的问题。</p>
    <h3>总结：</h3><ul><li><strong><code>LD_LIBRARY_PATH</code> 的作用是告诉操作系统在哪里查找共享库。</strong></li>
      <li><strong>它影响程序运行时使用的库版本，尤其在有多个版本的库时特别重要。</strong></li>
      <li><strong>如果不小心设置了错误的路径（比如指向不兼容的 CUDA/cuDNN 版本），会导致库版本冲突，从而引发错误。</strong></li>
    </ul>
    
    

## 2. 运行

### 2.1 测试

- comunity models

```shell
python -m scripts.animate --config configs/prompts/1_animate/1_1_animate_RealisticVision.yaml
python -m scripts.animate --config configs/prompts/1_animate/1_2_animate_FilmVelvia.yaml
python -m scripts.animate --config configs/prompts/1_animate/1_3_animate_ToonYou.yaml
python -m scripts.animate --config configs/prompts/1_animate/1_4_animate_MajicMix.yaml
python -m scripts.animate --config configs/prompts/1_animate/1_5_animate_RcnzCartoon.yaml
python -m scripts.animate --config configs/prompts/1_animate/1_6_animate_Lyriel.yaml
python -m scripts.animate --config configs/prompts/1_animate/1_7_animate_Tusun.yaml
```

- MotionLoRA control

```shell
python -m scripts.animate --config configs/prompts/2_motionlora/2_motionlora_RealisticVision.yaml
```

- SparseCtrl RGB and sketch

```shell
python -m scripts.animate --config configs/prompts/3_sparsectrl/3_1_sparsectrl_i2v.yaml
python -m scripts.animate --config configs/prompts/3_sparsectrl/3_2_sparsectrl_rgb_RealisticVision.yaml
python -m scripts.animate --config configs/prompts/3_sparsectrl/3_3_sparsectrl_sketch_RealisticVision.yaml
```



### 2.2 训练

