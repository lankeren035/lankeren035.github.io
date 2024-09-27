---
title: python tips
date: 2024-03-10 20:00:00
toc: true
tags: [python]
categories: [python]


---

#  

<!-- more -->



- 导入当前文件夹（建了\__init__.py）下的包：

    ```python
    from . import diffusion, unet
    ```

- 数字前自动补0：

  ```python
  name=f'{number:05d}.png'
  ```

  

-   `argparse` 库在处理命令行参数时，自动将命令行中的 `-`（连字符）转换为变量名中的 `_`（下划线） 


- 使用进度条：
  
    ```python
    iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
    ```

- 省略
```python
b, *_, device = *x.shape, x.device
```