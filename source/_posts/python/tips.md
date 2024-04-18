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

  