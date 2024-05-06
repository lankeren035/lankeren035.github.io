---
title: python中os模块的使用
date: 2024-04-1 20:00:00
toc: true
tags: [python]
categories: [python库]



---

#  

<!-- more -->

# python中os模块的使用

- 路径

  - 表示路径：

    `os.path.join()`

  - 输出路径

    `os.listdir()`

  - 创建文件夹

    ```python
    os.makedirs('/', exits_ok=True)
    ```

    

- 重命名：

  `os.rename(os.path.join(), os.path.join())`