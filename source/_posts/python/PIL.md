---
title: python中PIL模块的使用
date: 2024-04-1 20:00:00
toc: true
tags: [python]
categories: [python库]
---
#  
<!-- more -->

# python中PIL模块的使用

- 读取图像

  ```python
  from PIL import Image
  img = Image.open('/')
  #转成RGB
  img = Image.open('/').convert('RGB')
  ```

- 缩放图像

  ```python
  from PIL import Image
  img = Image.open('/')
  img_ = img.resize((512,512) , Image.BICUBIC)
  ```

- 保存图像

  ```python
  from PIL import Image
  img = Image.open()
  img.save('/')
  ```

  