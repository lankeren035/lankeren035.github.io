---
title: nvidia-smi无进程，但是GPU显存被占用了很多
date: 2025-2-13 10:00:00
tags: [debug]
categories: [debug]
comment: true
toc: true


---

#

<!--more-->

## 问题

- 调试python代码时发现调试功能总出错，有时无法连接端口，尝试使用debugpy-old，修改python插件版本等弄好之后再调试，中断调试之后发现虽然没有进程占用GPU，但是显存被占用了很多：

![](../../../themes/yilia/source/img/debug/nvidia-smi/1.png)

![](img/debug/nvidia-smi/1.png)



## 解决

- 查看进程占用

  ```shell
  fuser -v /dev/nvidia*
  ```

  发现有很多进程（看看是不是自己这个用户）

- 杀掉这些进程

  

  

  