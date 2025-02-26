---
title:  "Prompt-to-Prompt Image Editing with Cross Attention Control论文理解"

date:  2025-2-4 12:28:00

tags:  [视频编辑,论文]

categories:  [论文,视频]

comment:  false

toc:  true


---

#

<!--more-->

- arxiv v1: 2022/8

-  ICLR 2023 
- [论文地址](https://arxiv.org/abs/2208.01626)

- [项目地址](https://github.com/google/prompt-to-prompt)

https://blog.csdn.net/weixin_44966641/article/details/138038795



- prompt2prompt 提出通过替换 [UNet](https://so.csdn.net/so/search?q=UNet&spm=1001.2101.3001.7020) 中的交叉注意力图，在图像编辑过程中根据新的 prompt 语义生图的同时，保持图像整体布局结构不变。从而实现了基于纯文本（不用 mask 等额外信息）的图像编辑。 

