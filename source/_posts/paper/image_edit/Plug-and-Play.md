---
title:  "Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation论文理解"

date:  2025-2-26 12:28:00

tags:  [图像编辑,论文]

categories:  [论文]

comment:  false

toc:  true



---

#

<!--more-->

- arxiv v1: 2022/11

-  CVPR 2023 
-  [论文地址](https://openaccess.thecvf.com/content/CVPR2023/html/Tumanyan_Plug-and-Play_Diffusion_Features_for_Text-Driven_Image-to-Image_Translation_CVPR_2023_paper.html)

-  [项目地址](https://github.com/MichalGeyer/plug-and-play)



https://zhuanlan.zhihu.com/p/687368241



- **图像本身的空间信息在 residual block 和 SA 里控制，图像外部注入的信息（比如 prompt）在 CA 层里控制**。所以作者对输入的模板图进行 Unconditional 的 DDIM Inversion，保留空间布局信息到噪声本身里。 

- 所以作者干了一个很暴力的操作：假设 DDIM Inversion 得到的初始噪声为 xt ，我们先把 xt 过一次 Unconditional 的反向过程，可以得到很多中间的参数。给初始噪声 xt 添加 prompt P 生产新图的时候，在每个 timestep 把 Unet 中的某些层的 residual block 输出**强制替换**为之前 Unconditional 时 residual block 的输出。 

- timestep 小的时候，就不注入了，因为语义信息以及基本固定了，接下来就是让它自己生成细节信息。 