---
title: linux中下载kaggle上的数据集

date: 2024-2-24

tags: [kaggle]

categories: [经验]

comment: true

toc: true

---

#
<!--more-->

- 登录kaggle，点击头像，点击设置，下划找到Create New Token，点击下载json文件

![](D:\blog\themes\yilia\source\img\experience\website\1.png)

![](img\experience\website\1.png)

- 在linux中

```
pip install kaggle
```

- 将json文件放到用户根目录下的.kaggle文件夹下

```
cd ~
mkdir .kaggle
cd ~/.kaggle
```

- 下载数据集

![](D:\blog\themes\yilia\source\img\experience\website\2.png)

![](img\experience\website\2.png)

- 如果你使用的远程服务器无法科学上网，需要在国内找到下载地址，然后进行下载，右键正在下载的文件，复制下载地址，然后在服务器上通过`wget 网址`进行下载

![](D:\blog\themes\yilia\source\img\experience\website\3.png)

![](img\experience\website\3.png)

