---
title: Linux入门-0配置
date: 2021-03-01 1200:00
toc: true
tags: [Linux]
categories: [Linux]

---
#

<!-- more -->

## 安装



### 1 虚拟机
### 2 [下载finalshell](https://www.hostbuf.com/downloads/finalshell_install.exe)
### 3 连接finalshell
1. 在linux中右键，打开终端

2. 输入：ifconfig ,找到ens下的inet后面的数字，复制![](./img/linux/install/1.png)

   ![](D:\blog\themes\yilia\source\img\linux\install\1.png)
1. 打开finalshell点击左上文件图标，点击新串口左上带加号的文件图标，点击ssh连接，输入自定义的名称，主机输入刚刚的复制的一串数字，用户名、密码为linux用户的，点击确定
  ![](./img/linux/install/2.png)

  ![](D:\blog\themes\yilia\source\img\linux\install\2.png)
### 4（windows中使用ubuntu）
1. windows中搜索：windows功能，点击：启动或关闭windows功能，将：适用于Linux的子系统勾选并重启

2. 微软应用商店搜索ubuntu并下载，win10还需下载terminal

3. 在terminal中点击向下箭头，选择Ubuntu
  ![](./img/linux/install/3.png)

  ![](D:\blog\themes\yilia\source\img\linux\install\3.png)

4. 如果显示Error: 0x800701bc WSL 2 ?????????????????? 则返回windows命令行输入wsl --update

5. 完成后打开Ubuntu窗口，输入用户名，密码，密码

