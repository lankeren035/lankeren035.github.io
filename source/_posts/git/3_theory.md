---
title: "3- Git理论"

date: 2024-8-21 15:00:00

tags: [Git]

categories: [Git]

comment: true

toc: true

---

#### 

<!--more-->

# 3- Git理论

## 3.1 工作区

- Git本地有三个工作区：
  - 工作目录（working directory)
    - 平时存放项目代码的地方
  - 暂存区（stage/index)
    - 临时存放你的改动，实际上它只是一个文件，保存即将提交到文件列表的信息
  - 资源库（repository/ Git Directory)
    - 仓库区（本地仓库），安全存放数据的位置，这里面有你提交到所有版本的数据。其中HEAD指向最新放入仓库的版本
  - 如果再加上远程的git仓库就可以分为四个工作区
    - 托管代码的服务器，可以认为是你项目组中的一台电脑用于远程数据交换
  
- 文件在这四个区域之间的转换关系：
  
  $$工作目录\underset{\text{git checkout}}{\overset{\text{git add files}}{\rightleftharpoons}}暂存区\underset{\text{git reset}}{\overset{\text{git commit}}{\rightleftharpoons}}本地仓库\underset{\text{git pull}}{\overset{\text{git push}}{\rightleftharpoons}}远程仓库$$

