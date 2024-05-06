---
title: 5.5 并查集
date: 2023-08-07 00:00:00
tags: [数据结构,并查集]
categories: [数据结构]
comment: false
toc: true
---
#
<!--more-->

![](../../../../themes/yilia/source/img/datastruct/5_tree/set/1.png)
![数据结构](/img/datastruct/5_tree/set/1.png)
### 5.5.1 存储结构

- 双亲表示法
![](../../../../themes/yilia/source/img/datastruct/5_tree/set/2.png)
![数据结构](/img/datastruct/5_tree/set/2.png)


### 5.5.2 操作
![](../../../../themes/yilia/source/img/datastruct/5_tree/set/3.png)
![数据结构](/img/datastruct/5_tree/set/3.png)
- 时间复杂度

|操作|最坏时间复杂度|
|:---|:---|
|Find(x)|O(n)|
|Union(x,y)|O(1)|
- 优化union操作
    ![](../../../../themes/yilia/source/img/datastruct/5_tree/set/4.png)
    ![数据结构](/img/datastruct/5_tree/set/4.png)

    - 树高不超过$$\lfloor log_2n \rfloor+1$$
    - find时间复杂度为$$O( log_2n)$$
- 优化find操作

    ![](../../../../themes/yilia/source/img/datastruct/5_tree/set/5.png)
    ![数据结构](/img/datastruct/5_tree/set/5.png)

    - 每次查找时，将路径上的结点都放到根下面


​        