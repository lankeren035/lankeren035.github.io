---
title: 7.4 红黑树
tags: [数据结构]
categories: [数据结构]
date: 2023-12-12
comment: false
toc: true
---
#
<!--more-->

## 9.7 红黑树
![](../../../../themes/yilia/source/img/datastruct/7_search/red/1.png)
![数据结构](/img/datastruct/7_search/red/1.png)
<!--more-->

### 7.4-1 为什么要有红黑树
- 平衡二叉树插入/删除操作效率低，因为调整次数多
![](../../../../themes/yilia/source/img/datastruct/7_search/red/2.png)
![数据结构](/img/datastruct/7_search/red/2.png)

### 7.4-2 定义
- 二叉排序树
- 左根右
- 根叶黑
- 不红红
- 黑路同
![](../../../../themes/yilia/source/img/datastruct/7_search/red/3.png)
![数据结构](/img/datastruct/7_search/red/3.png)

### 7.4-3 性质
- 从根到叶子的最长的可能路径不多于最短的可能路径的两倍长
- 有n个结点的红黑树的高度至多为2log<sub>2</sub>(n+1)
- 若根节点黑高为h，则红黑树的高度至多为2h

### 7.4-4 操作
- 查找
- 插入
![](../../../../themes/yilia/source/img/datastruct/7_search/red/4.png)
![数据结构](/img/datastruct/7_search/red/4.png)
![](../../../../themes/yilia/source/img/datastruct/7_search/red/5.png)
![数据结构](/img/datastruct/7_search/red/5.png)
- 删除
