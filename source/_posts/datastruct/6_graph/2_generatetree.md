---
title: 6.2 最小生成树
date: 2023-08-07 00:00:00
tags: [数据结构,图,最小生成树]
categories: [数据结构]
comment: false
toc: true
---
#
<!--more-->

![](../../../../themes/yilia/source/img/datastruct/6_graph/generatetree/1.png)
![数据结构](/img/datastruct/6_graph/generatetree/1.png)


### 6.2-1 定义
- T是连通网的生成树，T中所有边的代价之和称为生成树的代价，代价最小的生成树称为最小生成树
- 最小生成树不唯一
- 生成树的边数=顶点数-1
- 只有连通网才有最小生成树

### 6.2-2 Prim算法
- 从某个顶点出发，选择代价最小的边，然后再选择与之相连的代价最小的边，直到所有顶点都被选中（看顶点）
![](../../../../themes/yilia/source/img/datastruct/6_graph/generatetree/2.png)
![数据结构](/img/datastruct/6_graph/generatetree/2.png)
    - 过程
![](../../../../themes/yilia/source/img/datastruct/6_graph/generatetree/4.png)
![数据结构](/img/datastruct/6_graph/generatetree/4.png)
![](../../../../themes/yilia/source/img/datastruct/6_graph/generatetree/5.png)
![数据结构](/img/datastruct/6_graph/generatetree/5.png)
![](../../../../themes/yilia/source/img/datastruct/6_graph/generatetree/6.png)
![数据结构](/img/datastruct/6_graph/generatetree/6.png)

### 6.2-3 Kruskal算法
- 从代价最小的边开始，依次选择代价更小的边，直到所有顶点都被选中（看边）
![](../../../../themes/yilia/source/img/datastruct/6_graph/generatetree/3.png)
![数据结构](/img/datastruct/6_graph/generatetree/3.png)
    - 过程
    ![](../../../../themes/yilia/source/img/datastruct/6_graph/generatetree/10.png)
![数据结构](/img/datastruct/6_graph/generatetree/10.png)
![](../../../../themes/yilia/source/img/datastruct/6_graph/generatetree/7.png)
![数据结构](/img/datastruct/6_graph/generatetree/7.png)
![](../../../../themes/yilia/source/img/datastruct/6_graph/generatetree/8.png)
![数据结构](/img/datastruct/6_graph/generatetree/8.png)
![](../../../../themes/yilia/source/img/datastruct/6_graph/generatetree/9.png)
![数据结构](/img/datastruct/6_graph/generatetree/9.png)
