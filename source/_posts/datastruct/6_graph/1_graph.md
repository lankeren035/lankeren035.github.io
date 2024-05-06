---
title: 6.1 图
date: 2023-08-07 00:00:00
tags: [数据结构,图]
categories: [数据结构]
comment: false
toc: true
---
#
<!--more-->

![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/1.png)
![数据结构](/img/datastruct/6_graph/graph/1.png)


## 6.1-1 定义
- 由顶点的有穷非空集合和顶点之间边的集合组成
- 通常表示为：G(V,E)，其中，G表示一个图，V是图G中顶点的集合，E是图G中边的集合。
- 线性表可以空，树可以空，图不可以


## 6.1-2 一些概念
![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/2.png)
![数据结构](/img/datastruct/6_graph/graph/2.png)
![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/3.png)
![数据结构](/img/datastruct/6_graph/graph/3.png)
![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/4.png)
![数据结构](/img/datastruct/6_graph/graph/4.png)
- 生成子图要包含所有顶点
![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/5.png)
![数据结构](/img/datastruct/6_graph/graph/5.png)
- 强连通分量：有向图中

- 连通图的生成树：连通图的极小连通子图

## 6.1-3 图的存储
![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/6.png)
![数据结构](/img/datastruct/6_graph/graph/6.png)
### 6.1-3.1 邻接矩阵
![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/7.png)
![数据结构](/img/datastruct/6_graph/graph/7.png)
- 出度：一行中非0元素的个数
- 入度：一列中非0元素的个数
- 空间复杂度：O(|v|<sup>2</sup>)
- 无向图->对称->矩阵压缩
- 设邻接矩阵A只含0、1，则A<sup>k</sup>中非零元素表示从i到j的长度为k的路径数

### 6.1-3.2 邻接表
![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/8.png)
![数据结构](/img/datastruct/6_graph/graph/8.png)
- 空间复杂度：O(|v|+|e|)

### 6.1-3.3 十字链表（有向图）
![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/9.png)
![数据结构](/img/datastruct/6_graph/graph/9.png)
- 空间复杂度：O(|v|+|e|)
- 如何找到指定顶点的所有出边：沿着绿色的箭头找
- 如何找到指定顶点的所有入边：沿着橙色的箭头找

### 6.1-3.4 邻接多重表（无向图）
![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/10.png)
![数据结构](/img/datastruct/6_graph/graph/10.png)
- 空间复杂度：O(|v|+|e|)
- 删除边、顶点很方便


## 6.1-4 基本操作
|函数|功能|
|:---|:---|
|Adjacent(G,x,y)|判断是否有从x到y的边，无向图只需判断一次|
|Neighbors(G,x)|返回与x邻接的顶点|
|InsertVertex(G,x)|插入顶点|
|DeleteVertex(G,x)|删除顶点|
|AddEdge(G,x,y)|插入边|
|RemoveEdge(G,x,y)|删除边|
|FirstNeighbor(G,x)|返回x的第一个邻接点|
|NextNeighbor(G,x,y)|返回x相对于y的下一个邻接点|
|Get_edge_value(G,x,y)|返回边(x,y)的权值|
|Set_edge_value(G,x,y,v)|设置边(x,y)的权值为v|

|有向图|无向图|
|:---|:---|
|![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/11.png)![数据结构](/img/datastruct/6_graph/graph/11.png)|![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/12.png)![数据结构](/img/datastruct/6_graph/graph/12.png)|
|![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/13.png)![数据结构](/img/datastruct/6_graph/graph/13.png)|![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/14.png)![数据结构](/img/datastruct/6_graph/graph/14.png)|
|![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/15.png)![数据结构](/img/datastruct/6_graph/graph/15.png)|![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/15.png)![数据结构](/img/datastruct/6_graph/graph/15.png)|
|![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/16.png)![数据结构](/img/datastruct/6_graph/graph/16.png)|![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/17.png)![数据结构](/img/datastruct/6_graph/graph/17.png)|
|![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/18.png)![数据结构](/img/datastruct/6_graph/graph/18.png)|![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/18.png)![数据结构](/img/datastruct/6_graph/graph/18.png)|
|![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/19.png)![数据结构](/img/datastruct/6_graph/graph/19.png)|![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/19.png)![数据结构](/img/datastruct/6_graph/graph/19.png)|
|![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/20.png)![数据结构](/img/datastruct/6_graph/graph/20.png)|![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/20.png)![数据结构](/img/datastruct/6_graph/graph/20.png)|

### 6.1-4.1 图的遍历
- 广度优先遍历（BFS）
![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/21.png)
![数据结构](/img/datastruct/6_graph/graph/21.png)

    - 找到所有与v相邻的顶点：FirstNeighbor(G,v)，NextNeighbor(G,v,w)
    - 标记哪个顶点已经访问过：visited[]
    - 用队列保存已经访问过的顶点
![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/22.png)
![数据结构](/img/datastruct/6_graph/graph/22.png)
    - 如果图是非连通的，需要对每个连通分量进行BFS
![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/23.png)
![数据结构](/img/datastruct/6_graph/graph/23.png)
    - 广度优先生成树
![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/24.png)
![数据结构](/img/datastruct/6_graph/graph/24.png)
    - 广度优先生成森林
- 深度优先遍历（DFS）
![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/25.png)
![数据结构](/img/datastruct/6_graph/graph/25.png)
![](../../../../themes/yilia/source/img/datastruct/6_graph/graph/26.png)
![数据结构](/img/datastruct/6_graph/graph/26.png)
    - 深度优先生成树
    - 深度优先生成森林
