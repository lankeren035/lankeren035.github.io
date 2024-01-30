---
title: 6.3 最短路径
date: 2023-08-07 00:00:00
tags: [数据结构,图,最短路径]
categories: [数据结构]
comment: false
toc: true
---
#
<!--more-->

### 

![](../../../../themes/yilia/source/img/datastruct/6_graph/shortestpath/1.png)
![数据结构](/img/datastruct/6_graph/shortestpath/1.png)
<!--more-->
### 6.3-1 单源最短路径
- ##### 1） BFS算法(无权)
    ![](../../../../themes/yilia/source/img/datastruct/6_graph/shortestpath/2.png)
    ![数据结构](/img/datastruct/6_graph/shortestpath/2.png)
    ![](../../../../themes/yilia/source/img/datastruct/6_graph/shortestpath/3.png)
    ![数据结构](/img/datastruct/6_graph/shortestpath/3.png)

- ##### 2）Dijkstra(带权)
    ![](../../../../themes/yilia/source/img/datastruct/6_graph/shortestpath/4.png)
    ![数据结构](/img/datastruct/6_graph/shortestpath/4.png)
    ![](../../../../themes/yilia/source/img/datastruct/6_graph/shortestpath/5.png)
    ![数据结构](/img/datastruct/6_graph/shortestpath/5.png)
    ![](../../../../themes/yilia/source/img/datastruct/6_graph/shortestpath/6.png)
    ![数据结构](/img/datastruct/6_graph/shortestpath/6.png)
    ![](../../../../themes/yilia/source/img/datastruct/6_graph/shortestpath/7.png)
    ![数据结构](/img/datastruct/6_graph/shortestpath/7.png)
    ![](../../../../themes/yilia/source/img/datastruct/6_graph/shortestpath/8.png)
    ![数据结构](/img/datastruct/6_graph/shortestpath/8.png)
    ![](../../../../themes/yilia/source/img/datastruct/6_graph/shortestpath/9.png)
    ![数据结构](/img/datastruct/6_graph/shortestpath/9.png)
    - 时间复杂度：O(|V|<sup>2</sup>)
    - 不可用于有负权边的图

### 6.3-2 各顶点间最短路径
- ##### 1）Floyd算法
    - 动态规划思想：

        - 1）允许在v0中转，求最短路径
        - 2）允许在v0、v1中转，求最短路径
        - 3）允许在v0、v1、v2中转，求最短路径
        - ...
    

    ![](../../../../themes/yilia/source/img/datastruct/6_graph/shortestpath/10.png)
    ![数据结构](/img/datastruct/6_graph/shortestpath/10.png)
    ![](../../../../themes/yilia/source/img/datastruct/6_graph/shortestpath/11.png)
    ![数据结构](/img/datastruct/6_graph/shortestpath/11.png)
    ![](../../../../themes/yilia/source/img/datastruct/6_graph/shortestpath/12.png)
    ![数据结构](/img/datastruct/6_graph/shortestpath/12.png)
    ![](../../../../themes/yilia/source/img/datastruct/6_graph/shortestpath/13.png)
    ![数据结构](/img/datastruct/6_graph/shortestpath/13.png)
    ![](../../../../themes/yilia/source/img/datastruct/6_graph/shortestpath/14.png)
    ![数据结构](/img/datastruct/6_graph/shortestpath/14.png)
    - 时间复杂度：O(|V|<sup>3</sup>)
    - 不能解决负权回路的问题