---
title: 7.2 二叉排序树
date: 2023-08-07 00:00:00
tags: [数据结构,查找,二叉排序树]
categories: [数据结构]
comment: false
toc: true
---
#
<!--more-->

### 

![](../../../../themes/yilia/source/img/datastruct/7_search/BST/1.png)
![数据结构](/img/datastruct/7_search/BST/1.png)
<!--more-->
- 左<根<右
- 中序遍历：升序排列
- 操作
    - 查找
![](../../../../themes/yilia/source/img/datastruct/7_search/BST/2.png)
![数据结构](/img/datastruct/7_search/BST/2.png)
    - 插入
![](../../../../themes/yilia/source/img/datastruct/7_search/BST/3.png)
![数据结构](/img/datastruct/7_search/BST/3.png)
    - 构造
![](../../../../themes/yilia/source/img/datastruct/7_search/BST/4.png)
![数据结构](/img/datastruct/7_search/BST/4.png)
    - 删除z
        - z是叶子

            删
        - z只有一颗左子树/右子树

            删，子树代替
        - z有两棵子树

            用前驱（左子树最右下）/后继（右子树最左下）代替，删除前驱/后继
- 查找效率分析

    - 最好：O(log<sub>2</sub>n)
    - 最坏：O(n)
