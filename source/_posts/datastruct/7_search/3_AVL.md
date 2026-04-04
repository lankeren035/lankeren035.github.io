---
title: 7.3 平衡二叉树
tags: [数据结构,查找,平衡二叉树]
categories: [数据结构]
comment: false
date: 2023-12-12
toc: true
---
#
<!--more-->



![](../../../../theme/yilia/source/img/datastruct/7_search/AVL/1.png)
![数据结构](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/datastruct/7_search/AVL/1.png)
<!--more-->

### 7.3-1 定义
- 左右子树高度差不超过1的二叉排序树，简称AVL树

### 7.3-2 操作
- 插入
    - 1） 插入二叉排序树
    - 2）调整最小不平衡子树A

        ![](../../../../theme/yilia/source/img/datastruct/7_search/AVL/6.png)
        ![数据结构](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/datastruct/7_search/AVL/6.png)

        - LL型：在A的左孩子的左子树插入

        ![](../../../../theme/yilia/source/img/datastruct/7_search/AVL/2.png)
        ![数据结构](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/datastruct/7_search/AVL/2.png)

        - RR型：在A的右孩子的右子树插入

        ![](../../../../theme/yilia/source/img/datastruct/7_search/AVL/3.png)
        ![数据结构](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/datastruct/7_search/AVL/3.png)

        - LR型：在A的左孩子的右子树插入
        
    
        ![](../../../../theme/yilia/source/img/datastruct/7_search/AVL/4.png)
        ![数据结构](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/datastruct/7_search/AVL/4.png)

        - RL型：在A的右孩子的左子树插入
        
        ![](../../../../theme/yilia/source/img/datastruct/7_search/AVL/5.png)
        ![数据结构](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/datastruct/7_search/AVL/5.png)

- 删除
![](../../../../theme/yilia/source/img/datastruct/7_search/AVL/7.png)
![数据结构](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/datastruct/7_search/AVL/7.png)