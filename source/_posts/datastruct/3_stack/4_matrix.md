---
title: 3.4 矩阵压缩
date: 2023-08-07 00:00:00
tags: [数据结构]
categories: [数据结构]
comment: false
toc: true
---
#
<!--more-->

![](../../../../themes/yilia/source/img/datastruct/3_stack/matrix/1.png)
![数据结构](/img/datastruct/3_stack/matrix/1.png)

## 3.4.1 对称矩阵
![](../../../../themes/yilia/source/img/datastruct/3_stack/matrix/2.png)
![数据结构](/img/datastruct/3_stack/matrix/2.png)
- 存主对角线+下三角

    $$a_{k}=a_{ij}=a_{ji} (i>=j)$$

    - 列优先

        $$k=\frac{i(i-1)}{2}+j-1$$
    - 行优先

        $$k=\frac{j(j-1)}{2}+i-1$$

## 3.4.2 三角矩阵
![](../../../../themes/yilia/source/img/datastruct/3_stack/matrix/3.png)
![数据结构](/img/datastruct/3_stack/matrix/3.png)
![](../../../../themes/yilia/source/img/datastruct/3_stack/matrix/4.png)
![数据结构](/img/datastruct/3_stack/matrix/4.png)
- 存主对角线+下三角/上三角+c

    $$a_{k}=a_{ij} (i>=j)$$

- 下三角，行优先

    $$k=\left\{\begin{array}{l}{\frac{i(i-1)}{2}+j-1} & {i \geq j} \\\\ {\frac{n(n-1)}{2}} & {i<j}\end{array}\right.$$

- 上三角，行优先
    
    $$k=\left\{\begin{array}{l}{\frac{(i-1)(2n-i+2)}{2}+j-i} & {i \leq j} \\\\ {\frac{n(n-1)}{2}} & {i>j}\end{array}\right.$$

## 3.4.3 三对角矩阵
![](../../../../themes/yilia/source/img/datastruct/3_stack/matrix/5.png)
![数据结构](/img/datastruct/3_stack/matrix/5.png)
![](../../../../themes/yilia/source/img/datastruct/3_stack/matrix/6.png)
![数据结构](/img/datastruct/3_stack/matrix/6.png)
- $$k=2i+j-3$$

## 3.4.4 稀疏矩阵
- 顺序存储
![](../../../../themes/yilia/source/img/datastruct/3_stack/matrix/7.png)
![数据结构](/img/datastruct/3_stack/matrix/7.png)

- 链式存储
![](../../../../themes/yilia/source/img/datastruct/3_stack/matrix/8.png)
![数据结构](/img/datastruct/3_stack/matrix/8.png)

![](../../../../themes/yilia/source/img/datastruct/3_stack/matrix/9.png)
![数据结构](/img/datastruct/3_stack/matrix/9.png)