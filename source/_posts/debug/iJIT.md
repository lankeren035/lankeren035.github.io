---
title: "undefined symbol: iJIT_NotifyEvent"
date: 2025-2-13 10:00:00
tags: [debug]
categories: [debug]
comment: true
toc: true

---

#

<!--more-->

## 问题

eswa

## 原因

这个符号在 MKL 2024.1 中被移除了，然而 PyTorch 是根据旧版本的 MKL 构建的。 MKL 全称 Intel Math Kernel Library， 是由 Intel 公司开发的，专门用于矩阵计算的库。

通过 conda 发布的 PyTorch 二进制文件是动态链接到 MKL 的，所以会遇到这个错误。
通过 pip 发布的 PyTorch 二进制文件是静态链接到 MKL 的（ 可以改用 pip install 来消除 MKL 的这个错误）

## 解决

```
pip install mkl==2024.0.0

```

