---
title: "get_expon_lr_func"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/utils.general_utils.py.get_expon_lr_func"
hexo-path:
---

## 1. 输入

| 参数名            | 类型              | 解释         |
| -------------- | --------------- | ---------- |
| lr_init        | -               | 初始 lr      |
| lr_final       | -               | 最终 lr      |
| lr_delay_steps | default=0       | delay 持续步数 |
| lr_delay_mult  | default=1.0     | delay 倍率   |
| max_steps      | default=1000000 | 总衰减步数      |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| - | - |

## 3. 操作逻辑

1. 指数衰减调度
2. 如果 `lr_delay_steps > 0`，前期还会额外乘一个平滑的 delay 系数

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_general_utils.py\|general_utils.py]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | - |
