---
title: "read_next_bytes"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/scene.colmap_loader.py.read_next_bytes"
hexo-path:
---

## 1. 输入

| 参数名                  | 类型          | 解释                                                         |
| -------------------- | ----------- | ---------------------------------------------------------- |
| fid                  | -           | 文件指针                                                       |
| num_bytes            | -           | 读取的字节数                                                     |
| format_char_sequence | -           | 数据类型，比如Q：无符号 8 字节整数，q：有符号 8 字节整数，I：无符号 4 字节整数，i：有符号 4 字节整数 |
| endian_character     | default='<' | 小端读取                                                       |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| - | - |

## 3. 操作逻辑

1. 读取num_bytes个字节
2. 按照指定的大小端/数据格式进行解析
3. 指针后移

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_colmap_loader.py\|colmap_loader.py]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | - |
