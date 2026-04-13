---
title: "BinningState"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.rasterizer_impl.h.CudaRasterizer.BinningState"
hexo-path:
---
- 存“把高斯复制到各个 tile 后，用于排序和分桶的中间结果”，一些索引
## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| - | - | - |

## 2. 属性

| 属性名                      | 类型        | 来源函数    | 解释                                                               |
| ------------------------ | --------- | ------- | ---------------------------------------------------------------- |
| sorting_size             | size_t    | <class> | 排序算法所需临时空间的字节数                                                   |
| point_list_keys_unsorted | uint64_t* | <class> | 排序前的 key 数组。每个“复制后的高斯实例”对应一个 key，里面编码了： 高位：tile id， 低位：depth key |
| point_list_keys          | uint64_t* | <class> | 排序后的 key 数组                                                      |
| point_list_unsorted      | uint32_t* | <class> | 每个复制实例对应的是哪个原始高斯索引，第 123 个复制实例，其实来自原始高斯 `i`，那这里就存 `i`。           |
| point_list               | uint32_t* | <class> | 排序后的 value 数组                                                    |
| list_sorting_space       | char*     | <class> | 给排序算法用的临时 workspace 地址                                           |

## 3. 方法

| 方法名 | 解释 |
|---|---|
| [[source/_posts/obsidian/3dgs_code_learn/functions/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h/CudaRasterizer/BinningState/3dgs_fromChunk\|fromChunk]] | - |
