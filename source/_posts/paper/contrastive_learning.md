---
title: 对比学习

date: 2024-10-16

tags: [对比学习]

categories: [论文]

comment: true

toc: true

---

#

<!--more-->

# 对比学习

-  有的paper将对比学习称为自监督学习，有的将其称为无监督学习。 有的paper将对比学习称为自监督学习
- 目的： 学习一个编码器，此编码器对同类数据进行相似的编码，并使不同类的数据的编码结果尽可能的不同 

- 通过对比数据对的“相似”或“不同”以获取数据的高阶信息。
  - 比如你可以明显看出来，cat1和cat2是类似的，而cat1和dog1是不同的。



## 1）SimCLRv2

- 例如你想分类猫和狗，你有一堆图片，但是没有标签。
  1. 对图片`x`进行数据增广：黑白（`xi`)、反转(`xj`)...
  2. `xi`和`xj`相互称为`Positive Pairs`，他们来自同一个图片`x`
  3. 来源不同的两张图称为`negative pairs`
  4. 将图片输入网络，希望网络学习到`Positive Pairs`是相似的，`negative pairs`是不相似的。

- 对比学习的三个步骤：
  1. 数据扩增（Data augmentation）
  2. Encoding（将数据转换成representation）
  3. Loss minimization（比较特征向量的相似性）
     - 向量的相似性可以用夹角余弦值



- 损失函数（ infoNCE loss ）： $l_ { i , j } = -log \space \frac{ exp( sim( z_ i, z_ j) / \tau) }{ \sum_ { k=1 } ^ {2N } 1_ { [k \neq i] } exp( sim ( z_i, z_ k ) / \tau ) }$

  - N：batch size，对于N个样本，通过数据增强得到N对正样本对（2N个）。对于正样本对`xi和xj`其他2(N-1)个样本都是负样本。
  - sim计算相似度，可以用余弦相似度。
  - 负样本只出现在分母上，可见要使损失最小，则正样本相似度必须大，负样本相似度必须小。

