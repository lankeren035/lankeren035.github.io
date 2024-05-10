---
title: 8.6 编码器-解码器架构
date: 2024-5-9 10:00:00
tags: [机器学习,pytorch]
categories: [机器学习]
comment: true
toc: true
---
#  
<!--more-->
# 6 编码器-解码器架构

- 机器翻译是序列转换模型的一个核心问题，其输入和输出都是长度可变的序列。为了处理这种类型的输入和输出，我们可以设计一个架构：

    - 第一个组件是一个编码器（encoder）：它接受一个长度可变的序列作为输入，并将其转换为具有固定形状的编码状态
    - 第二个组件是解码器（decoder）：它将固定形状的编码状态映射到长度可变的序列

![](../../../../../../themes/yilia/source/img/deeplearning/code/pytorch/8_moden_rnn/5_encoder/1.png)
![](img/deeplearning/code/pytorch/8_moden_rnn/5_encoder/1.png)

## 6.1 编码器

- 实现一个接口，只指定长度可变的序列作为编码器的输入X。任何继承这个Encoder基类的模型将完成代码实现。


```python
from torch import nn

class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```

## 6.2 解码器

- 解码器接口中，我们新增一个init_state函数，用于将编码器的输出（enc_outputs）转换为编码后的状态。注意，此步骤可能需要额外的输入，例如：输入序列的有效长度，为了逐个地生成长度可变的词元序列，解码器在每个时间步都会将输入（例如：在前一时间步生成的词元）和编码后的状态映射成当前时间步的输出词元。


```python
#@save
class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

## 6.3 合并编码器和解码器

- 在前向传播中，编码器的输出用于生成编码状态，这个状态又被解码器作为其输入的一部分。


```python
#@save
class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基本类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```
