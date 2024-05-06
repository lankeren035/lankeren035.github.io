---
title: 7.2 文本预处理
date: 2024-5-1 08:00:00
tags: [机器学习,pytorch]
categories: [机器学习]
comment: true
toc: true
---
#  
<!--more-->

## 2 文本预处理
- 将文本作为字符串加载到内存中。
- 将字符串拆分为词元（如单词和字符）。
- 建立一个词表，将拆分的词元映射到数字索引。
- 将文本转换为数字索引序列，方便模型操作。


```python
import collections
import re
from d2l import torch as d2l
```

## 2.1 读取数据集
- 从一个小语料库加载文本，包含30000多个单词
- 将数据集读取到由多条文本行组成的列表中，其中每条文本行都是一个字符串。


```python
#@save
d2l.DATA_HUB['time_machine'] = (
    d2l.DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a'
)

def read_time_machine(): #@save
    '''将时间机器数据集加载到文本的列表中'''
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+',' ', line).strip().lower() for line in lines] #忽略标点符号和大小写

lines = read_time_machine()
print(f'# 文本总行数：{len(lines)}')
print(lines[0])
print(lines[10])
```

    Downloading ..\data\timemachine.txt from http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt...
    # 文本总行数：3221
    the time machine by h g wells
    twinkled and his usually pale face was flushed and animated the
    

## 2.2 词元化
- 将文本行列表（lines）作为输入，列表中的每个元素是一个文本序列（如一条文本行）
- 每个文本序列又被拆分成一个词元列表，词元（token）是文本的基本单位。


```python
def tokenize(lines, token='word'): #@save
    '''将文本行拆分为单词或字符词元'''
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误，未知词元类型：' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])
```

    ['the', 'time', 'machine', 'by', 'h', 'g', 'wells']
    []
    []
    []
    []
    ['i']
    []
    []
    ['the', 'time', 'traveller', 'for', 'so', 'it', 'will', 'be', 'convenient', 'to', 'speak', 'of', 'him']
    ['was', 'expounding', 'a', 'recondite', 'matter', 'to', 'us', 'his', 'grey', 'eyes', 'shone', 'and']
    ['twinkled', 'and', 'his', 'usually', 'pale', 'face', 'was', 'flushed', 'and', 'animated', 'the']
    

## 2.3 词表
- 词元的类型是字符串，而模型需要的输入是数字。我们需要构建一个字典，通常也叫做词表（vocabulary），用来将字符串类型的词元映射到从0开始的数字索引中。
- 先将训练集中的所有文档合并在一起，对它们的唯一词元进行统计，得到的统计结果称之为语料（corpus）。然后根据每个唯一词元的出现频率，为其分配一个数字索引。很少出现的词元通常被移除，这可以降低复杂性。语料库中不存在或已删除的任何词元都将映射到一个特定的未知词元“<unk>”。
- 可以选择增加一个列表，用于保存那些被保留的词元，例如：填充词元（“<pad>”）；序列开始词元（“<bos>”）；序列结束词元（“<eos>”）


```python
class Vocab: #@save
    '''文本词表'''
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens) # Counter({'the': 2261, 'i': 1267})
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True) # [('the', 2261), ('i', 1267)]
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens #['<unk>']
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)} # {'<unk>': 0}
        for token, freq in self._token_freqs:
            if freq< min_freq: #出现频率小于min_freq的词元被忽略
                continue
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) -1
    def __len__(self):
        return len(self.idx_to_token)
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk) # 返回token_to_idx[tokens]，无则返回self.unk
        return [self.__getitem__(token) for token in tokens]
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    @property
    def unk(self): # 未知词元的索引为0
        return 0
    @property
    def token_freqs(self):
        return self._token_freqs
    
def count_corpus(tokens): #@save
    '''统计词元的频率'''
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens) # Counter({'the': 2261, 'i': 1267})
```

- 用语料库来构建词表，然后打印前几个高频词元及其索引。


```python
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])
```

    [('<unk>', 0), ('the', 1), ('i', 2), ('and', 3), ('of', 4), ('a', 5), ('to', 6), ('was', 7), ('in', 8), ('that', 9)]
    

- 将每一条文本行转换成一个数字索引列表。


```python
for i in [0, 10]:
    print('文本：',tokens[i])
    print('索引：',vocab[tokens[i]])
```

    文本： ['the', 'time', 'machine', 'by', 'h', 'g', 'wells']
    索引： [1, 19, 50, 40, 2183, 2184, 400]
    文本： ['twinkled', 'and', 'his', 'usually', 'pale', 'face', 'was', 'flushed', 'and', 'animated', 'the']
    索引： [2186, 3, 25, 1044, 362, 113, 7, 1421, 3, 1045, 1]
    

## 2.4 整合所有功能
- 将所有功能打包到load_corpus_time_machine函数中，该函数返回corpus（词元索引列表）和vocab（时光机器语料库的词表）。
- 为了简化训练，使用字符（而不是单词）实现文本词元化；
- 数据集中的每个文本行不一定是一个句子或一个段落，还可能是一个单词，因此返回的corpus仅处理为单个列表，而不是使用多词元列表构成的一个列表。


```python
def load_corpus_time_machine(max_tokens=-1): #@save
    '''返回时光机数据集的词元索引列表和词表'''
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
print(len(corpus), len(vocab))
```

    170580 28
    
