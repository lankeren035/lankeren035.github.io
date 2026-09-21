## 1. Attention
### 1.1 单头注意力机制
$$softmax(\frac{QK^T}{\sqrt{d}})V$$
- q【B,H,Lq,dk】
- k【B,H,Lkv,dk】
- v【B,H,Lkv,dk】
- Q 表示当前 token 想寻找什么信息，KK 表示每个 token 用来和查询进行匹配的特征，二者通过 QKTQK^T 计算相似度并得到注意力权重；VV 则表示真正被提取和聚合的信息，最终输出是对各个 VV 按注意力权重做加权求和。因此可以简单记成：**Q 决定“想找什么”，K 决定“谁和你匹配”，V 决定“真正拿走什么信息”。**
```python
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self,dropout_p=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.attention_weights = None
    def forward(self,q,k,v,mask=None):
        d_k = q.size(-1)
        scores = q @ k.transpose(-1,-2) / math.sqrt(d_k)
        if mask is not None:
            # python中没有inf，要先字符串然后再转
            # mask中为0的地方填负无穷，经过后面softmax之后变为0
            scores = scores.masked_fill(mask==0,float('-inf'))
        self.attention_weights = F.softmax(scores, dim=-1)
        return self.dropout(self.attention_weights) @ v
```

### 1.2 自注意力
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_p=0.0, bias=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.to_q = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.to_k = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.to_v = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout_p)
        self.proj_out = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, mask=None):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        scores = q @ k.transpose(-1,-2) / math.sqrt(self.hidden_dim)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -math.inf)
        attention_weights = F.softmax(scores, dim=-1)
        output = self.dropout(attention_weights) @ v
        return self.proj_out(output)
        
```

### 1.3 交叉注意力
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    def __init__(self, q_dim, context_dim, hidden_dim, dropout_p=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.to_q = nn.Linear(q_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(context_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(context_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.proj_out = nn.Linear(hidden_dim, q_dim)
    def forward(self, x, context, mask=None):
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        scores = q @ k.transpose(-1,-2) / math.sqrt(self.hidden_dim)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -math.inf)
        attention_weights = F.softmax(scores, dim=-1)
        output = self.dropout(attention_weights) @ v
        return self.proj_out(output)
    
```

### 1.4 多头注意力
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None
    def forward(self, q, k, v, mask=None):
        B, Lq, Dq = q.shape
        _, Lk, Dk = k.shape
        _, Lv, Dv = v.shape
        
        assert Dq == Dk
        assert Lk == Lv
        assert Dq % self.num_heads == 0
        assert Dv % self.num_heads == 0
        
        d_k = Dk // self.num_heads
        d_v = Dv // self.num_heads
        
        q = q.reshape(B, Lq, self.num_heads, d_k).transpose(1,2)
        k = k.reshape(B, Lk, self.num_heads, d_k).transpose(1,2)
        v = v.reshape(B, Lv, self.num_heads, d_v).transpose(1,2)
        
        scores = q @ k.transpose(-1,-2) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -math.inf)
        self.attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(self.attention_weights)
        output = attention_weights @ v
        output = output.transpose(1,2)
        #transpose()通常不会真的重新排列内存里的数据，它只是修改 tensor 的 `stride`，告诉 PyTorch：现在用另一种顺序解释原来的内存。所以这时候 tensor 往往是 non-contiguous（内存不连续） 的。`view()` 要求数据在内存中的排列能够直接按照目标 shape 解释
        output = output.contiguous().view(B,Lq,Dv)
        return output
```

### 1.5 多头自注意力
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, q_dim, hidden_dim, num_heads, dropout):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads
        self.num_heads = num_heads
        self.attention_weights = None
        
        self.to_q = nn.Linear(q_dim,hidden_dim,bias=False)
        self.to_k = nn.Linear(q_dim, hidden_dim, bias = False)
        self.to_v = nn.Linear(q_dim, hidden_dim, bias= False)
        
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(hidden_dim, q_dim)
        
    def forward(self, x, mask=None):
        B,L,_ = x.shape
        q = self.to_q(x).view(B,L,self.num_heads,self.head_dim).transpose(1,2)
        k = self.to_k(x).view(B,L,self.num_heads,self.head_dim).transpose(1,2)
        v = self.to_v(x).view(B,L,self.num_heads,self.head_dim).transpose(1,2)
        
        scores = q@k.transpose(-1,-2)/math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -math.inf)
        self.attention_weights = F.softmax(scores,dim=-1)
        attention_weights = self.dropout(self.attention_weights)
        output = attention_weights @ v
        output = output.transpose(1,2)
        output = output.contiguous().view(B,L,self.num_heads*self.head_dim)
        return self.proj_out(output)
```

### 1.6 多头交叉注意力
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, q_dim, context_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        assert hidden_dim % num_heads ==0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.attention_weights = None
        
        self.to_q = nn.Linear(q_dim,hidden_dim,bias=False)
        self.to_k = nn.Linear(context_dim,hidden_dim,bias=False)
        self.to_v = nn.Linear(context_dim,hidden_dim,bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(hidden_dim,q_dim)
    def forward(self, x, context, mask=None):
        B,L_q,_ = x.shape
        L_k = context.size(1)
        q = self.to_q(x).view(B,L_q,self.num_heads,self.head_dim).transpose(1,2)
        k = self.to_k(context).view(B,L_k,self.num_heads,self.head_dim).transpose(1,2)
        v = self.to_v(context).view(B,L_k,self.num_heads,self.head_dim).transpose(1,2)
        scores = q @ k.transpose(-1,-2) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -math.inf)
        self.attention_weights = F.softmax(scores,dim=-1)
        attention_weights = self.dropout(self.attention_weights)
        
        output=attention_weights @ v
        output=output.transpose(1,2)
        output=output.contiguous().view(B,L_q,self.num_heads*self.head_dim)
        return self.proj_out(output)
```

### 1.7 多头交叉注意力-图像
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadCrossAttention_img(nn.Module):
    def __init__(self, q_dim, context_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.attention_weights = None
        
        self.proj_in = nn.Conv2d(q_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.to_k = nn.Linear(context_dim, hidden_dim, bias = False)
        self.to_v = nn.Linear(context_dim, hidden_dim, bias = False)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.proj_out = nn.Conv2d(hidden_dim, q_dim, kernel_size=1, stride=1, padding=0)
    def forward(self,x,context,mask=None):
        B,C,H,W = x.shape
        L_q = H*W
        L_k = context.size(1)
        x = self.proj_in(x).view(B,-1,L_q).transpose(1,2)
        q = self.to_q(x).view(B,L_q,self.num_heads,self.head_dim).transpose(1,2)
        k = self.to_k(context).view(B,L_k,self.num_heads,self.head_dim).transpose(1,2)
        v = self.to_v(context).view(B,L_k,self.num_heads,self.head_dim).transpose(1,2)
        
        scores = q @ k.transpose(-1,-2) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -math.inf)
        self.attention_weights = F.softmax(scores,dim=-1)
        attention_weights = self.dropout(self.attention_weights)
        output = attention_weights@v
        output = output.transpose(1,2)
        output = output.contiguous().view(B,L_q,-1)
        output = self.out(output).transpose(1,2)
        output = output.contiguous().view(B,-1,H,W)
        return self.proj_out(output)
        
```
### 1.8 多查询注意力

- 用于减少 K/V 的参数量、计算量和尤其是推理时 KV Cache 的显存占用，同时尽量保留多头 Q 的表达能力。
- 最大的收益在自回归大模型推理。生成第 t 个 token 时，历史 token 的 K/V 会缓存起来。使用多查询注意力，可以压缩group_size倍。
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiQueryCrossAttention(nn.Module):
    def __init__(self, q_dim, context_dim, hidden_dim, num_heads=1, num_groups=1, dropout=0.0):
        super().__init__()
        assert hidden_dim % num_heads ==0
        assert num_heads % num_groups ==0
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.group_size = num_heads // num_groups
        self.head_dim = hidden_dim // num_heads
        self.attention_weights = None
        self.to_q = nn.Linear(q_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(context_dim, self.head_dim*num_groups, bias = False)
        self.to_v = nn.Linear(context_dim, self.head_dim*num_groups, bias = False)
        self.dropout= nn.Dropout(dropout)
        self.proj_out = nn.Linear(hidden_dim, q_dim)
    def forward(self, x, context, mask=None):
        B,L_q,_ = x.shape
        L_k = context.size(1)
        q = self.to_q(x).view(B,L_q,self.num_heads,self.head_dim).transpose(1,2)
        k = self.to_k(context).view(B,L_k,self.num_groups,self.head_dim).transpose(1,2)
        v = self.to_v(context).view(B,L_k,self.num_groups,self.head_dim).transpose(1,2)
        k = k.repeat_interleave(self.group_size,dim=1)
        v = v.repeat_interleave(self.group_size,dim=1)
        scores = q @ k.transpose(-1,-2) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -math.inf)
        self.attention_weights = F.softmax(scores,dim=-1)
        attention_weights = self.dropout(self.attention_weights)
        out = attention_weights @ v
        out = out.transpose(1,2)
        out = out.contiguous().view(B,L_q,self.head_dim*self.num_heads)
        return self.proj_out(out)
```
### 1.9 多头交叉注意力 + KV Cache
- kvcache用到self attention可以用于自回归，累加之前的kv
- kvcache用到crossattention时，可以用于自回归累加kv，也有的context一直不变的情况下，只用于存储kv，以后不用通过tok和tov了

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadCrossAttention_KVCache(nn.Module):
    def __init__(self, q_dim, context_dim, hidden_dim, num_heads,dropout=0.):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.attention_weights = None
        self.to_q = nn.Linear(q_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(context_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(context_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(hidden_dim, q_dim)
        self.cache_k = None
        self.cache_v = None
    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None
    def forward(self, x, context, mask=None, use_cache=False):
        B,L_q,_ = x.shape
        L_k = context.size(1)

        q = self.to_q(x).view(B,L_q,self.num_heads,self.head_dim).transpose(1,2)
        k = self.to_k(context).view(B,L_k,self.num_heads,self.head_dim).transpose(1,2)
        v = self.to_v(context).view(B,L_k,self.num_heads,self.head_dim).transpose(1,2)
        
        if use_cache:
            if self.cache_k is not None:
                k = torch.cat([self.cache_k,k], dim=-2)
                v = torch.cat([self.cache_v,v], dim=-2)                
            self.cache_k = k
            self.cache_v = v
        scores = q @ k.transpose(-1,-2) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -math.inf)
        self.attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(self.attention_weights)
        output = attention_weights @ v
        output = output.transpose(1,2).contiguous().view(B,L_q,self.num_heads*self.head_dim)
        return self.proj_out(output)
```
### 1.10 多头交叉注意力 + LoRA
- crossattention中lora通常加到toq和tov中，这里仅加到toq作为演示
- 什么标准 LoRA 经常写成 `[rank, q_dim]` 和 `[hidden_dim, rank]`。主要是为了和 PyTorch `nn.Linear` 的权重存储方式保持一致。Linear是[out_features, in_features] 的保存形式，实际计算$y=xW^T$ 所以这里也是反向写的维度
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadCrossAttention_LoRA(nn.Module):
    def __init__(self, q_dim, context_dim, hidden_dim, num_heads, dropout=0., rank=0):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.attention_weights = None
        self.rank = rank
        
        self.to_q = nn.Linear(q_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(context_dim, hidden_dim, bias =False)
        self.to_v = nn.Linear(context_dim, hidden_dim, bias = False)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(hidden_dim, q_dim)
        
        for param in [self.to_q, self.to_k, self.to_v, self.proj_out]:
            for p in param.parameters():
                p.requires_grad = False
        self.requires_grad_(False)
        if rank>0:
            self.Q_A = nn.Parameter(torch.zeros([rank,q_dim]))
            self.Q_B = nn.Parameter(torch.zeros([hidden_dim,rank]))
            nn.init.normal_(self.Q_A, mean=0., std=0.02)
    def forward(self, x, context, mask=None):
        B,L_q,_ = x.shape
        L_k = context.size(1)
        q = self.to_q(x) 
        if self.rank > 0:
            q = q+ (x @ self.Q_A.t()) @ self.Q_B.t()
        q=q.view(B,L_q,self.num_heads,self.head_dim).transpose(1,2)
        k = self.to_k(context).view(B,L_k,self.num_heads,self.head_dim).transpose(1,2)
        v = self.to_v(context).view(B,L_k,self.num_heads,self.head_dim).transpose(1,2)
        
        scores = q @ k.transpose(-1,-2) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -math.inf)
        self.attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(self.attention_weights)
        output = attention_weights @ v
        output = output.transpose(1,2).contiguous().view(B,L_q,self.num_heads*self.head_dim)
        return self.proj_out(output)
```

## 2. 基础机器学习
### 2.1 Softmax
$$\hat{y}_ j= \frac{exp(o_ j)}{ \sum_ {i=1}^ qexp(o_ i)}$$
- 求导
- 交叉熵
- 温度
- attention中softmax为什么dim=-1？q@k得到的attention【B,H,Lq,Lk】表示第 `b` 个样本、第 `h` 个头里，第 `i` 个 query 对第 `j` 个 key 的相关性。从里面取出一个【B,H,Lq1,:】表示第 `b` 个样本、第 `h` 个头里，第 `i` 个 query 对所有 key 的相关性。attention想做的是：对于当前这个 query，它应该分别关注这些 key 多少？因此堆key维度做归一化。
- 为什么attention中要除以$\sqrt{d_k}$ ？防止 QK分子部分随维度 dk​ 增大而数值越来越大，导致 Softmax 过于尖锐、梯度变小。假设初始的q和k均值方差：$E[q_i​]=E[k_i​]=0, Var(q_i​)=Var(k_i​)=1$ ，分子部分处理之后按照随机变量和均值方差定义：
    - 均值不变：$E[q_i​_ki​]=E[q_i​]E[k_i​]=0$ 
    - 方差：$$
\begin{aligned}
\operatorname{Var}(q_i k_i)
&= \operatorname{E}[(q_i k_i)^2] - \operatorname{E}^2[q_i k_i] \\
&= \operatorname{E}[q_i^2 k_i^2] - \operatorname{E}^2[q_i]\operatorname{E}^2[k_i] \\
&= \operatorname{E}[q_i^2]\operatorname{E}[k_i^2] - \operatorname{E}^2[q_i]\operatorname{E}^2[k_i] \\
&= \left(\operatorname{Var}(q_i)+\operatorname{E}^2[q_i]\right)
   \left(\operatorname{Var}(k_i)+\operatorname{E}^2[k_i]\right)
   - \operatorname{E}^2[q_i]\operatorname{E}^2[k_i] \\
&= \operatorname{Var}(q_i)\operatorname{Var}(k_i)
 + \operatorname{Var}(q_i)\operatorname{E}^2[k_i]
 + \operatorname{Var}(k_i)\operatorname{E}^2[q_i]
\end{aligned}
$$
    - q矩阵【Lq, head_dim】, k矩阵【Lk, head_dim】
    - 得到的attention【Lq, Lk】的每个元素是head_dim个$q_i$ 与head_dim个$k_i$ 相乘然后相加，因此$Var(attention_i) = Var(\sum_{i\in headdim} q_ik_i) = headdim*Var(q_ik_i)(方差相加)$ 
    - 可以发现，由于矩阵运算的逐项相加带来的标准差为$\sqrt{d_k}$ ，因此把他除去了，需要注意的是attention矩阵的分布与v的分布已经不同了，数量级有差异了。 
```python
import torch
def softmax(x, dim):
    x = x - x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x)
    # sum之后最后一个维度就没了，需要保持
    return x_exp / x_exp.sum(dim=dim, keepdim=True)
```

### 2.2 常用损失
#### 2.2.1 MSE损失
$$MSE = \frac{1}{N}\sum_{i=1}^{N}(x_i - y_i)^2$$
```python
import torch
import torch.nn.functional as F
def mse_loss(x, target):
    assert x.shape == target.shape
    return (x - target).square().mean()

F.mse_loss(x,target,reduction='mean')
F.mse_loss(x,target,reduction='none')
F.mse_loss(x,target,reduction='sum')
```
#### 2.2.2 交叉熵损失
- 主要处理分类问题，假设一个样本有 C 个类别，模型最后输出：$logits = [z_1,z_2,\cdots,z_C]$ 这些 `logits` 还不是概率。
$$L=-\log\frac{e^{z_k}}{\sum_{j=1}^{C}e^{z_j}}$$
```python
import torch
import torch.nn.functional as F
def cross_entropy_loss(logits, labels, reduction='mean'):
    log_probs = F.log_softmax(logits, dim=1)
    target_log_probs = log_probs.gather(dim=1, index=labels.unsqueeze(1))
    loss = -target_log_probs.squeeze(1)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss
```
- gather函数：
    - 假设预测概率：
        [ [-0.2, -2.0, -3.0], 
        \[-1.5, -0.3, -2.2], 
        [-3.0, -1.2, -0.1] ]) 
    - labels =[0, 1, 2])
    - 先将label扩展维度：[
                        [0]
                        [1]
                        [2]
                     ]
     - 然后行不变，在dim=1维度上按index取值。第0行取dim1=0的值，第1行取dim1=1的值。
### 2.3 Linear层
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, in_feature, out_feature, bias=True):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.weight = nn.Parameter(torch.randn(out_feature, infeature))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_feature))
        else:
            self.register_parameter("bias", None)
    def forward(self, x):
        return x @ self.weight.T + self.bias
        return F.linear(x,self.weight,self.bias)
```
### 2.4 梯度下降
#### 2.4.1 Stochastic Gradient Descent
- 拟合$y=2x$ ，模型：$\hat y=Wx$ ，损失$L=(Wx-y)^2$ 
- 损失对模型权重求导：$\frac{\partial L}{\partial W} =2x(Wx-y)$ 
- 更新权重：$w \leftarrow w-\eta \frac{\partial L}{\partial W}$ 
```python
import numpy as np
x_data = np.arange(1,9)
y_data = x_data * 2
w = 1.0
lr=0.01

def forward(x):
    return x*w
def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)**2
def grad(x,y):
    return 2*x*(x*w-y)
for epoch in range(100):
    for x,y in zip(x_data, y_data):
        l = loss(x,y)
        w -= lr*grad(x,y)
    print("epoch=",epoch,'w=',w,'loss=',l)
```

#### 2.4.2 mini-batch Gradient Descent
```python
import numpy as np

x_data = np.arange(1,9)
y_data = x_data*2
w = 1.
lr = 0.01
batch_size = 2
def forward(x):
    return x*w
def loss(x,y):
    y_pred = (forward(x)
    return np.mean(y_pred-y)**2)
def grad(x,y):
    y_pred = (forward(x)
    return np.mean(2*x*(y_pred - y))
for epoch in range(100):
    for iter in range(0,len(x_data),batch_size):

        x = x_data[iter:iter+batch_size]
        y = y_data[iter:iter+batch_size]
        l = loss(x,y)
        g = grad(x,y)
        w -= lr * g
        print('epoch=',epoch,'iter=',iter,'w=',w,'loss=',l)
```
### 2.5 归一化
| **归一化方法**             | **计算均值与方差的轴**                               | **Mean / Var 形状** | **样本间是否隔离**             | **核心适用场景**                                                    |
| --------------------- | ------------------------------------------- | ----------------- | ----------------------- | ------------------------------------------------------------- |
| **BatchNorm (BN)**    | 跨 Batch ($N$) 和空间 ($H, W$)，以**通道 $C$** 为单位  | $(1, C, 1, 1)$    | **否**（同 Batch 样本互相影响）   | **CV 经典网络**<br><br>  <br><br>（CNN 图像分类、目标检测等，需较大 Batch Size）  |
| **LayerNorm (LN)**    | 跨通道 ($C$) 和空间/序列，以**单样本 $N$** 为单位           | $(N, 1, H, W)$    | **是**（样本间完全独立）          | **NLP / 大模型**<br><br>  <br><br>（Transformer、RNN，适配变长序列与单样本推理） |
| **InstanceNorm (IN)** | 仅跨空间 ($H, W$)，以**单样本 $N$ + 单通道 $C$** 为单位    | $(N, C, 1, 1)$    | **事**（样本间、通道间均独立）       | **图像风格处理**<br><br>  <br><br>（风格迁移、StyleGAN、去亮暗对比度噪声）          |
| **GroupNorm (GN)**    | 跨组内通道 ($C/G$) 和空间 ($H, W$)，以**单样本 $N$** 为单位 | $(N, G, 1, 1)$    | **是**（不受 Batch Size 影响） | **小 Batch 高分辨率 CV 任务**<br><br>  <br><br>（如大图语义分割、3D 医疗影像检测）   |
#### 2.5.1 Batch Normalization
$$\begin{aligned}
&\mathrm{BN} ( \mathbf{x})= \boldsymbol{ \gamma} \odot \frac{ \mathbf{x}- \hat{ \boldsymbol{ \mu}}_ \mathcal{B}}{ \hat{ \boldsymbol{ \sigma}}_ \mathcal{B}} + \boldsymbol{ \beta}\\\\
&\hat{\boldsymbol{\mu}}_ {\mathcal{B}} = \frac1{| \mathcal{B}|} \sum_{ \mathbf{x} \in \mathcal{B}} \mathbf{x}, \\\\ 
&\hat{\boldsymbol{ \sigma}}_ { \mathcal{B}}^ 2 = \frac1{| \mathcal{B}|} \sum_ { \mathbf{x} \in \mathcal{B}}( \mathbf{x} - \hat{ \boldsymbol{ \mu}}_ \mathcal{B})^ 2+ \epsilon. 
\end{aligned}$$
- BN的作用：随着网络层数的加深，中间特征的数值范围可能越来越大，导致优化困难。例如经过一百层之后数据尺度放大了100倍，于是这一层不仅要学习：“我要提取什么特征”，还得不断适应：“前一层现在把数值尺度搞成多少了”这两个问题耦合在一起，优化会困难。BN先将数据分布规范到均值0，方差1的分布，然后进行一个缩放，让网络自己学习在已经稳定的坐标系里，我到底需要多大的尺度和偏移。
- BN层**以通道为单位**，去统计整个 Batch 数据的分布。只留channel

```python
import torch
import torch.nn as nn

class BatchNorm(nn.Module):
    def __init__(self, channel, num_dims, momentum=0.1, eps=1e-5):
        # 输入通道，张量维度，动量系数用于控制历史全局统计量与当前 Batch 统计量的更新比例
        # eps放置分母除0
        super().__init__()
        if num_dims == 2:
            shape = (1,channel)
        elif num_dims == 4:
            shape = (1,channel,1,1)
        else:
            raise ValueError('num_dims = 2 or 4')
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        
        # 全局滑动平均均值，不参与反向传播，记录整个训练集均值的指数移动平均。
        # 训练时有batch多个样本算均值，推理时可能batch=1,无法有效计算均值
        self.register_buffer("moving_mean", torch.zeros(shape))
        self.register_buffer("moving_var", torch.ones(shape))
        
        self.momentum = momentum
        self.eps = eps
    def forward(self, x):
        if not self.training:
            x_hat = (x-self.moving_mean) / torch.sqrt(self.moving_var+self.eps)
        else:
            if x.dim() == 2:
                dims = (0,)
            else:
                dims = (0,2,3)
            mean = x.mean(dim=dims, keepdim=True)
            var = x.var(dim=dims, keepdim=True, unbiased=False)
            var_unbias = x.var(dim=dims, keepdim=True, unbiased=True)
            x_hat = (x-mean) / torch.sqrt(var+self.eps)
            
            with torch.no_grad():
                self.moving_mean.mul_(1-self.momentum)
                self.moving_mean.add_(self.momentum * mean)
                self.moving_var.mul_(1-self.momentum)
                self.moving_var.add_(self.momentum * var_unbias)
        return self.gamma* x_hat+ self.beta
```
#### 2.5.2 Layer Normalization
- LN只在channel维度计算均值，只去channel
- BN和LN的gamma和beta都是只在通道维度，因为只有通道维度才有不同语义。
```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, embedding_dim,eps=1e-5):
        super().__init__()
        # gamma和beta是针对通道的，不同通道才有不同的缩放
        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))
        self.eps = eps
    def forward(self,x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x-mean) / torch.sqrt(var+self.eps)
        return self.gamma * x_hat + self.beta
```
#### 2.5.3 Instance Normalization
- 常用于风格迁移，保留batch和channel维度，单张特征图得到一个均值，**一张图片的风格（画风、色彩、笔触浓度）大部分编码在特征图各个通道的均值和方差中**。IN 针对每一张特征图单独归一化，直接抹去原图（内容图）自带的对比度和色调，相当于把原图变成了一张“无风格的白稿”
```python
import torch
import torch.nn as nn

class InstanceNorm(nn.Module):
    def __init__(self, num_features=None, eps=1e-5, affine=False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            if num_features is None:
                raise ValueError()
            self.gamma = nn.Parameter(torch.ones(1,num_features,1,1))
            self.beta = nn.Parameter(torch.zeros(1,num_features,1,1))
    def forward(self, x):
        if x.dim() == 3:
            # N,C,L
            dims = (2,)
        elif x.dim() == 4:
            # N,C,H,W
            dims = (2,3)
        else:
            raise ValueError()
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        x_hat = (x-mean) / torch.sqrt(var+self.eps)
        if self.affine:
            gamma = self.gamma if x.dim() == 4 else self.gamma.squeeze(-1)
            beta = self.beta if x.dim() == 4 else self.beta.squeeze(-1)
            return gamma * x_hat + beta
        return x_hat
```

#### 2.5.4 Group Normalization
- 将LN的通道分组，每组进行归一化。去掉group_dim和H,W
- 注意这是CNN版本的GN，相对于CNN的LN增加了组，CNN的LN不仅仅消除channle还消除HW

```python
import torch
import torch.nn as nn

class GroupNorm(nn.Module):
    def __init__(self, channels, num_groups, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.channels = channels
        self.num_groups = num_groups
        self.group_size = channels // num_groups
        self.gamma = nn.Parameter(torch.ones(1,channels,1,1))
        self.beta = nn.Parameter(torch.zeros(1,channels,1,1))
    def forward(self, x):
        N,C,H,W = x.shape
        x = x.view(N,self.num_groups,self.group_size,H,W)
        mean = x.mean(dim=(2,3,4), keepdim=True)
        var = x.var(dim=(2,3,4),keepdim=True,unbiased=False)
        
        x_hat = (x-mean) / torch.sqrt(var+self.eps)

        x_hat = x_hat.view(N,C,H,W)
        out = self.gamma * x_hat + self.beta
        return out
```
### 2.6 K-Means
- 一种最经典且高效的**无监督学习（Unsupervised Learning）聚类算法**。它的核心目标是将无标签的 $N$ 个数据点划分为 $K$ 个不同的簇（Cluster），使得**同一个簇内的数据点尽量紧密，不同簇之间的数据点尽量稀疏**。
- K-Means 的核心逻辑是最小化**簇内平方和（Within-Cluster Sum of Squares, WCSS）**，也称为组内误差平方和（SSE）：$$J = \sum_{i=1}^k \sum_{x \in C_i} ||x-\mu_i||^2$$
    - $k$：预先指定的聚类簇数。
    - $C_i$：第 $i$ 个簇所包含的所有数据点的集合。
    - $\mu_i$：第 $i$ 个簇的**中心点（Centroid / 质心）**，即该簇内所有点的均值：$\mu_i = \frac{1}{\vert{}C_i\vert{}} \sum_{x \in C_i} x$。
    - 
```python
import torch

def kmeans(X, k=3, max_iters=100, thresh=1e-4):
    # 1. 初始随机化
    n_sample, n_dims = X.shape
    indices = torch.randperm(n_sample)[:k] # 生成0到N-1随机打乱整数序列，取前k个
    centers = X[indices].clone() # 取出k个初始中心
    labels = None
    for iter in range(max_iters):
        # 2. 计算各点到中心点的距离，并根据距离计算labels
        distances = torch.cdist(X,centers,p=2) #2表示欧氏距离
        #distances = torch.norm(X[:,None,:] - centers[None,:,:], dim=2) #n_sample, k
        # dim消除某个维度，每个样本寻找最近的簇中点，得到的是索引
        labels = torch.argmin(distances, dim=1) #n_sample
        new_centers = torch.zeros_like(centers)
        
        #3. 针对每个簇进行中心点更新
        for i in range(k):
            mask = (labels == i)
            if mask.any():
                new_centers[i] = X[mask].mean(dim=0) #遍历这个簇的所有样本，计算平均xyz
            else: #这个簇没分配到任何点
                # 空括号表示得到的是一个随机数，0维，而不是一维向量
                new_centers[i] = X[torch.randint(0,n_sample,())].clone()
        # 4. 判断中心点移动情况，每个簇中心的移动向量的二范数
        center_shift = torch.norm(centers - new_centers, dim=1).max() #k,d -> k ->1
        centers = new_centers
        if center_shift < thresh:
            break
    return centers, labels   #簇中心，每个点属于哪个簇
```

### 2.7 KNN
- 用于分类和回归任务，核心思想是 **“近朱者赤，近墨者黑”**：若一个样本在特征空间中的 $K$ 个最相邻样本大多数属于某一个类别，则该样本也属于这个类别。
- KNN 没有显式的“训练”过程（`fit` 阶段只是将训练数据保存在内存中），所有的计算都推迟到了“预测”阶段（`predict`）。
- 预测步骤：
    - **算距离**：计算测试样本与**所有**训练样本之间的距离（通常使用欧式距离）。
    - **找近邻**：对距离进行升序排序，挑选出距离最近的 $K$ 个训练样本。
    - **多数表决（投票）**：统计这 $K$ 个近邻中出现频率最高的类别标签，作为当前测试样本的预测类别。
```python
import torch
class KNN:
    def __init__(self, k=3):
        self.k = k
        self.x_train = None
        self.y_train = None
    def fit(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train
    def predict(self,x):
        distances = torch.cdist(x,self.x_train,p=2) #n_test,ntrain
        # ntest,k
        _, topk_indices = torch.topk(distances, k=self.k, dim=1, largest=False) #n_test,k
        nearest_k = self.y_train[topk_indices]
        # mode求众数
        y_pred = torch.mode(nearest_k, dim=1).values
        return y_pred
    def predict(self,x):
        #扩展维度会很耗费显存
        distances = torch.norm(self.x_train[None,:,:] - self.x[:,None,:],dim=2)
        k_indices = torch.argsort(distances, dim=1)[:,:self.k]
        y_test = torch.zeros((x.shape[0]))
        for i in range(x.shape[0]):
            k_nearest_labels = [self.y_train[j] for j in k_indices[i]]
            # mostecommn返回一个列表，最多的前几个数第一个【0】取列表第一个
            # 元素是key:value形式，取第一个key
            y_test[i] = Counter(k_nearest_labels).most_common(1)[0][0]
        return y_test
        
# 代替 torch.cdist 的完美降维替代方案 
x2 = torch.sum(x ** 2, dim=1, keepdim=True) # (N_test, 1) 
y2 = (self.x_train**2).sum(1).unsqueeze(0)# (1,N_train) 
xy = x @ self.x_train.T # (N_test, N_train) # 利用展开公式求解距离 
distances = torch.sqrt(x2 + y2.T - 2 * xy)
```

---
## 3. 位置编码
- Attention计算过程中，并不会将tokens的位置信息包含入计算过程，这导致即使tokens的顺序不同，其运算结果也是相同的，在空间信息十分重要的图像模型上，这更是不可接受的，因此基于Vision Transformer 的技术必须使用位置编码对图像token嵌入位置信息。
### 3.1 绝对位置编码
- 给每个位置的词向量添加一个单独的位置向量。最简单的编码是二进制编码。
#### 3.1.1 可学习的位置编码
- 直接将位置编码设置为可训练参数，如max_tokens = 512 ，维度为768，那么就初始化一个512\*768的矩阵，然后参与训练即可。（类似将类别变成一个embedding可学习向量）, 训练的时候可学，测试的时候冻结
- 存在的问题：
    - 没有外推性
    - 相对位置关系不明确
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class learnable_positional_encoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.PE = nn.Parameter(torch.randn(1,max_len, d_model))
    def forward(self,x):
        x = x + self.PE[:,:x.size(1),:]
        return x
```
#### 3.1.2 三角位置编码
##### 1）定义
- 也称为Sinusoidal位置编码，是Transformer论文所提出的一种位置编码：$$\begin{aligned} &p_{k,2i} = sin \left( \frac{k}{10000^{2i/d}} \right) \\\\ & p_{k,2i+1} = cos \left( \frac{k}{10000^{2i/d}} \right) \end{aligned}$$
    - k表示有K个位置需要表示，当前位置的索引为k
    - d表示将每个位置映射成为d维的向量
    - i 表示第几组 sin/cos 频率
    - 例如我有3个位置，我希望把位置表达为4维向量（每个位置表达成2种频率）：$$P=
\begin{bmatrix}
\sin(0) & \cos(0) & \sin(0/100) & \cos(0/100) \\
\sin(1) & \cos(1) & \sin(1/100) & \cos(1/100) \\
\sin(2) & \cos(2) & \sin(2/100) & \cos(2/100)
\end{bmatrix}$$
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class sin_positional_encoding(nn.Module):
    def __init__(self. max_len, d_model):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0,max_len).unsqueeze(1)
        div_term = torch.exp(
            -math.log(10000) * torch.arange(0, d_model, 2)/d_model
        )
        angles = position * div_term
        pe[:,0::2] = torch.sin(angles)
        pe[:,1::2] = torch.cos(angles)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    def forward(slef,x):
        x = x+self.pe[:,:x.size(1),:]
```
##### 2）优势
- 两个位置编码间存在旋转关系，定义$$\omega_i = \frac{1}{10000^{2i/d}}$$ ，位置k：$$p_k^{(i)} = \begin{bmatrix} \sin(k \omega_i) \\ \cos(k \omega_i) \end{bmatrix}$$​，位置k+m做了一个移动：$$p_{k+m}^{(i)} = \begin{bmatrix} \sin((k+m)\omega_i) \\ \cos((k+m)\omega_i) \end{bmatrix}$$ ，做三角函数展开：$$p_{k+m}^{(i)} = \begin{bmatrix} \cos(m \omega_i) & \sin(m \omega_i) \\ -\sin(m \omega_i) & \cos(m \omega_i) \end{bmatrix} p_k^{(i)}$$ 所以，原始位移移动了m相当于位置编码乘以一个旋转矩阵（相对位置信息），旋转角度与位移有关：$$\theta_i = m \omega_i = \frac{m}{10000^{2i/d}}$$
##### 3）缺陷
- Sinusoidal 绝对位置编码本身蕴含相对位置关系，但将其与 token 相加并经过 Attention 的 WQ​,WK​ 投影后，这种“只依赖相对距离”的结构无法被显式保证。例如：
    1. 原始位置编码有：$$ p_i^T p_j = \sum_k \cos \left( (i-j)\omega_k \right) $$它只依赖相对距离：$p_i^T p_j = f(i-j)$ 
    2. 但是在attention中，位置编码与 token 相加：$$ h_i = x_i + p_i $$
    3. 拿token去算attention：$$ q_i k_j^T = (x_i+p_i) W_Q W_K^T (x_j+p_j)^T $$ 展开：$$ q_i k_j^T = x_i W_Q W_K^T x_j^T + x_i W_Q W_K^T p_j^T + p_i W_Q W_K^T x_j^T + p_i W_Q W_K^T p_j^T $$其中纯位置项变成：$$ p_i W_Q W_K^T p_j^T $$经过 (W_Q,W_K) 后，一般无法保证：$$ p_i W_Q W_K^T p_j^T = f(i-j) $$也就是说**原本只由相对距离决定的结构，不再被严格保留**。
##### 4）特性
- 周期性（正弦余弦本身具备的周期性）每个维度具有周期性，但整个多频率编码不是简单的一个固定短周期循环。
- 连续性（相邻位置的编码变化较小，有利于捕捉局部语义与上下文关系）
- 长距离衰减性：近距离位置通常具有较高的位置编码相似度；随着距离增大，不同频率逐渐产生相位差，使整体相似度通常降低或发生振荡，但并不存在严格的单调长距离衰减。
- 唯一性（每个位置上的编码是唯一的）
### 3.2 相对位置编码
- 针对绝对位置编码的缺陷，有没有办法让attention计算之后仍然可以**显式保留只由相对距离 i−j 决定的位置关系**？
#### 3.2.1 基本思想
- 对于位置 (i) 和位置 (j)，其相对位置定义为：$\Delta_{ij} = j - i$ 。
- Attention 原本只根据 $q_i$ 与$k_j$ 的内容相似度计算权重：​$a_{ij} = \frac{q_i k_j^T}{\sqrt{d}}$ 。
- 相对位置编码的核心就是：**在计算 token (i) 对 token (j) 的 Attention 时，显式加入二者的相对位置 (j-i)**。
#### 3.2.2 Relative Position Bias
- 一种最简单的方法是给每个相对距离设置一个可学习的 bias：$$a_{ij} = \frac{q_i k_j^T}{\sqrt{d}} + b_{j-i}$$
    -  $b_{j-i}$ 是一个距离**标量**包含相对位置信息（i-j）；这样就可以保证即使两个 token 内容完全相同，但距离不同，其 Attention score 也会不同。
- 如果直接为所有可能的相对距离都学习一个参数，序列很长时参数量会增加。因此一种简单方式是限制最大相对距离：$\Delta_{ij} = \operatorname{clip}(j-i,,-K,,K)$ ，这样只需要学习 (2K+1) 个相对位置参数。
##### 1）T5
- 不为每一个相对距离单独设置参数，而是首先计算：r=j−i，再把相对距离映射到不同的 bucket：b=bucket(j−i)。划分逻辑：
    - 近距离划分更细，远距离划分更粗：$$\begin{aligned} & 0 \rightarrow b_0,\qquad 1 \rightarrow b_1,\qquad 2 \rightarrow b_2, \\\\ & \cdots \\\\ & 8 \sim 11 \rightarrow b_8,\\\\ &  12 \sim 15 \rightarrow b_9,\qquad \end{aligned}$$
- 每个 Attention Head、每个 bucket 对应一个可学习的偏置（标量）：$$a_{ij}^{(h)} = \frac{q_i^{(h)} {k_j^{(h)}}^T}{\sqrt{d_h}} + B_{h,\operatorname{bucket}(j-i)}$$
##### 2）ALiBi
- 直接人为构造一个线性偏置，不学习$b_{j-i}$ ：$$a_{ij}^{(h)} = \frac{q_i^{(h)} {k_j{(h)}}T}{\sqrt{d_h}} - m_h |i-j|$$
    - $m_h$ 是第 h 个 Attention Head 对应的固定斜率。距离越远，Attention score 被减得越多。$m_h$ 大：远距离衰减快，更关注局部；$m_h$小：远距离衰减慢，更容易关注长距离 token。
#### 3.2.3 Relative Position Embedding
- 前面每个距离表达成一个标量，这里每个距离表达成一个向量（有通道维度了），然后让它直接参与 Query 和 Key 的相关性计算：$$\begin{aligned} a_{ij} &= \frac{q_i \left( k_j + r_{j-i} \right)^T}{\sqrt{d}} \\\\ &= \frac{q_i k_j^T}{\sqrt{d}} + \frac{q_i r_{j-i}^T}{\sqrt{d}} \end{aligned}$$其中第二项表示 Query 对相对位置 (j-i) 的相关性。因此 Relative Position Embedding 比标量 bias 表达能力更强，它不只是规定“这个距离加多少分”，而是让**不同 Query 可以根据自身内容，对同一个相对位置产生不同响应**。
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class relative_positional_encoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        # embedding 层就是一个可学习的矩阵，输入的是查询下标
        self.pe = nn.Embedding(2*max_len-1, d_model)
        self.max_len = max_len
    def forward(self, len_q, len_k):
        positions_q = torch.arange(len_q)
        positions_k = torch.arange(len_k)
        relative_pos = positions_q[:,None] - positions_k[None,:] + self.maxlen - 1
        relative_pos_emb = self.pe(relative_pos)
        return relative_pos_emb

class CrossAttention(nn.Module):
    def __init__(slef, q_dim, context_dim, hidden_dim, dropout):
        self.hidden_dim = hidden_dim
        self.attention_weights = None
        
        self.rpe = relative_positional_encoding(max_len=512, d_model=hidden_dim)
        self.to_q = nn.Linear(q_dim, hidden_dim, bias-False)
        self.to_v = nn.Linear(context_dim,hidden_dim,bias=False)
        self.to_k = nn.Linear(context_dim,hidden_dim,bias-False)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(hidden_dim,q_dim)
        
    def forward(self, x, context, mask=None):
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        score_content = q@k.transpose(-1,-2)
        r = self.rpe(q.size(1), k.size(1))
        # nqd,qkd 对齐后 nq1d,1qkd, 点积会在最后一个维度求和：nqk
        # score_pos = q.transpose(0,1)@r.transpose(1,2)
        # score_pos = score_pos.transpose(0,1)
        score_pos = torch.einsum(
            'nqd,qkd->nqk',
            q,
            r
        )
        score = (score_content+score_pos) / math.sqrt(self.hidden_dim)
        if mask is not None:
            score = scores.masked_fill(mask==0, -math.inf)
        self.attention_weights = F.softmax(score, dim=-1)
        out = self.dropout(self.attention_weights) @ v
        return self.proj_out(out)
        
```
### 3.3 旋转位置编码
- 把 token 在序列/空间位置上的平移，编码成 Q、K 在通道二维子空间里的旋转。
#### 3.3.1 基本思想
- 根据 token 的绝对位置，对 Q 和 K做旋转，使旋转后的 $QK^T$ 天然只显式依赖相对位置 j-i。原始：$$q_i=x_iW_Q,\qquad k_j=x_jW_K$$RoPE 处理后：$$\tilde q_i=R_iq_i,\qquad \tilde k_j=R_jk_j$$其中 $R_i,R_j$ 是由位置 (i,j) 决定的旋转矩阵。
#### 3.3.2 二维旋转
1. 假设token的通道数是d，仿照三角编码，将两个通道当作一个频率，因此可以定义$m=\frac{d}{2}$ 组频率：$$\omega_m=\frac{1}{10000^{2m/d}}$$
2. 将每个位置定义成一个旋转角度（从0旋转到i）:$$\theta_{i,m}=i\omega_m$$对应的旋转矩阵：$$R_{i,m}= \begin{bmatrix} \cos(i\omega_m) & -\sin(i\omega_m)\ \\\\ \sin(i\omega_m) & \cos(i\omega_m) \end{bmatrix}$$于是原始二维（d=2） Query经过位置i对应的旋转之后：$$\begin{bmatrix} \tilde q_{i,2m}\ \\\\ \tilde q_{i,2m+1} \end{bmatrix} = \begin{bmatrix} \cos(i\omega_m) & -\sin(i\omega_m)\ \\\\ \sin(i\omega_m) & \cos(i\omega_m) \end{bmatrix} \begin{bmatrix} q_{i,2m} \\\\ \ q_{i,2m+1} \end{bmatrix}$$
3. 更一般的形式（通道=d）：$$\begin{bmatrix} \tilde q_{i,0} \\ \tilde q_{i,1} \\ \tilde q_{i,2} \\ \tilde q_{i,3} \\ \vdots \\ \tilde q_{i,d_h-2} \\ \tilde q_{i,d_h-1} \end{bmatrix} = \begin{bmatrix} \cos(i\omega_0) & -\sin(i\omega_0) & 0 & 0 & \cdots & 0 & 0 \\ \sin(i\omega_0) & \cos(i\omega_0) & 0 & 0 & \cdots & 0 & 0 \\ 0 & 0 & \cos(i\omega_1) & -\sin(i\omega_1) & \cdots & 0 & 0 \\ 0 & 0 & \sin(i\omega_1) & \cos(i\omega_1) & \cdots & 0 & 0 \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & 0 & 0 & \cdots & \cos\left(i\omega_{\frac{d_h}{2}-1}\right) & -\sin\left(i\omega_{\frac{d_h}{2}-1}\right) \\ 0 & 0 & 0 & 0 & \cdots & \sin\left(i\omega_{\frac{d_h}{2}-1}\right) & \cos\left(i\omega_{\frac{d_h}{2}-1}\right) \end{bmatrix} \begin{bmatrix} q_{i,0} \\ q_{i,1} \\ q_{i,2} \\ q_{i,3} \\ \vdots \\ q_{i,d_h-2} \\ q_{i,d_h-1} \end{bmatrix}$$
#### 3.3.3 保留相对位置
- 旋转后的位置i和位置j的token做Query 和 Key 的内积：$\tilde q_i^T\tilde k_j=(R_iq_i)^T(R_jk_j)=q_i^TR_i^TR_jk_j$ 
- 由于选旋转矩阵满足：$$\begin{aligned} &R_i^T=R_{-i} \\\\&R_{-i}R_j=R_{j-i}  \end{aligned}$$
- 所以$$\boxed{\tilde q_i^T\tilde k_j=q_i^TR_{j-i}k_j}$$ 而中间的旋转矩阵只依赖j-i。
- 为什么只旋转 Q 和 K？
    - Attention 权重由qk确定，位置信息主要需要影响的是：**token (i) 应该关注 token (j) 的程度**，所以只需要作用在 (Q,K) 上。而v只是在聚合。算关系的在qk
```python
import torch
import torch.nn as nn
import math

class rotary_positional_encoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            -math.log(10000)*torch.arange(0,d_model,2)/d_model    
        )
        # [Lmax,1] * [D/2], 广播后 [Lmax,1] * [1, D/2]
        # 输出[Lmax,D/2] , 每个位置生成多个频率的角度
        angles = position*div_term
        # [Lmax,D] ，每个频率相邻复制一份，因为每两个通道用一个频率
        cos_pos = torch.cos(angles).repeat_interleave(2,dim=-1)
        sin_pos = torch.sin(agnles).repeat_interleave(2,dim=-1)
        self.register_buffer('cos_pos',cos_pos)
        self.register_buffer('sin_pos',sin_pos)
    def forward(slef,x):
        L = x.size(-2)
        cos_pos = self.cos_pos[:L]
        sin_pos = self.sin_pos[:L]
        # 将x的相邻通道交换，并将奇数通道变负数
        x_rot = torch.stack( # 先取出奇数通道，然后取出偶数通道
            [-x[...,1::2], x[...,0::2]],
            dim=-1
        ).reshape_as(x)
        # x和xrot互补，x取0通道时，xrot就取1通道，x取1时xrot就取0
        x = x*cos_pos + x_rot*sin_pos
        return x
```
## 4. 神经网络
### 4.1 反向传播
```python
import numpy as np

# 初始化全连接
def init_parameter(channels):
    # 假设 indim, hiddendim, outdim ，网络就有两层
    # 第i层的W输入维度是 channels[i-1]，输出维度是channels[i]
    # 因为矩阵乘法是W取转置。
    parameters = {}
    for i in range(1,len(channels)):
        parameters['w'+str(i)] = np.random.random([channels[i],channels[i-1]])
        parameters['b'+str(i)] = np.zeros([1,channels[i]])
    return parameters
    
def sigmoid(z):
    return 1. / (1+np.exp(-z))
def sigmoid_dz(z): #sigmoid求导
    return sigmoid(z)*(1-sigmoid(z))
def relu_dz(z):
    return z>0
def tanh_dz(z):
    return 1-np.tanh(z)**2
def forward(x, parameters):
    a = []
    z = []
    caches = {}
    a.append(x) # 原始输入x
    z.append(x) # 这一项没必要，为了下标对齐。
    layers = len(parameters)//2 # 每层有w和b两个矩阵
    for i in range(1, layers+1):
        # 上一层的输入a @ W^T + b
        z_tmp = a[-1].dot(parameters['w'+str(i)].T) + parameters['b'+str(i)]
        z.append(z_tmp) #每一层的输出保存，激活前
        if i==layers:
            a.append(z_tmp) #最后一层不激活
        else:
            a.append(sigmoid(z_tmp)) #激活后作为下一层的输入
    caches['z']=z
    caches['a']=a
    return caches, a[-1]
def cal_loss(x,y):
    return np.mean(np.square(x-y))
def backward(parameters, caches, x,y):
    layers = len(parameters) //2 
    grads = {}
    # torch用numel()
    dz = 2. * (x-y) / y.size #numpy用size表示所有维度相乘
    for i in range(layers,0,-1):
        a_prev = caches['a'][i-1]
        grads['w'+str(i)] = dz.T@a_prev
        grads['b'+str(i)] = np.sum(
            dz,
            axis = 0,
            keepdims=True
        )
        if i>1:
            da_prev = dz @ parameters['w'+str(i)]
            dz = (
                da_prev * sigmoid_dz(caches['z'][i-1])
            )
    return grads

def update_grad(parameters, grads, lr):
    layers = len(parameters) // 2
    for i in range(1,layers+1):
        parameters['w'+str(i)] -= lr*grads['w'+str(i)]
        parameters['b'+str(i)] -= lr*grads['b'+str(i)]
    return parameters

x = np.arange(0.,1.,0.01)
y = 20*np.sin(2*np.pi*x)
x = x.reshape(100,1)
y = y.reshape(100,1)
parameters = init_parameters([1,20,1])
for i in range(5000):
    caches,output = forward(x,parameters)
    grads = backward(parameters, caches, output, y)
    parameters = update_grad(parameters, grads,lr=0.1)
    if i %100==0:
        print(cal_loss(output,y))
```
### 4.2 二维卷积算法
```python
import torch
import torch.nn.functional as F
def conv2d(img, kernels, bias, stride=1, padding=0):
    N,in_ch,H,W=img.shape
    out_ch,in_ch,kh,kw = kernels.shape
    # 每in_ch个二维矩阵组成一组，得到一个输出维度
    p = padding
    if p:
        img = F.pad(img,(p,p,p,p)) #l,r,t,b，默认补0
    # // stride表示可以走多少步  +1表示初始位置
    out_h = (H + 2*padding - kh) // stride +1
    out_w = (W + 2*padding - kw) // stride +1
    
    outputs = torch.zeros([N,out_ch,out_h,out_w])
    # 枚举所有batch,通道，像素
    for n in range(N):
        for out in range(out_ch):
            for i in range(in_ch):
                for h in range(out_h):
                    for w in range(out_w):
                        region = img[n,i,h*stride:(h*stride+kh),w*stride:(w*stride+kw)]
                        # 这一步+=是针对遍历所有输入通道，所有输入通道的结果求和得到一个输出通道的结果。
                        outputs[n][out][h][w] += torch.sum(region*kernels[out][i]) 
    outputs += bias.view(1,-1,1,1)
    return outputs

def conv2d_fast(img, kernels, bias, stride=1, padding=0):
    N,C,H,W = img.shape
    out_ch, in_ch, kh, kw = kernels.shape
    assert C == in_ch
    
    out_h = (H+2*padding - kh) // stride+1
    out_w = (W+2*padding - kw) // stride+1
    # unfold把卷积窗口摊平
    # 原本一个卷积窗口要做C*kh*kw次运算得到一个输出通道的一个像素，以此为基本单位拆分窗口
    # 根据卷积核情况可以把图片拆分成：N,C_in*kh*kw,out_h*out_w
    # batch维度不变，第二个维度相当于将一个窗口内空间维度上的kw*kh个像素以及C个通道放到一条向量上，组成一个通道。最后一个维度就是输出结果有几个卷积窗口
    img_unfold = F.unfold(img,(kh,kw),padding=padding,stride=stride)
    # 卷积核同样摊平：out_ch, in_ch*kw*kh
    kernels_flat = kernels.view(out_ch,-1)
    # 二维矩阵的.()相当于transpose(0,1)
    output_flat = img_unfold.transpose(1,2).matmul(kernels_flat.t()).transpose(1,2).contiguous()
    # N, out_ch, out_h*out_w
    output = output_flat.view(N,out_ch,out_h,out_w)
    output += bias.view(1,-1,1,1)
    return output
    
inputs = torch.randn(2,2,5,5) #B,C,H,W
kernel = torch.randn(3,2,3,3) #out_c, in_c,kh,kw
bias = torch.randn(3) #out_c
res = conv2d(inputs, kernels=kernel,bias=bias,stride=2,padding=1)
print(res)
res2 = F.conv2d(input, kernel, bias=bias, stride=2, padding=1)
print(res2)
# 判断两个张量的值是否近似相等
print(torch.allclose(res, res2))
```
### 4.3 CNN
```python
import torch
import torch.nn as nn
import torch.utils.data as Data

torch.manual_seed(1) 设置随机种子，用于复现

# 超参数
EPOCH= 1
LR= 0.001
BATCH_SIZE=50
DOWNLOAD_MNIST= True

# 1. 划分数据集
train_data = torchvision.datasets.MNIST(
    root = './MNIST/',
    train = True,
    transform = torchvision.transforms.ToTensor(), #原图是28,28的纯文本图片同时像素从 `0~255` 转成浮点数 `0~1`
    download=DOWNLOAD_MNIST
)
test_data = torchvision.datasets.MNIST(
    root = './MNIST/',
    train = False
)
train_loader = Data.DataLoader(
    dataset = train_data,
    batch_size = BATCH_SIZE,
    shuffle=True
)
# 取2000条测试数据，(2000, 28, 28) -> (2000, 1, 28, 28)
test_x = torch.unsqueeze(test_data.data, dim=1).float()[:2000] / 255.
test_y = test_data.targets[:2000] #[1]的数字lable

# 2. 定义网络
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1, #输入图片（1，28，28）
                out_channels = 16,
                kernel_size = 5,
                stride=1,
                padding=2 # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) #图片尺寸下降（16,14,14）
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels = 16,
                out_channels = 32,
                kernel_size = 5,
                stride = 1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) #32,7,7
        )
        self.out = nn.Linear(32*7*7,10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1) #B，CHW
        out = self.out(x)
        return out

cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        pred_y = cnn(batch_x)
        loss = loss_func(pred_y, batch_y)
        optimizer.zero_grad() # 清空上一层梯度
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            torch.save(cnn.state_dict(), 'abs.pt')
            cnn.eval()
            with torch.no_grad():
                test_output = cnn(test_x)
                # max消除dim那个维度
                # 返回的形式为torch.return_types.max(
                #           values=tensor([0.7000, 0.9000]),
                #           indices=tensor([2, 2]))
                # 后面的[1]代表获取indices
                # pred_y = test_output.argmax(dim=1)
                pred_y = torch.max(test_output, 1)[1].numpy()
            print('epoch: ',epoch, '| train loss: %.4f'%loss.data.numpy())
            cnn.train()
            
# 打印前十个测试结果和真实结果进行对比
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].numpy()
print(pred_y, 'pred')
print(test_y[:10].numpy(), 'rel')

```
### 4.4 Transformer
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def RoPE(x):
    # 多头
    N,H,L,D = x.shape
    pos = torch.arange(L).unsqueeze(1)
    div_term = torch.exp(
        -math.log(10000) * torch.arange(0,D,2)/D
    )
    angles = pos * div_term
    pos_cos = torch.cos(angles).repeat_interleave(2,dim=-1)
    pos_sin = torch.sin(angles).repeat_interleave(2,dim=-1)
    x_rot = torch.stack(
        [
        -x[...,1::2], x[...,0::2]
        ],
        dim=-1
    ).view(N,H,L,D)
    return x*pos_cos + x_rot*pos_sin

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, model_dim, hidden_dim, num_head, drop):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_head
        self.num_head = num_head
        
        self.to_q = nn.Linear(model_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(model_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(model_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(drop)
        self.out = nn.Linear(hidden_dim, model_dim)
        
        self.attention_weights = None
    def forward(self, x, context=None, mask = None):
        N,Lq,_ = x.shape
        if context is None:
            context = x
        N,Lk,_ = context.shape
        q = self.to_q(x).view(N,Lq,self.num_head,self.head_dim).transpose(1,2)
        k = self.to_k(context).view(N,Lk,self.num_head,self.head_dim).transpose(1,2)
        v = self.to_v(context).view(N,Lk,self.num_head,self.head_dim).transpose(1,2)
        q = RoPE(q)
        k = RoPE(k)
        scores = q@k.transpose(-1,-2) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -math.inf)
        self.attention_weights = F.softmax(scores, dim=-1)
        attention = self.dropout(self.attention_weights)
        out = attention @ v
        out = out.transpose(1,2).contiguous().view(N,Lq,-1)
        return self.out(out)
         
class LayerNorm(nn.Module):
    def __init__(self, model_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(model_dim))
        self.beta = nn.Parameter(torch.zeros(model_dim))
    def forward(self,x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x-mean)/torch.sqrt(var+self.eps)
        return x*self.gamma + self.beta
        
class FeedForwardNet(nn.Module):
    def __init__(self, model_dim, hidden_dim, drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, hidden_dim)
            nn.GELU()
            nn.Dropout(drop)  
            nn.Linear(hidden_dim, model_dim)
            nn.Dropout(drop)        
        )

    def forward(self,x):
        return self.net(x)

class Transformer_Encoder(nn.Module):
    def __init__(self, model_dim, ffn_dim, num_head, atten_dim):
        super().__init__()
        self.LN1 = LayerNorm(model_dim)
        self.LN2 = LayerNorm(model_dim)
        self.attention = MultiHeadSelfAttention(model_dim,atten_dim,num_head,drop=0.1)
        self.ffn = FeedForwardNet(model_dim, ffn_dim, drop=0.1)
    def forward(self,x):
        x = self.LN1(x+self.attention(x))
        x = self.LN2(x+self.ffn(x))
        return x
    
```