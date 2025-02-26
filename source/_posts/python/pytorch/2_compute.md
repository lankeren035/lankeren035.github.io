---
title: 2- torch计算操作
date: 2025-2-25 10:00:00
tags: [pytorch,python]
categories: [pytorch]
comment: true
toc: true


---

#

<!--more-->



- torch.bmm : 对应batch相乘

```python
import torch
a = torch.tensor(
    [
        [
            [1,2],
            [1,2]
        ],
        [
            [2,3],
            [4,5]
        ]
    ]
)

b = torch.tensor(
    [
        [
            [1,4],
            [2,2]
        ],
        [
            [3,1],
            [1,2]
        ]
    ]
)
print(torch.bmm(a,b))
```

```
tensor([[[ 5,  8],
         [ 5,  8]],

        [[ 9,  8],
         [17, 14]]])
```



