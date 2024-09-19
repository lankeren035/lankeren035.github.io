---
title: batch normalization与layer normalization
date: 2024-9-16 15:00:00
tags: [机器学习]
categories: [机器学习]
comment: true
toc: true

---

#### 

<!--more-->

# batch normalization与layer normalization

## 1. cv中的归一化

### 1.1 二维

```
[
	[4, 3, 2],
	[3, 3, 2],
	[2, 2, 2]
]
```

- 维度: (batch, feature)

- #### batch normalization:

  - 跨batch处理

  ```
  # 处理feature 0
  从batch 0中取出feature 0 :4
  从batch 1中取出feature 0 :3
  从batch 2中取出feature 0 :2
  对[
  	4,
  	3,
  	2,
  ]进行归一化, 中间一个肯定是0:
  [
  	x1,
  	0,
  	-x1,
  ]
  
  # 处理feature 1
  从batch 0中取出feature 1 :3
  从batch 1中取出feature 1 :3
  从batch 2中取出feature 1 :2
  处理后前两个肯定一样:
  [
  	a, 
  	a, 
  	b,
  ]
  
  # 处理feature 2
  从batch 0中取出feature 2 :2
  从batch 1中取出feature 2 :2
  从batch 2中取出feature 2 :2
  这三个数相等, 归一化后都是0:
  [
  	0,
  	0,
  	0,
  ]
  
  #最后结果:
  [
  	[x1,  a, 0]
  	[0,   a, 0]
  	[-x1, b, 0]
  ]
  ```

- #### layer normalization

  - 按batch处理

  ```
  [4, 3, 2] 归一化
  [3, 3, 2] 归一化
  [2, 2, 2] 归一化
  ```




### 1.2 三维

```
[
	[
		[1.0, 4.0, 7.0],
        [0.0, 2.0, 4.0]
    ],
    [
    	[1.0, 3.0, 6.0],
        [2.0, 3.0, 1.0]
    ]
]

```

- 维度: (batch, feature, h*w)

- #### batch normalization

  - 跨batch处理

  ```
  # 处理feature 0
  从batch 0中取出feature 0 :[1.0, 4.0, 7.0]
  从batch 1中取出feature 0 :[1.0, 3.0, 6.0]
  进行归一化, 两个1原来的位置归一化后的结果肯定一样:
  [
  	[
  		[a1, x2, x3],
          [          ]
      ],
      [
      	[a1, x4, x5],
          [          ]
      ]
  ]
  
  # 处理feature 1
  从batch 0中取出feature 1 : [0.0, 2.0, 4.0]
  从batch 1中取出feature 1 : [2.0, 3.0, 1.0]
  归一化后两个2的位置肯定一样:
  [
  	[
  		[a1, x2, x3],
          [x6, b1, x7]
      ],
      [
      	[a1, x4, x5],
          [b1, x8, x9]
      ]
  ]
  ```

- #### layer normalization
  - 按batch处理

    ```
    归一化处理batch0
    	[
    		[1.0, 4.0, 7.0],
            [0.0, 2.0, 4.0]
    	]
    
    	
    	
    归一化处理batch1
        [
        	[1.0, 3.0, 6.0],
            [2.0, 3.0, 1.0]
        ]
    
    
    ```

  - 在nlp中是对batch 0中的feature 0归一化, 然后对batch 0的feature 1归一化, 然后对batch 1的feature 0归一化...