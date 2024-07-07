---
title: python处理文件
date: 2024-06-29 20:00:00
toc: true
tags: [python]
categories: [python]
---

#  

<!-- more -->

# python处理文件



## 1. 批量重命名文件夹

```python
import os
path = 'C:\\Users\\18048\\Desktop\\人-测试\\14th\\单词\\人物\\新建文件夹'
txt = 'test.txt'

dirs = os.listdir(path)
print(dirs)
with open(path+txt,'r',encoding='utf-8') as file:
    for (line,dir) in zip(file,dirs):
        name = line.strip()
        print(name,dir)
        os.rename(os.path.join(path,dir),os.path.join(path,name))
```

