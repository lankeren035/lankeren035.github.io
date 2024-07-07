---
title: python中面向对象的使用
date: 2024-5-9 10:00:00
tags: [python]
categories: [python]
comment: true
toc: true
---
#  
<!--more-->
# python中面向对象的使用

- 定义接口：
```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError("Subclass must implement abstract method")
```