---
title: 0_python基础
date: 2026-10-31
tags:
comment: true
toc: true
published: true
permalink:
hexo-path:
---
#
<!--more-->

# 1. 数据存储与运算

## 1.1 字面量与变量

| 名称  | 解释           | 举例                   |
| --- | ------------ | -------------------- |
| 字面量 | 直接书写的固定值（数据） | 数字<br>字符串<br>布尔<br>空 |
| 变量  | 存储单个数据的容器    | a=xxx                |

### 1）变量命名规则
1. 只能包含字母、数字、下划线
2. 不能数字开头
3. 区分大小写
4. 不能占用关键字

## 1.2 数据类型
| 名称    | 解释  | 举例  |
| ----- | --- | --- |
| int   |     |     |
| float |     |     |
| str   |     |     |
| bool  |     |     |
| None  |     |     |
- 获取数据类型：type(x)
- 是否为指定类型：isinstance(x, int)
### 1）字符串
```python
# 1. 定义
'a',   "a",   """a"""

# 2. 格式化
#1
print("aaaa%s"%'abc')
print("aaaa%s%s"%('abc','abc'))
#2
name = '小明'
print(f'你好我是{name}')

```
## 1.3 输入输出

## 1.4 运算符

### 1. 算数运算
| 名称    | 解释  | 举例  |
| ----- | --- | --- |
| +-*/% |     |     |
| //    |     |     |
| **    |     |     |

### 2. 赋值运算
### 3. 比较运算

### 4. 逻辑运算
| 名称  | 解释  | 举例  |
| --- | --- | --- |
| and |     |     |
| or  |     |     |
| not |     |     |
# 2. 控制流
## 2.1 判断
```python
if xxx:
	pass
elif xxx:
	pass
elif xxx:
	pass
else:
	pass
```
## 2.2 模式匹配
```python
day = input("请输入星期几(1-7)：")
match day:
	case '1':
		pass
	case '2' | '3':
		pass
	case '4' if xxx:
		pass
	case _:
		pass
```
## 2.3 循环
```python
while xxx:
	pass
else:
	pass
	
for x in range(start, end, step):
	pass
else:
	pass
```

# 3. 数据存储容器
| 名称    | 有序性 | 重复元素   | 可变性 | 索引访问 | 切片  | 使用场景      |
| ----- | --- | ------ | --- | ---- | --- | --------- |
| str   |     |        | 不可变 |      |     | 文本处理      |
| list  |     |        |     |      |     | 有序可重复数据集合 |
| tuple |     |        | 不可变 |      |     | 固定数据记录    |
| set   | 无序  | 不允许    |     | 不支持  | 不支持 | 去重数据集合    |
| dict  |     | key不允许 |     | 不支持  | 不支持 | 键值对       |


## 3.1 列表（list）
| 名称    | 举例                    |
| ----- | --------------------- |
| 索引    | a[0]                  |
| 删除    | del                   |
| 切片    | a[start:end:step]     |
| 解包    | *a                    |
| 列表生成式 | [i for i in a if xxx] |
常用方法：

| 名称          | 解释            |
| ----------- | ------------- |
| append(x)   | 末尾追加          |
| insert(i,x) | 指定位置插入        |
| remove(x)   | 删除第一个匹配项      |
| pop(x)      | 按索引删除（默认最后一个） |
| sort()      | 排序            |
| reverse()   | 翻转            |

## 3.2 字符串（str）
- 不可变性
常用方法：

| 名称            | 解释            |
| ------------- | ------------- |
| find(a)       | 第一个匹配项，无则-1   |
| count(a)      | 统计次数          |
| upper()       | 转大写           |
| lower()       | 转小写           |
| split(‘ ’)    | 分割            |
| strip(a)      | 取出两端的字符，默认空白符 |
| replace(a,b)  | 替换            |
| startswith(a) | 是否以此开头        |

## 3.3 元组（tuple）
- 不可变性，有序，相对于list更适合只能查询的数据
常用方法：
- 定义单个元素元组时，最后面要加一个逗号

| 名称       | 解释      |
| -------- | ------- |
| count(a) | 统计      |
| index(a) | 查询，无则报错 |

## 3.4 集合（set）
- 去重
常用方法：

| 名称                   | 解释                       |
| -------------------- | ------------------------ |
| add(a)               | 添加                       |
| remove(a)            | 删除                       |
| pop()                | 随机删除一个                   |
| clear()              | 清空                       |
| difference(a)<br>-   | 差集，包含在第一个集合但不包含在第二个集合的元素 |
| union(a)<br>\|       | 并集                       |
| intersection(a)<br>& | 交集                       |

## 3.5 字典（dict）
- key不能是可变类型，key不能重复
常用方法：

| 名称       | 解释        |
| -------- | --------- |
| pop(a)   | 删除指定的key  |
| get(a)   | 查询        |
| keys()   | 返回所有key   |
| values() | 返回所有value |
| items()  | 返回所有键值对   |
# 4. 函数
## 4.1 基础
```python
def circle_area_len(r): #形参
	"""
	该函数用于根据圆的半径, 计算圆的面积和圆的周长
    :param r: 圆的半径    
    :return: 圆的面积 , 圆的周长    
    """    
    return 3.14 * r * r, 2 * 3.14 * r
al = circle_area_len(10) # 实参
```
## 4.2 进阶
### 4.2.1作用域
global：声明全局变量
### 4.2.2 传参
#### 1）位置参数
- 调用与定义顺序一致
```python
f(a,b,c)
```
#### 2）关键字参数
- 必须在位置参数之后
```python
f(a,c=1,b=2)
```
#### 3）默认参数
- 必须放在没有默认值的参数列表的后面
```python
def f(a, b=2)
```

#### 4）不定长参数
- 位置传递：args保存成元组
- 关键字传递：kwargs保存成字典
```python
def f(*args, **kwargs):
	pass
f(1,2,3,a=1,b=2)
```
#### 5）参数类型
- 普通类型
- 函数
### 4.2.3 匿名函数
```python
# lambda 参数列表: 函数体
lambda a,y: x+y
```
### 4.2.4 类型注解
```python
def f(a: list[int]) -> tuple[int, int, float]:
	pass
```
# 5. 模块
- 一个py文件就是一个模块
## 5.1 导入模块
```python
import xxx
import xxx as 别名
from xxx import yyy
from xxx import yyy as 别名
from xxx import *
```

## 5.2 自定义模块
- `__all__` 是一个模块级别的特殊变量，用于指定`import *`时 导入哪些模块
- `__name__`是一个内置变量，表示当前模块的名字
	- 当模块自己直接运行时，`__name__`的值为`"__main__"` 
	- 当被导入时`__name__`的值为文件名 
```python
__all__ = ['f1', 'f2'] # f3不会被别的文件导入
def f1():
	pass
def f2():
	pass
def f3():
	pass
```
## 5.3 软件包

- 一个文件夹，下面有一个`__init__.py` 
- 在通过 `from 包名 import *` 导入全部模块的时候，需要在 `__init__.py` 文件中添加 `__all__=[]`，控制允许导入的模块列表 
```python
import xx.yy
from xx.yy import zz
from xx.yy import **
```
# 6. 面向对象基础
## 6.1 类与对象
- `__init__`时初始化方法，创建对象自动调用
- self是类中定义的方法的第一个参数，表示当前创建的实例对象
## 6.2 实例方法
- 普通的方法
```python
class a:
	def f(self):
		pass
```
## 6.3 魔法方法
- 以双下划线开头和结尾的特殊方法，用于定义类的特殊行为，不需要我们手动调用，Python会在合适的时机自动调用

| 名字                                               | 解释                                                                                                      |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `__init__`                                       | 初始化方法                                                                                                   |
| `__str__`                                        | 字符串表示的方法                                                                                                |
| `__eq__`                                         | 比较两个对象是否相等                                                                                              |
| `__lt__`<br>`__le__`<br>`__gt__`<br>`__ge__`<br> | 支持比较两个对象的大小(小于(less than), 小于等于(less than or equal), 大于(greater than), 大于等于(greater than or equal))<br> |
```python
class Car:
	pass
c1 = Car()
print(c1) # 输出对象的内存地址
c2 = Car()
print(c1 == c2) # 默认基于对象内存地址比较 False
print(c1 < c2) # 默认自定义的对象之间无法比较大小
```
## 6.4 属性
- 实例属性：每个对象独立拥有：定义时：`self.xxx`，操作时：`实例对象.属性`
- 类属性：所有实例共享，通过`类名.属性名`的方式操作

# 7. 异常
- 程序运行过程中出现的错误，它会中断程序的正常执行流程。
```python
try:
	pass
except NameError as e:
	pass
finally:
	pass
```

# 8. 面向对象高级
## 8.1 封装
- 把数据(属性)和操作数据的函数(方法)捆绑在一起，形成一个独立的单元（类），并隐藏内部的实现细节（私有属性，私有方法），只对外暴露必要的功能（方法）。
- 私有属性，私有方法：以 __ 作为开头。只能在类的内部访问，在外部是无法访问的；
## 8.2 继承
- 子类继承父类，会将父类公共的属性和方法继承下来（不含私有）
- 重写是指子类继承父类后，如果父类中的方法不满足需求，可以在子类中重新定义父类中已有的方法（方法名相同），从而用子类的实现替换父类的实现。
- 子类中如何调用父类方法：`super().f()` 或者`父类名.f(self)`
- 多继承：同时继承了多个父类的情况（会将多个父类中的非私有的属性和方法都继承下来）
## 8.3 多态
- 同一个方法，具有不同的形态、行为、表现。如：定义函数时，参数类型指定为父类类型，在执行的时候传入不同的子类对象，就具有不同的形态
- 鸭子类型也是多态的一种体现，鸭子类型关注的是对象的行为（有什么方法），而不是对象的类型（无需继承关系）。如果它走路像鸭子，叫起来也像鸭子，那么他就是鸭子。
- java, c++中的多态通常时第一种，基于继承关系的。python中的多态通常是第二种。
```python
class Car:
    def charge(self):
    print('正在补充燃料...')
    
class FuelCar(Car):
    def charge(self):
        print('燃油汽车正在加油 ...')
class ElectricCar(Car):
    def charge(self):
        print('电动汽车正在充电...')

def handle_charge(car: Car):
    car.charge()
# 同一调用语句,传入对象不同,触发不同形为
handle_charge(FuelCar("BMW", "X5", "白色", 500, "张三"))  
handle_charge(ElectricCar("BYD", "汉", "白色", 500, "张三"))



class Dog:
	def swimming(self): 
		print(f'{self.age} 岁的 {self.name} 正在游泳...')
class Duck:
	def swimming(self):  
		print(f'{self.age} 岁的 {self.name} 正在游泳...')
class Pig:
	def swimming(self):  
		print(f'{self.age} 岁的 {self.name} 正在游泳...')
def go_swimming(duck: Duck): # 这里的Duck注解，只是一个君子协定，你可以实际传入别的
	duck.swimming()

# 鸭子类型的优势是不需要存在继承关系，只要对象有相应的方法就能使用。
go_swimming(Dog("旺财", 4))
go_swimming(Duck("唐老鸭", 2))
go_swimming(Pig("佩奇", 1))
```