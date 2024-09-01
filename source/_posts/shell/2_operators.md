---
title: 2- shell 操作符
date: 2024-8-30 13:00:00

tags: [shell]

categories: [shell]

comment: true

toc: true


---

#

<!--more-->

# 2- shell 操作符

## 2.0 命令替换符

- `$()`：将里面的命令执行输出作为参数

  ```shell
  echo $(ls)
  echo `ls`
  ```

  

## 2.1 运算符

- `$((运算式))` ：里面可以用数学符号，如>=

- `$[运算式]`：不能用数学符号，只能用-ge

  - | 特性           | `[ ]`                              | `(( ))`                                     |
    | -------------- | ---------------------------------- | ------------------------------------------- |
    | **主要用途**   | 字符串比较、文件测试、简单整数比较 | 算术运算和整数比较                          |
    | **支持操作符** | -eq`, `-ne`, `-gt`, `-lt           | `==`, `!=`, `>`, `<`, `+`, `-`, `*`, `/` 等 |
    | **变量处理**   | 需要使用 `$` 进行变量引用          | 变量不需要 `$`                              |
    | **返回值**     | 逻辑真假 (`true` or `false`)       | 数值结果（非零为真，零为假）                |


- expr 运算式（运算式要空格）

  ```shell
  # 计算
  expr \( 7 - 2 \) \* 2 #注意空格，转义
  
  # $()返回命令的结果
  tempvar=$(expr \( 7 - 2 \) \* 2)
  tempvar=`expr \( 7 - 2 \) \* 3`  #反引号
  ```

  - 编写一个add命令计算两个数相加：

    ```shell
    echo -e '#!/bin/bash\nsum=$[$1+$2]\necho "sum=$sum"' > add
    . add 2 5
    ```

    

## 2.2 条件判断

- shell中的true为0，因为命令成功执行返回的是0。

### 1）基本语法

- `test 表达式`
- `[ 表达式 ]`
  ```shell
  tempvar=hello
  test $tempvar = hello
  echo $?
  
  test $tempvar = hello0
  echo $?
  
  [ $tempvar = hello0 ]
  echo $?
  ```

### 2）常用判断条件

- linux中可以用`let 语句`的方式实现类似其他语言的功能，如

  ```shell
  let i++
  let i+=1
  ```

  

- 大小判断

  | 符号    | 解释                   |
  | ------- | ---------------------- |
  | -eq     | =（equal）             |
  | -ne     | $\neq$（not equal）    |
  | -lt     | <（less than）         |
  | -le     | $\leq$（less equal）   |
  | -gt     | >（grater than）       |
  | -ge     | $\geq$（grater equal） |
  | =<br>!= | 字符串之间比较         |



- 权限判断

  | 符号 | 解释     |
  | ---- | -------- |
  | -r   | 读权限   |
  | -w   | 写权限   |
  | -x   | 执行权限 |

  ```
  [ -r hello.sh ]
  echo $?
  ```

  

- 文件类型判断

  | 符号 | 解释               |
  | ---- | ------------------ |
  | -e   | 文件存在           |
  | -f   | 文件存在，且是文件 |
  | -d   | 文件存在，且是目录 |

- 三元表达式：

  ```shell
  [ 1 -le 3 ] && echo yes || echo no
  [ 1 -ge 3 ] && echo yes || echo no
  ```

  

