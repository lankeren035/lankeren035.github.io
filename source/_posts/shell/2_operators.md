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

## 2.1 运算符

- `$((运算式))` ：里面可以用数学符号，如>=
- `$[运算式]`：不能用数学符号，只能用-ge

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

  

