---
title: 1- shell 概述
date: 2024-8-30 12:00:00

tags: [shell]

categories: [shell]

comment: true

toc: true



---

#

<!--more-->

# 1- shell 概述

- shell翻译为：贝壳，它是Linux核心与外层应用之间的一个接口，将外部应用的命令解释成Linux内核可以理解的指令。

```shell
cat /etc/shells #查看当前支持的shell版本
```

```shell
/bin/sh
/bin/bash #bash是从sh中发展过来的
/usr/bin/bash
/bin/rbash
/usr/bin/rbash
/usr/bin/sh
/bin/dash
/usr/bin/dash #ubuntu使用dash
/usr/bin/tmux
/usr/bin/screen
```



- 查看系统使用的是哪个shell：

  ```shell
  ls -l /bin/ | grep bash
  # 或者使用
  # echo $SHELL
  ```

  - 看到有箭头：`rbash -> bash`说明系统将rbash自动转换成bash执行。

    

## 1.1 shell脚本入门

### 1）脚本格式

- 以`#!/bin/bash`开头，指定解析器。

### 2）第一个shell脚本

- 编写：

    ```shell
    echo -e '#!/bin/bash\necho "hello world!"' > ./hello.sh
    ```

- 运行：

  ```shell
  bash hello.sh # 创建一个bash进程来执行
  sh hello.sh
  hello.sh
  # 直接在当前命令行执行，没有子进程
source hello.sh
  . hello.sh
  ```
  
  - 直接使用文件名发现无法运行，这是因为我们没有对文件的`执行`权限。
  
    ```shell
    chmod +x hello.sh
    # 因为'./’不在环境变量中，所以无法直接使用文件名执行
    # 直接使用文件名执行不太安全。
    ./hello.sh #使用路径的方式执行
    ```
  
  - 由于前三种执行方法都会创建一个子进程，而子进程与父进程环境可能不一样，在子环境里操作影响不到父环境（子进程里面修改了父进程的全局变量，到父进程里面就没改了）。这也就是为什么每次创建环境变量后要source一下配置文件。
  
    ```shell
    ps -f #可以看到有个-bash进程
    bash
    ps -f #可以看到多了个bash进程，其父进程是-bash
    exit
    ps -f #推出了子进程
    ```
  
    

## 1.2 变量

### 1.2.1 系统变量

#### 1）常用的系统变量


>`$HOME `  `$PWD`  `$SHELL`  `$USER`

#### 2）查看系统变量

```shell
echo $HOME
env | less
printenv | less
printenv HOME
set | less #查看所有变量
```



### 1.2.2 用户变量

#### 1）基本语法

- 1. 定义变量（`=`后面不能有空格）：

  - 局部变量：

    ```shell
    var='hello Linux!'
    ```

    ```shell
    echo $var
    
    # 证明他是局部变量
    bash
    
    echo $var
    
    # 退出
    exit
    ```

  - 全局变量：

    ```
    globalvar='hello world' #先定义局部变量
    export globalvar
    ```

    ```shell
    bash
    
    echo $globalvar
    
    # 退出
    exit
    ```

    - 在子shell中修改全局变量不会影像父shell：

      ```shell
      bash
      
      globalvar='hello'
      echo $globalvar
      
      exit
      
      echo $globalvar
      ```

    - 只读变量

      ```shell
      readonly constvar=7
      ```

      

- 2. 删除变量（只读变量不可unset）

     ```shell
     unset 变量名
     ```

     ```shell
     set | grep globalvar
     unset globalvar
     set | grep globalvar #输出一个_=gloablvar是因为'_'通常用来保存最后一个命令的结果
     ```
     
     

#### 2）变量定义规则

- 变量名可包含（不能以数字开头）：
  - 字母
  - 数字
  - 下划线
- 环境变量建议大写
- 等号两侧不能有空格
- bash中默认变量都是string类型，无法直接进行数值计算
- 变量的值如果包含空格，需要用引号。

### 1.2.3 特殊变量

- 运行脚本时输入的参数。

#### 1）`$n`

- `$0`：脚本名称

- `$1`：脚本输入的第一个参数

- `${10}`：脚本输入的第十个参数

  ```shell
  echo -e '#!/bin/bash\necho "脚本名称：$0"\necho "变量1：$1"\necho "变量2：$2"\n' > specialvar.sh
  chmod +x specialvar.sh
  ./specialvar.sh 111 222
  ```

#### 2）`$#`

- 输入参数的个数

#### 3）`$*`   、   `$@`

- 不带双引号时：等价
- 带双引号时：
  - `"$*"`：整体返回所有参数
  - `"$#"`：数组返回所有参数

#### 4）`$?`

- 最后一次执行的命令的返回状态。如果值为0，说明上一个命令正确执行

  ```shell
  ./specialvar.sh
  echo $?
  
  ./ssssss.sh
  echo $?
  ```

  

## 1.3 输入输出

- 输出：

  ```shell
  echo hello
  ```

- 输入

  ```
  read -t 7 -p "10s内输入你的名字：" name
  ```

  