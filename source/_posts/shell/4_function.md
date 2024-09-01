---
title: 4- shell 函数
date: 2024-8-31 13:00:00

tags: [shell]

categories: [shell]

comment: true

toc: true




---

#

<!--more-->

# 4- shell 函数

## 4.1 系统函数（系统命令）

- basename：获取文件名

  ```shell
  echo script name: $(basename /hom/test/hello.sh .sh)
  ```

- dirname：获取路径名

  ```shell
  #!/bin/bash
  
  echo script path: $(cd $(dirname $0); pwd)
  ```

  

## 4.2 自定义函数

- 函数返回值只能通过`$?`系统变量获得；可以在函数里面写return，return只能跟数值（0-255，因为要赋给$?）。如果不加return，则以最后一条命令运行结果作为返回值。

  ```shell
  #!/bin/bash
  
  # 使用return要限制返回值
  function add1(){
  	s=$[ $1 + $2 ] #也可使用位置参数
  	return $s
  }
  
  # 使用return返回字符串
  function add2(){
  	s=$[ $1 + $2 ] #也可使用位置参数
  	return "和为： "$s
  }
  
  # 直接输出
  function add3(){
  	s=$[ $1 + $2 ] #也可使用位置参数
  	echo $s
  }
  
  
  read -p "输入参数1：" a
  read -p "输入参数2：" b
  
  add1 $a $b
  echo 使用return返回数值：$?
  add2 $a $b
  echo 使用return返回字符串
  echo 使用命令替换符输出$(add3 $a $b)
  ```
  
