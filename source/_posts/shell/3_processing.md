---
title: 3- shell 流程控制
date: 2024-8-31 12:00:00

tags: [shell]

categories: [shell]

comment: true

toc: true



---

#

<!--more-->

# 3- shell 流程控制

## 3.1 分支语句

### 3.1.1 if语句

```shell
#!/bin/bash

if [ "$1"x = "hello"x ] #为了防止传入参数为空导致出错
then
    echo hello
elif [ "$1"x = "world"x ]
then
	echo world
else
	echo "$1"?
fi
```

```shell
# 用';'可以将命令写在一行
tempvar=20
# 使用-a表示and
if [ $tempvar -gt 18 -a $tempvar -lt 35 ]; then echo 青年; else echo 不是青年; fi  #如果if语句的]没带空格会导致if语法错误，判断出错，走向else
# 使用&&
if [ $tempvar -gt 18 ] && [ $tempvar -lt 35 ]; then echo 青年; else echo 不是青年; fi
```



### 3.1.2 case语句

```shell
#!/bin/bash

case $1 in
1)
	echo one
;; #相当于break
2)
	echo two
;;
3)
	echo three
;;
*) #default
	echo other
;;
esac
```



## 3.2 循环语句

### 3.2.1 for语句

```shell
#!/bin/bash

for (( i=0; i<100; i++))
do
	sum=$[ $sum + $i ]
	sum1=$(($sum1+$i))
done
echo $sum
echo $sum1
```

```shell
for i in {1..100}; do sum=$[$sum+$i]; done; echo $sum
```



### 3.2.2 while语句

```shell
#!/bin/bash
i=1
while (($i<=100))
do
	let sum+=i
	let i++
done
echo $sum
```

