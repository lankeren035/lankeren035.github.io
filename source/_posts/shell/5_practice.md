---
title: 5- shell 综合应用
date: 2024-8-31 14:00:00

tags: [shell]

categories: [shell]

comment: true

toc: true
---

#

<!--more-->

# 5- shell 综合应用

## 5.1 每天将指定目录备份到指定路径。

```shell
#!/bin/bash

dest=./backup

# 合法性判断
if (($# != 1))# 判断参数格式
then
	echo 参数个数不对
	exit
elif [ ! -d $1 ] #判断输入路径是否存在
then
	echo 输入的路径不对
	exit
else
	mkdir -p $dest
	date=$(date +%y%m%d)
	# 获取绝对路径
	input_dir_name=$(basename $1)
	output_dir_name=$(basename $dest)
	input_dir_path=$(cd $(dirname $1); pwd)
	output_dir_path=$(cd $(dirname $dest); pwd)
	output_name=archive_${input_dir_name}_$date.tar.gz
	output_path=$output_dir_path/$output_dir_name/$output_name

	
	echo 开始归档
	tar -czf $output_path $input_dir_path/$input_dir_name
fi

if (($?==0))
then
	echo 归档：$output_path
else
	echo 归档错误
fi
```

- 写入定时任务：

  ```shell
  crontab -l
  crontab -e
  ```

  - 写入：（每天凌晨2点备份）

    ```shell
    0 2 * * * bash文件路径 备份路径
    ```

    

## 5.2 发送消息

- 使用linux自带的mesg和write工具像其他已登录的用户发送消息。

  ```shell
  who -T #查看当前有几个用户登录，以及是否打开消息功能（+代表打开）
  mesg y #打开消息功能
  write 用户名 控制台名 #who命令获得控制台名
  ```

  ```shell
  #!/bin/bash
  
  read user msg_enable terminal <<< $(who -T | grep -i -m 1 "$1" | awk '{print $1,$2,$3}') #-i忽略大小写，-m取一条
  
  if [ -z "$user" ] #-z判断是否为空，用户是否登录
  then
  	echo $1不在线
  	exit
  elif [ "$msg_enable" != '+' ]
  then
  	echo $1无法接收消息
  	exit
  elif [ -z $2 ] #是否有消息
  then
  	echo 没有消息发送
  	exit
  else
  	message=$(echo $* | cut -d " " -f 2-)
  	echo $message | write $user $terminal
  fi
  
  if [ $? != 0 ]
  then
  	echo 发送失败
  else
  	echo 发送成功
  fi
  exit
  ```

  