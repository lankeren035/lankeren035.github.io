---
title: 6- shell 文本处理工具
date: 2024-8-31 15:00:00

tags: [shell]

categories: [shell]

comment: true

toc: true

---

#

<!--more-->

# 6- shell 文本处理工具

## 6.1 cut

- 将文件拆分成一个表格，裁剪出指定的区域

```shell
cut [-f -d -c] 文件名
```

| 参数 | 解释                                         |
| ---- | -------------------------------------------- |
| -f   | 截取第几列                                   |
| -d   | 分隔符，默认\t（按照分隔符将文件分成很多列） |
| -c   | 按字符提取                                   |

```shell
echo "hello" | cut -c 1-3 #提取第1到第3个字符

echo -e '你好 hello\n北 上\n京 海' > cut.txt
cut -d " " -f 1 cut.txt
cut -d " " -f 2,3 cut.txt
cut -d " " -f 2- cut.txt

cat /etc/passwd | grep bash
cat /etc/passwd | grep bash$ | cut -d ":" -f 1,6,7
ifconfig | grep 'inet ' | cut -d ' ' -f 10
```



## 6.2 awk

- 将文件逐行读入，默认以空格分隔，切开的部分再进行分析处理。

```shell
awk [-F -v] '/pattern1/{action1} /pattern2/{action2} ...' 文件名
```

- pattern：正则匹配

- action：对匹配内容执行操作

- 关键字

  | 关键字 | 解释             |
  | ------ | ---------------- |
  | BEGIN  | 处理前做什么操作 |
  | END    | 处理后做什么操作 |

- 内置变量

  | 变量     | 说明         |
  | -------- | ------------ |
  | FILENAME | 文件名       |
  | NR       | 当前行号     |
  | NF       | 当前行有几列 |

  

  ```shell
  # 搜索以root开头的所有行，并输出第7列
  cat /etc/passwd | grep ^root | cut -d ':' -f 7
  cat /etc/passwd | awk -F : '/^root/{print $7}' #分割后每一列就是一个位置参数
  
  # 搜索以root开头的所有行，并输出第1,7列用逗号分割
  cat /etc/passwd | awk -F : '/^root/{print $1","$7}'
  
  # 搜索第1,7列用逗号分割，且在所有行前面添加：'user, shell'，在后面添加’end'
  cat /etc/passwd | awk -F : 'BEGIN{print "user, shell"} {print $1","$7} END{print "end"}'
  
  # 将第三行数值都+i
  cat /etc/passwd | awk -F : -v i=1 '{print $3+i}'
  
  # 输出文件名，每行的行号，每行有几列
  cat /etc/passwd | awk -F : '{print FILENAME","NR","NF}' 
  
  # 输出空行所在的行号
  ifconfig | grep -n ^$
  ifconfig | awk '/^$/ {print NR}'
  
  # 找到ip
  ifconfig | grep 'inet ' | cut -d ' ' -f 10
  ifconfig | awk '/inet /{print $2}'
  ```

  

