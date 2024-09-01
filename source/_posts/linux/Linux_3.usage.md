---
title: Linux入门-3实用操作1
date: 2024-03-03 13:00:00
toc: true
tags: [Linux]
categories: [Linux]
---

#

<!--more-->

# 3 Linux实用操作

## 3.1 快捷键

| 快捷键    | 说明                                   |
| --------- | -------------------------------------- |
| ctrl + c  | 停止程序<br>重新输入命令               |
| ctrl + d  | 退出账户<br>退出程序界面               |
| history   | 历史命令                               |
| !命令前缀 | 执行上一次匹配前缀的命令               |
| ctrl + r  | 搜索历史命令（回车执行，左右键不执行） |
| ctrl + l  | 清空终端（clear）                      |

| 移动光标  |              |
| --------- | ------------ |
| ctrl + a  | 跳到命令开头 |
| ctrl + e  | 跳到命令结尾 |
| ctrl + 左 | 左跳一个单词 |

## 3.2 软件安装

### 3.2.1 yum（centos）

- RPM包软件管理器，用于自动化安装配置Linux软件，并可以自动解决依赖问题。相当于Linux应用商店

```bash
yum [-y] [install | remove |search] 软件名
```

| 选项    | 解释     |
| ------- | -------- |
| -y      | 自动确认 |
| install | 安装     |
| remove  | 卸载     |
| search  | 搜索     |

- ubuntu中用apt



## 3.3 服务管理

- systemctl命令

```bash
systemctl start | stop | status | enable |disable 服务名
```

|                | 说明       |
| -------------- | ---------- |
| status         | 查看状态   |
| enable         | 开机自启动 |
| NetworkManager | 主网络服务 |
| network        | 副网络服务 |
| firewalld      | 防火墙服务 |
| sshd，ssh      |            |

- 部分软件安装后没有自动集成到systemctl中，我们可以手动添加  

## 3.4 软链接

- 类似快捷方式

```bash
ln -s 起点 终点 
```

## 3.5 时间

- date命令

```bash
date [-d] [+格式化字符串]
```

| 格式 | 说明                               |
| ---- | ---------------------------------- |
| %Y   | 年                                 |
| %y   | 两位数字的年                       |
| %m   | 月                                 |
| %d   | 日                                 |
| %H   | 小时                               |
| %M   | 分钟                               |
| %S   | 秒                                 |
| %s   | 从1970-01-01 0:0:0到现在过了多少秒 |

​	例子

```bash
date "+%Y-%m-%d %H:%M:%S"
date -d "+1day" +%Y%m%d #后一天
date +%s #时间戳
```

- 修改时区

```bash
rm -f /etc/localtime #删除文件
sudo ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime #将该文件链接为localtime
```

- ntp程序

  通过ntp程序自动校准时间

```bash
yum -y install ntp #安装
systemctl start ntpd #启动服务
systemctl enable ntpd #开机自启动
```

​		手动校准

```bash
ntpdate -u ntp.aliyun.com
```

