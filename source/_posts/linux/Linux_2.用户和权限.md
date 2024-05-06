---
title: Linux入门-2用户和权限
date: 2021-03-01 20:00:00
toc: true
tags: [Linux]
categories: [Linux]

---
#

<!--more-->

### 1. su命令
su [-] [用户名]

### 2. sudo命令（以管理员身份运行）

sudo Linux命令
- 先进入root用户
- 再visudo，G到最后一行，o到末尾，再末尾输入：用户名 ALL=(ALL)      NOPASSWD: ALL
- 再esc，:wq退出 

### 3. 用户组
- 创建：groupadd 组名
- 删除：groupdel 组名

### 4. 用户
#### 4-1创建用户
useradd [-g -d] 用户名
（useradd 用户名 -g 组名 -d 目录）
- g 指定用户的组，不写则组与用户名同
- d指定用户home目录位置
#### 4-2 删除用户
userdel [-r] 用户名
- r 删除home目录
#### 4-3 查看用户所属组
id [用户名]
#### 4-4 修改用户所属组
usermod -aG 组名 用户名
#### 4-5 查看系统用户(组)
getent passwd或group
- 用户显示的信息：用户名:密码:用户组:描述信息:HOME目录:执行终端（默认bash)
- 组显示信息：组名:组认证:组ID

### 5 权限

![](./lmg/linux/user/8.png)

![](D:\blog\themes\yilia\source\img\linux\user\8.png)