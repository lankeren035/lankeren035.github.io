---
title: Linux入门-2用户和权限
date: 2021-03-03 12:00:00
toc: true
tags: [Linux]
categories: [Linux]

---
#

<!--more-->

# 2 用户和权限

| <a href='#su'>切换用户</a>             | su 用户名                                 |
| -------------------------------------- | ----------------------------------------- |
| <a href='#sudo'>管理员运行</a>         | sudo 命令                                 |
| <a href='#groupadd'>创建用户组</a>     | groupadd 组名                             |
| <a href='#groupdel'>删除用户组</a>     | groupdel 组名                             |
| <a href='#useradd'>创建用户</a>        | useradd 用户名 -g 组名 -d 用户家目录      |
| <a href='#userdel'>删除用户</a>        | userdel -r 用户名                         |
| <a href='#id'>查看用户所属组</a>       | id 用户名                                 |
| <a href='#usermod'>修改用户所属组</a>  | usermod -aG 组名 用户名                   |
| <a href='#getent'>查看系统用户(组)</a> | getent passwd或group                      |
| <a href='#chmod'>修改权限</a>          | chmod [-R] 权限 文件或文件夹              |
| <a href='#chown'>修改所属用户</a>      | `chown [-R] [用户][:用户组] 文件或文件夹` |



### 1. su（switch user）<span id='su'>命令</span>

`su [-] [用户名]`

- 退出用户：exit或ctrl+d

### 2. sudo<span id='sudo'>命令</span>（以管理员身份运行）

`sudo Linux命令`

- 需要先为用户配置sudo认证  
  - 先进入root用户
  - 再visudo，G到最后一行，o到末尾，再末尾输入：`用户名 ALL=(ALL)      NOPASSWD: ALL`
  - 再esc，:wq退出 

### 3. 用户组

- 创建：<span id='groupadd'>groupadd</span> 组名
- 删除：<span id='groupdel'>groupdel</span> 组名

### 4. 用户

#### 4-1创建用户
<span id='useradd'>useradd [-g -d] 用户名</span>
（useradd 用户名 -g 组名 -d 目录）

- g 指定用户的组，不写则组与用户名同
- d指定用户home目录位置
#### 4-2 删除用户
<span id='userdel'>userdel [-r] 用户名</span>

- r 删除home目录
#### 4-3 查看用户所属组<span id='id'> </span>

`id [用户名]`

#### 4-4 修改用户所属组<span id='usermod'> </span>

`usermod -aG 组名 用户名`

#### 4-5 查看系统用户(组)<span id='getent'> </span>

getent passwd或group
- 用户显示的信息：用户名:密码:用户组:描述信息:HOME目录:执行终端（默认bash)
- 组显示信息：组名:组认证:组ID

### 5 权限

#### 5.1 权限信息

![](./img/linux/user/8.png)

![](D:\blog\themes\yilia\source\img\linux\user\8.png)

- 权限信息（序号1）

  总共10位：

  | 类型                              | 所属用户权限 |      |      | 所属用户组权限 |      |      | 其他用户权限 |      |      |
  | --------------------------------- | ------------ | ---- | ---- | -------------- | ---- | ---- | ------------ | ---- | ---- |
  | -/d/l                             | r/-          | w/-  | x/-  | r/-            | w/-  | x/-  | r/-          | w/-  | x/-  |
  | -：文件<br>d：文件夹<br>l：软连接 | 读           | 写   | 执行 |                |      |      |              |      |      |

  对于文件夹：

  - r：可ls里面的内容
  - w：可以在里面创建、删除、改名
  - x：可cd到此

#### 5.2 <span id='chmod'>修改权限</span>

- chmod命令

  ```bash
  chmod [-R] 权限 文件或文件夹
  ```

  - -R：递归

  - 例子：

    ```bash
    chmod u=rwx,g=rx,o=x test.txt #所有者，所属组，其他用户
    chmod 751 test.txt
    
    chmod +x test2.txt #为三者都增加执行权限
    ```
    
    

#### 5.3 <span id='chown'>修改所属用户</span>

- chown命令

```bash
chown [-R] [用户][:用户组] 文件或文件夹
```

  

