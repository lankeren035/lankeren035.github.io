---
title: linux中ssh连接不上

date: 2025-2-10 12:00:00

tags: [Linux]

categories: [Linux]

comment: true

toc: true
---

#  

<!--more-->

#### 问题

- 本地使用ssh连接linux时发现无法连接，但是别的用户却可以

#### 解决

1. 查看配置文件

   ```shell
   vim /etc/ssh/sshd_config
   ```

2. 查看是否有： `AllowUsers 用户名`

3. 发现这里有别的用户，没有目标用户，把目标用户名加上

4. 重启服务然后就可以连上了

   ```shell
   sudo systemctl restart sshd
   ```

   