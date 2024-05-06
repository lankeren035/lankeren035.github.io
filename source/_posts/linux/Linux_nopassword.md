---
title: Linux服务器免密登录
date: 2024-03-09 22:00:00
toc: true
tags: [Linux]
categories: [Linux]

---

#

<!--more-->

# Linux服务器免密登录

## 1 生成本地公钥

- 本地运行（可以先到user/.ssh下查看有没有id_rsa.pub文件，有就不用生成了）：

  ```bash
  ssh-keygen
  ```

  一直回车就行。

## 2 将生成的公钥上传到服务器

- 在服务器端创建~/.ssh

  ```bash
  mkdir .ssh
  ```

  

- 本地cd到：user/用户名/.ssh。将id_rsa.pub复制到服务器的~/.ssh/authorized_keys（root用户的根目录与普通用户的根目录不同）

  ```bash
  scp -P 端口 id_rsa.pub 用户名@IP:~/.ssh/authorized_keys
  ```

  