---
title: vscode远程主机不满足运行vscode服务器的先决条件
date: 2025-04-21 20:00:00
toc: true
tags: [vscode]
categories: [vscode]
---

#

<!--more-->

## 原因

- vscode有更新，对低版本的GLIBC不再满足需求。 

## 解决

- 防止vscode自动更新：设置->搜索: update -> Update: Mode设置成none