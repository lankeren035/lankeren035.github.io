---
title: vscode中插件无法使用
date: 2025-02-12 2:00:00
toc: true
tags: [vscode]
categories: [经验]

---

# 

<!--more-->

## 1. 问题

- 用vscode连接远程服务器，发现copilot插件无法安装：`Unterminate string in JSON at position 4096`，显示`安装时出错，尝试手动下载`，手动下载后也没用

## 2. 原因

- 可能是 vscode-server中一些扩展文件发生了一些未知的兼容性错误 

## 3. 解决

1. 将服务器端的`.vscode-server/extensions`删除，重启vscode连接服务器，重新安装扩展