---
title: vscode中远程python代码没有自动补全
date: 2025-02-12 3:00:00
toc: true
tags: [vscode]
categories: [经验]


---

#

<!--more-->

## 1. 问题

- 使用vscode连接服务器，在服务器上写python代码发现没有自动补全，本地写代码有自动补全。



## 2. 解决

- 进入设置，搜索：`setting`，点击进入配置文件，加入下面的设置：

  ```json
  "python.languageServer": "Jedi"
  ```

  