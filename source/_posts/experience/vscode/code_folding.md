---
title: vscode代码无法缩进
date: 2025-12-5 3:00:00
toc: true
tags: [vscode]
categories: [经验]



---

#

<!--more-->



### vscode代码无法缩进

VS Code 有两种折叠策略：一种是基于缩进（Indentation），一种是基于语言服务（Language Server）。有时候自动模式会失效。

- **解决方法：**

  1. 打开设置 (`Ctrl + ,` 或 `Cmd + ,`)。
  2. 搜索 `folding strategy`。
  3. 找到 **Editor: Folding Strategy**。
  4. 将其从 `auto` 更改为 **`indentation`**。

  - *注意：改为 `indentation` 强制 VS Code 仅根据缩进层级来折叠，这对 Python 非常有效。*