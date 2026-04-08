---
title: Obsidian使用
date: 2026-04-03 11:01:27
published: true
permalink: obsidian-notes
tags:
  - 经验
  - Obsidian
hexo-path: /2026/04/03/experience/obsidian/Obsidian
---
#
<!--more-->

# Obsidian 使用

## 1. 目标

代码学习。目标如下：

- 记录整个项目的主流程
- 记录模块、文件、函数的作用
- 形成一张总蓝图
- 后续修改项目时，这套笔记可以直接作为蓝图使用

本次采用的工具：

- Obsidian
- Canvas
- Excalidraw（可选）

---

## 2. vault 的概念

Obsidian 的项目本质上就是一个本地文件夹。这个文件夹里包含：

- `.md` 笔记
- `.canvas` 画布
- `.obsidian` 配置
- 图片和附件

所以后面同步 Obsidian 项目，本质上就是同步整个 vault 文件夹。

---


## 4. 启用 Canvas 和安装 Excalidraw

### 4.1 启用 Canvas

操作路径：

- `Settings`
- `Core plugins`
- 打开 `Canvas`
后续使用：
- 邮件新建白板
### 4.2 安装 Excalidraw

操作路径：

- `Settings`
- `Community plugins`
- 关闭 `Restricted mode`
- `Browse`
- 搜索 `Excalidraw`
- 安装并启用
后续使用：
- 邮件新建绘图文件

说明：

- Canvas：适合做总蓝图
- Excalidraw：适合画局部流程图

---

## 5. 推荐的笔记分层

不要把所有内容都写进一篇大文档。建议拆成多层。
- 模块层
- 文件层
- 函数层

## 6. 内部链接


例如在 `training().md` 里写：

```md
[[train.py]]
[[Scene]]
[[GaussianModel]]
[[render()]]
```


## 10. 如何避免误创建的笔记跑到根目录

如果点击了未解析链接，Obsidian 会按默认设置决定新笔记放在哪里。

建议修改：

- `Settings`
- `Files and links`
- `Default location for new notes`

设置为：

```text
Same folder as current file
```



# 2. Obsidian笔记多台设备同步
- [[Syncthing]]
# 3. excalidraw的使用
- [[excalidraw]]
