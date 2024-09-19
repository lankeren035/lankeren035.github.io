---
title: tmux使用
date: 2024-03-11 22:00:00
toc: true
tags: [Linux]
categories: [Linux]


---

#

<!--more-->



# Linux中使用tmux

-  假设你在本地windows powershell中连接了远端服务器，然后打开了服务器的tmux，然后在tmux中运行了一个python代码，代码还没跑完就将本地的windows powershell关掉了，但是你的服务器还能接着跑代码。



- 

  |              |                                 |
  | ------------ | ------------------------------- |
  | 查看会话     | tmux ls                         |
  | 删除会话     | tmux kill-session -t 会话名或id |
  | 激活会话     | tmux attach -t 会话名或id       |
  | 退出当前会话 | ctrl+d                          |

- 

  |                  |                         |
  | ---------------- | ----------------------- |
  | 向左打开一个窗口 | ctrl+b %                |
  | 向下打开一个窗口 | ctrl+b "                |
  | 调整窗口大小     | ctrl+b+方向（狂按方向） |
|                  |                         |
  
  

### 打开鼠标滚轮
```
ctrl + b
:
set -g mouse on
```


<details> <summary>点击展开/折叠 \u25BC &#9660</summary>
这里是折叠的内容。

你可以在这里添加更多的文字、代码或其他内容。

</details>