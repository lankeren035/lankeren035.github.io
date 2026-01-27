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

  |              |                                                   |
  | ------------ | ------------------------------------------------- |
  | 新建会话     | tmux<br>tmux new -s 会话名                        |
  | 查看会话     | tmux ls                                           |
  | 删除会话     | tmux kill-session -t 会话名或id                   |
  | 激活会话     | tmux a -t 会话名或id<br>tmux attach -t 会话名或id |
  | 重命名会话   | tmux rename-session -t 旧名 新名 <br>Ctrl+b $     |
  | 切换会话     | tmux switch -t 会话名                             |
  | 退出当前会话 | ctrl+d                                            |

- 

  |                      |                         |
  | -------------------- | ----------------------- |
  | 向左打开一个窗口     | ctrl+b %                |
  | 向下打开一个窗口     | ctrl+b "                |
  | 调整窗口大小         | ctrl+b+方向（狂按方向） |
| 光标切换到上方窗格   | Ctrl+b 上               |
  | 光标切换到另一个窗格 | ctrl+b o                |
  | 查看窗口编号         | ctrl+b w                |
  | 关闭当前子窗口       | ctrl+d<br>exit          |
  | 复制模式             | ctrl+b [                |
  
  
  
  

### 打开鼠标滚轮

```
ctrl + b
:
set -g mouse on
```

### 将进程添加到tmux

- 安装reptyr

    ```bash
    sudo apt install reptyr
    ```
```

- 启动tmux

- 添加进程（如果进程有子进程则无法添加）

  ```bash
  reptyr <PID>
```

  



