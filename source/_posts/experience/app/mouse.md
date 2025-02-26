---
title: 鼠标侧键更改

date: 2024-3-18 08:00:00

tags: [鼠标]

categories: [经验]

comment: true

toc: true


---

#
<!--more-->

# 更改鼠标侧键

- 原始的鼠标侧键是在浏览器/文件资源管理器中上一页下一页，想将其修改为其他快捷键。

## 1. X-Mouse Button Control

- [下载X-Mouse Button Control](https://www.highrez.co.uk/scripts/download.asp?package=XMouse)

- 找到侧键位置，更改即可

  ![](../../../../themes/yilia/source/img/experience/app/mouse/1.png)

  ![](img/experience/app/mouse/1.png)

  - 此外还可以自己设置不同的方案等。

## 2. autohotkey

- [下载](https://www.autohotkey.com/)(两个都要下载)

   ![](../../../../themes/yilia/source/img/experience/app/mouse/2.png)

  ![](img/experience/app/mouse/2.png) 

- 新建一个文件：

     ![](../../../../themes/yilia/source/img/experience/app/mouse/3.png)

    ![](img/experience/app/mouse/3.png) 

- 写入如下内容，实现短按、长按不同功能：

  ```ahk
  #Persistent
  #UseHook
  
  XButton1::
  KeyWait, XButton1, T0.2
  If ErrorLevel ; 表示按键被长按
      Send, #{v} ; 模拟按下 Win+V
  Else
      Send, {Enter} ; 模拟按下 enter
  Return
  
  XButton2::
  KeyWait, XButton2, T0.2
  If ErrorLevel ; 表示按键被长按
      Send, ^{v} ; 模拟按下 Ctrl+V
  Else
      Send, ^{c} ; 模拟按下 Ctrl+C
  Return
  ```

  双击运行即可。
  
- 将该文件放到：`C:\Users\123\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup`开机自启动。