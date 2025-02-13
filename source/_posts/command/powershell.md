---
title:  "powershell 常用命令"

date:  2025-2-10 3:00:00

tags:  [命令, powershell]

categories:  [命令]

comment:  false

toc:  true



---

#

<!--more-->



|                    |                                                              |      |
| ------------------ | ------------------------------------------------------------ | ---- |
| 设置代理           | \$env:HTTP_PROXY="http://127.0.0.1:7890"<br/>​\$env:HTTPS_PROXY="https://127.0.0.1:7890" |      |
| 查看端口占用       | netstat  -ano \| findstr :28000                              |      |
| 杀掉进程           | taskkill  /PID 17896 /F                                      |      |
| 查看目录下图片数量 | Get-ChildItem  -Recurse -File -Include *.jpg, *.jpeg, *.png, *.gif, *.bmp, *.tiff \| Measure-Object \|  Select-Object -Property Count |      |
|                    |                                                              |      |
|                    |                                                              |      |
|                    |                                                              |      |

