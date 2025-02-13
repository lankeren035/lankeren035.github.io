---
title:  "conda 常用命令"

date:  2025-2-10 1:00:00

tags:  [命令, conda]

categories:  [命令]

comment:  false

toc:  true


---

#

<!--more-->



|                      |                                                              |      |
| -------------------- | ------------------------------------------------------------ | ---- |
| 创建虚拟环境         | conda  create --name python310 python=3.10 <br>python -m venv 项目目录/venv |      |
| 删除虚拟环境         | conda  remove -n 名称 --all                                  |      |
| 激活虚拟环境         | conda activate 环境名称<br>source  ./venv/bin/activate       |      |
| 退出虚拟环境         | [conda] deactivate                                           |      |
| 查看环境             | conda info --envs                                            |      |
| 安装requirements.txt | conda  install --file requirements.txt                       |      |
| 清理pkg              | conda clean  --packages                                      |      |

