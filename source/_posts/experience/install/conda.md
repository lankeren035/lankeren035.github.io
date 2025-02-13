---
title: Linux下安装conda

date: 2024-10-13 08:00:00

tags: [安装]

categories: [安装]

comment: true

toc: true


---

#
<!--more-->

# Linux下安装conda

```shell
curl -L -O https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh #下载
bash Anaconda3-2024.06-1-Linux-x86_64.sh -b #安装 -b非交互式安装，自动接受许可协议
conda -V
conda init bash
source ~/.bashrc
```

- 可能出现下载过程崩溃的问题：

```
WARNING: md5sum mismatch of tar archive
expected: a9c1b381ebd833088072d1e133217d05
     got: b01bf8a3030b99a35e211d2173d70287  -
Unpacking payload ...
[48178] Cannot open PyInstaller archive from executable (/home/user/anaconda3/_conda) or external archive (/home/user/anaconda3/_conda.pkg)
```

- 安装后如果无法使用conda命令

  ```shell
  ~/anaconda3/bin/conda init bash
  source ~/.bashrc
  conda -V
  
  ```

  