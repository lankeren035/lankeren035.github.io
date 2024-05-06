---
title: Linux下安装miniconda

date: 2024-4-23 08:00:00

tags: [安装]

categories: [安装]

comment: true

toc: true

---

#
<!--more-->

# Linux下安装miniconda
1. 下载：
```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

2. 安装：
```bash
sh Miniconda3-latest-Linux-x86_64.sh
```

    然后一路回车

3. 刷新终端
```bash
source ~/.bashrc
```

4. 检验
```bash
conda -V
```

5. 若要卸载，可直接删除已安装的文件夹后，并删除相应的环境变量：
```bash
rm -rf /usr/local/miniconda/
rm -rf /usr/local/anaconda/
```

6. 删除后，打开 ~/.bashrc 文件，删除以下conda的路径变量：
```bash
export PATH=" /usr/local/anaconda/bin:$PATH" 
export PATH=" /usr/local/miniconda3/bin:$PATH" 
```