---
title: "Wan2.1"
date: 2025-03-4 15:00:00
toc: true
tags: [Linux]
categories: [Linux]

---

#

<!--more-->

- 环境（先手动安装torch>2.4）

```shell
conda create -n py310 python=3.10 -y
pip3 install torch torchvision torchaudio
pip3 install ninja
git clone https://github.com/Wan-Video/Wan2.1.git
cd Wan2.1
pip install -r requirements.txt
```

