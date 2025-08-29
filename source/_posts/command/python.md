---
title:  "python 常用命令"

date:  2025-2-10 5:00:00

tags:  [命令, python]

categories:  [命令]

comment:  false

toc:  true




---

#

<!--more-->

## 1. pip

|                 |                                  |      |
| --------------- | -------------------------------- | ---- |
| pip安装包       | pip install -r  requirements.txt |      |
| pip查看可用版本 | pip index versions numpy         |      |
| pip升级         | pip install --upgrade pip        |      |
|                 |                                  |      |
|                 |                                  |      |
|                 |                                  |      |

## 2. 运行

|         |                                                              |
| ------- | ------------------------------------------------------------ |
| 调试    | python -m debugpy --listen localhost:5678 --wait-for-client $(which torchrun) --nnodes=1 --nproc_per_node=1 train.py --config configs/training/v1/config.yaml<br>python -m debugpy --listen localhost:xxxx --wait-for-client main.py --参数等等 |
| 指定GPU | CUDA_VISIBLE_DEVICES=gpu_ids test.py                         |
|         |                                                              |

## 3. huggingface

|      |                                                              |
| ---- | ------------------------------------------------------------ |
|      | pip install "huggingface_hub[cli]"<br/>huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./Wan2.1-T2V-14B |
|      |                                                              |

