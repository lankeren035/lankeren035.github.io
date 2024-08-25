---
title: 3- docker命令

date: 2024-8-25 12:00:00

tags: [docker]

categories: [docker]

comment: true

toc: true





---

#

<!--more-->

# 3- docker命令

## 3.1 帮助命令

```shell
docker version #显示docker版本信息
docker info #显示docker系统信息，包括镜像和容器数量
docker 命令 --help #帮助命令
```

- [帮助文档](https://docs.docker.com/reference/cli/docker/)



## 3.2 镜像命令

| 命令                               | 参数                                                      | 解释                                                         |
| ---------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------ |
| `docker images`                    | -a，--all   #列出所有镜像 <br>-q， --quiet  #只显示镜像id | REPOSITORY：镜像的仓库源<br>TAG：镜像的标签<br>IMAGE ID：镜像id |
| docker search 内容                 | --filter=STARS=3000  #搜索stars大于3000的                 | 从docker hub搜索镜像                                         |
| docker pull 名称[:tag]             | -a<br>-q                                                  | 下载镜像, tag是版本                                          |
| docker rmi 名称/id1 名称/id2       | -f                                                        | 删除镜像                                                     |
| docker rmi -f $(docker images -aq) |                                                           | 删除全部镜像                                                 |



## docker pull

```shell
# 下载镜像： docker pull 镜像名[:tag]
docker pull mysql
```

```shell
using default tag: latest #如果不写tag, 默认latest
latest: pulling form library/mysql
5b54d594fba7: Pull complete #分层下载， docker image的核心，联合文件系统
07e7d6a8a868: Pull complete
dd8f4d07efa5: Pull complete
076d396a6205: Pull complete
cf6b2b93048f: Pull complete
530904b4a8b7: Pull complete
fb1e55059a95: Pull complete
4bd29a0dcde8: Pull complete
b94a001c6ec7: Pull complete
cb77cbeb422b: Pull complete
2a35cdbd42cc: Pull complete
Digest: sha256:dc255ca50a42b3589197000b1f9bab2b4e010158d1a9f56c3db6ee145506f625 # 签名信息，防伪标志
Status: Downloaded newer image for mysql:latest
docker.io/library/mysql:latest #真实地址 等价于：docker pull docker.io/library/mysql:latest
```

- 分层下载的好处是可以共用。



## 3.3 容器命令

