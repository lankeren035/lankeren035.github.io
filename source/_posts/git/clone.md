---
title: "Git克隆项目中断 error: RPC 失败。curl 92 HTTP/2 stream 0 was not closed cleanly: CANCEL (err 8)"

date: 2025-2-11 13:00:00

tags: [Git]

categories: [Git]

comment: true

toc: true



---

####

<!--more-->



#### 使用git clone时发现老是无法克隆项目：

```shell
git clone https://github.com/guoyww/AnimateDiff.git
正克隆到 'AnimateDiff'...
remote: Enumerating objects: 718, done.
remote: Counting objects: 100% (38/38), done.
remote: Compressing objects: 100% (26/26), done.
fatal: 远端意外挂断了/718), 46.53 MiB | 2.34 MiB/s  
fatal: 过早的文件结束符（EOF）
fatal: index-pack 
```

- 网不好或者项目含有多个版本，一起下载空间较大，可以选择下载最新版本

  ```shell
  git clone --depth 1 https://github.com/guoyww/AnimateDiff.git
  ```

  

