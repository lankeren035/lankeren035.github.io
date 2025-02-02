---
title: Git连接不上Github

date: 2025-2-2 13:00:00

tags: [Git]

categories: [Git]

comment: true

toc: true


---

####

<!--more-->

# 1. hexo博客无法上传

## 2.0 现象

- 使用`hexo d`上传博客时发现无法上传，显示

  ```shell
  ssh: connect to host github.com port 22: Connection timed out
  fatal: Could not read from remote repository.
  ```

- 使用`git push`却可以上传
  - 因为`git push`使用的时`https`，而`hexo d`使用的是`ssh`

## 2.1 测试

```shell
ssh -T git@github.com
# 输出
# ssh: connect to host github.com port 22: Connection timed out

ssh -T git@ssh.github.com
# 可以连接
```

- 原因： 在国内/公司/学校等网络环境下，`github.com:22` 常被阻断，导致连接超时
-  GitHub 官方提供了 `ssh.github.com`（可以使用端口 22 或 443）用来替代 `github.com:22`

## 2.2 解决

### 1）方案1

- 在`user/username/.ssh/config`中添加：

  ```shell
  Host github.com
    HostName ssh.github.com
    Port 22
    User git
    IdentityFile ~/.ssh/id_rsa
  ```

### 2）方案2

- 将`_config.yml`中的`deploy`的`repo`部分修改：

  1. 使用https协议
  
  ```shell
  # repo: git@github.com:lankeren035/lankeren035.github.io.git
  # 改成
repo: https://github.com/lankeren035/lankeren035.github.io.git
  ```
  
  2. 使用`ssh.github.com`
  
  ```shell
  # repo: git@github.com:lankeren035/lankeren035.github.io.git
  # 改成
  repo: ssh://git@ssh.github.com:22/lankeren035/lankeren035.github.io.git
  ```
  
  