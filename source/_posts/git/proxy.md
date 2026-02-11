---
title: "git使用clash代理"

date: 2026-2-11 14:00:00

tags: [Git]

categories: [Git]

comment: true

toc: true

---



####



<!--more-->

## git使用clash代理

```shell
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
```



- 某个项目ssh走代理

  ```shell
  Host github.com
    HostName ssh.github.com
    Port 443
    User git
    IdentityFile ~/.ssh/id_rsa
    ProxyCommand connect -H 127.0.0.1:7890 %h %p
  ```

  

