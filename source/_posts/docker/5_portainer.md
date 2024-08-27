---
title: 5- portainer可视化面板安装

date: 2024-8-26 13:00:00

tags: [docker]

categories: [docker]

comment: true

toc: true

---

#

<!--more-->

# 5- portainer可视化面板安装

- portainer（先用这个）

  ```shell
  docker run -d -p 8088:9000 --restart=always -v /var/run/docker.sock:/var/run/docker.sock --privileged=true portainer/portainer
  ```

- Rancher（CI/CD再用）

然后可以在浏览器访问：`ip:8088`