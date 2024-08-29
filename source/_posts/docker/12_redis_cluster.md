---
title: 12- redis集群部署

date: 2024-8-29 12:00:00

tags: [docker]

categories: [docker]

comment: true

toc: true


---

#

<!--more-->

# 12- redis集群部署

- 创建6个节点，分为三个主机三个从机（从机类似备份）

  ```shell
  # 创建网卡
  docker network create redis --subnet 172.38.0.0/16
  
  # 创建6个redis配置
  for port in $(seq 1 6); \
  do \
  mkdir -p ./mydata/redis/node-${port}/conf
  touch ./mydata/redis/node-${port}/conf/redis.conf
  cat << EOF >./mydata/redis/node-${port}/conf/redis.conf
  port 6379
  bind 0.0.0.0
  cluster-enabled yes
  cluster-config-file nodes.conf
  cluster-node-timeout 5000
  cluster-announce-ip 172.38.0.1${port}
  cluster-announce-port 6379
  cluster-announce-bus-port 16379
  appendonly yes
  EOF
  
  sudo docker run -p 637${port}:6379 -p 1637${port}:16379 --name redis-${port} \
  -v ./mydata/redis/node-${port}/data:/data \
  -v ./mydata/redis/node-${port}/conf/redis.conf:/etc/redis/redis.conf \
  -d --net redis --ip 172.38.0.1${port} redis:5.0.9-alpine3.11 redis-server /etc/redis/redis.conf;
  done
  
docker ps
  ```
  
  - 如果发现没有正在运行的容器，说明出问题了（通过日志发现是配置文件没有被加载，发现前面的路径写错了）
  
    ```
    docker logs redis-1
    ```
  
    

- 随便进入一个容器：

  ```shell
  docker exec -it redis-1 /bin/sh #redis里面是没有bash的
  redis-cli --cluster create 172.38.0.11:6379 172.38.0.12:6379 172.38.0.13:6379 172.38.0.14:6379 172.38.0.15:6379 172.38.0.16:6379 --cluster-replicas 1 
  yes
  ```

  - 查看集群情况：

    ```shell
    redis-cli -c
    cluster info #可以看到cluster_size:3
    cluster nodes #可以看到节点情况
    set a b #在数据库中存储一对键值对a:b, 存储到了node3
    get a #输出b
    ```

- 打开另一个窗口：

  ```shell
  docker stop redis-3 #删除node3
  ```

- 第一个窗口中node3被停止了

  ```shell
  get a #由于刚刚停止了node3，这里还处于node3，因此显示无法连接
  ```

  - ctr + c

    ```shell
    redis-cli -c
    get a #此时转向了node4，说明有数据有备份
    cluster nodes #可以看到node3显示fail了
    ```

    