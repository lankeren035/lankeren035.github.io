---
title: Milvus安装
date: 2026-09-14
tags:
  - 经验
categories:
  - 经验
comment: true
toc: true
published: true
permalink: experience/test
hexo-path:
---
#
<!--more-->
#
<!--more-->

1. **问题现象**
    
    - `docker compose up -d` 显示 `milvus-standalone` 已启动，但很快从 `docker ps` 中消失。
        
    - `docker ps -a` 显示：
        
        ```text
        milvus-standalone   Exited (134)
        ```
        
    - 说明 Milvus 启动后异常退出。
        
2. **查看 Milvus 日志**
    
    ```bash
    docker logs milvus-standalone
    ```
    
    关键输出：
    
    ```text
    init with etcd failed
    error="context deadline exceeded"
    
    panic: failed to create etcd client: context deadline exceeded
    ```
    
    说明根因是：
    
    ```text
    Milvus → etcd:2379 连接超时
    ```
    
    不是 Milvus 本身启动参数问题。
    
3. **排除 OOM**
    
    ```bash
    docker inspect milvus-standalone \
      --format 'OOM={{.State.OOMKilled}} Exit={{.State.ExitCode}}'
    ```
    
    输出：
    
    ```text
    OOM=false Exit=134
    ```
    
    说明不是内存不足。
    
4. **验证 etcd 自身是否正常**
    
    ```bash
    docker exec milvus-etcd \
      etcdctl --endpoints=http://127.0.0.1:2379 endpoint health
    ```
    
    输出：
    
    ```text
    127.0.0.1:2379 is healthy
    ```
    
    说明：
    
    ```text
    etcd 服务本身正常
    ```
    
5. **验证 Docker DNS**  
    从同一个 `milvus` 网络启动临时容器：
    
    ```bash
    docker run --rm \
      --network milvus \
      --entrypoint /bin/sh \
      milvusdb/milvus:v2.5.14 \
      -c 'getent hosts etcd'
    ```
    
    得到：
    
    ```text
    172.18.0.3 etcd
    ```
    
    说明：
    
    ```text
    Docker DNS 正常
    etcd 名称可以正确解析
    ```
    
6. **验证容器间 TCP 通信**
    
    ```bash
    curl http://etcd:2379/health
    ```
    
    结果：
    
    ```text
    connect to 172.18.0.3:2379 failed
    Connection timed out
    ```
    
    MinIO 同样：
    
    ```text
    minio:9000 → timeout
    ```
    
    所以可以确定：
    
    ```text
    DNS 正常
    服务正常
    但 Docker 容器之间的网络转发被阻断
    ```
    
7. **检查 Docker 网络**
    
    ```bash
    docker network inspect milvus
    ```
    
    得到：
    
    ```text
    subnet: 172.18.0.0/16
    minio: 172.18.0.2
    etcd:  172.18.0.3
    ```
    
    网络配置本身正常。
    
8. **检查 iptables**  
    普通 `iptables`：
    
    ```bash
    sudo iptables -S FORWARD
    ```
    
    显示 `ACCEPT`，但同时提示：
    
    ```text
    iptables-legacy tables present
    ```
    
    因此继续检查：
    
    ```bash
    sudo iptables-legacy -S FORWARD
    ```
    
    得到：
    
    ```text
    -P FORWARD DROP
    ```
    
    并且现有规则只放行：
    
    ```text
    docker0
    ```
    
    没有放行自定义 Docker bridge：
    
    ```text
    milvus → 172.18.0.0/16
    ```
    
    **最终根因：**
    
    ```text
    系统同时存在 iptables-nft 和 iptables-legacy。
    
    iptables-legacy 的 FORWARD 默认策略是 DROP，
    并且只放行 docker0，
    导致 milvus 自定义 bridge 网络中的容器互相访问被丢弃。
    ```
    
9. **解决方法**  
    临时放行 `milvus` 网络内部通信：
    
    ```bash
    sudo iptables-legacy -I FORWARD 1 \
      -s 172.18.0.0/16 \
      -d 172.18.0.0/16 \
      -j ACCEPT
    ```

	或者
```bash
    SUBNET=$(docker network inspect milvus \
  --format '{{(index .IPAM.Config 0).Subnet}}')

sudo iptables-legacy -C FORWARD \
  -s "$SUBNET" -d "$SUBNET" -j ACCEPT 2>/dev/null || \
sudo iptables-legacy -I FORWARD 1 \
  -s "$SUBNET" -d "$SUBNET" -j ACCEPT
```
    再测试：
    
    ```bash
    curl http://etcd:2379/health
    ```
    
    成功返回，MinIO 也恢复访问。
    
    最后启动 Milvus：
    
    ```bash
    docker start milvus-standalone
    ```
    
