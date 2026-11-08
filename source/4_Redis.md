---
title: 4_Redis
date: 2026-11-04
tags:
comment: true
toc: true
published: true
permalink:
hexo-path:
---
#
<!--more-->

# 1. 简介

Redis（Remote Dictionary Server）是一个开源的内存数据库，遵守 BSD 协议，它提供了一个高性能的键值（key-value）存储系统，常用于缓存、消息队列、会话存储等应用场景。

- **性能极高：** Redis 以其极高的性能而著称，能够支持每秒数十万次的读写操作24。尤其是在缓存、会话管理、排行榜等。
- **丰富的数据类型：** Redis 不仅支持基本的键值存储，还提供了丰富的数据类型，包括**字符串、列表、集合、哈希表、有序集合**等。
- **原子性操作：** Redis 的所有操作都是原子性的，对于确保数据的一致性和完整性至关重要，尤其是在高并发环境下处理事务时。
- **持久化：** 可以将内存中的数据保存到磁盘中
- **支持发布/订阅模式：** Redis 内置了发布/订阅模式（Pub/Sub），允许客户端之间通过消息传递进行通信。这使得 Redis 可以作为消息队列和实时数据传输的平台。
- **单线程模型：** 尽管 Redis 是单线程的，但它通过高效的事件驱动模型来处理并发请求，确保了高性能和低延迟。单线程模型也简化了并发控制的复杂性。
- **主从复制：** Redis 支持主从复制，可以通过从节点来备份数据或分担读请求，提高数据的可用性和系统的伸缩性。
- **应用场景广泛：** 缓存系统、会话存储、排行榜、实时分析、地理空间数据索引等。
- **社区支持：**
- **跨平台兼容性：**


Redis 与其他 key-value 存储有什么不同？
- **丰富的数据类型：** Redis 不仅仅支持简单的 key-value 类型的数据，还提供了 list、set、zset（有序集合）、hash 等数据结构的存储。
- **高性能的读写能力：** 这种高性能主要得益于 Redis 将数据存储在内存中。
- **原子性操作：** 对于确保数据的一致性和完整性非常重要。
- **持久化机制：**
- **丰富的特性集：** Redis 还支持 publish/subscribe（发布/订阅）模式、通知、key 过期等高级特性。这些特性使得 Redis 可以用于消息队列、实时数据分析等复杂的应用场景。
- **主从复制和高可用性：** 
- **支持 Lua 脚本：** Redis 支持使用 Lua 脚本来编写复杂的操作，这些脚本可以在服务器端执行，提供了更多的灵活性和强大的功能。
- **单线程模型：**

# 2. 安装

# 3. 配置
```shell
CONFIG GET *   #读取运行时配置
CONFIG GET CONFIG_SETTING_NAME   #查看指定配置文件
CONFIG SET CONFIG_SETTING_NAME NEW_CONFIG_VALUE #修改配置
```

- 参数说明：

|序号|配置项|说明|
|---|---|---|
|1|daemonize no|Redis 默认不是以守护进程的方式运行，可以通过该配置项修改，使用 yes 启用守护进程（Windows 不支持守护线程的配置为 no ）|
|2|pidfile /var/run/redis.pid|当 Redis 以守护进程方式运行时，Redis 默认会把 pid 写入 /var/run/redis.pid 文件，可以通过 pidfile 指定|
|3|port 6379|指定 Redis 监听端口，默认端口为 6379，作者在自己的一篇博文中解释了为什么选用 6379 作为默认端口，因为 6379 在手机按键上 MERZ 对应的号码，而 MERZ 取自意大利歌女 Alessia Merz 的名字|
|4|bind 127.0.0.1|绑定的主机地址|
|5|timeout 300|当客户端闲置多长秒后关闭连接，如果指定为 0 ，表示关闭该功能|
|6|loglevel notice|指定日志记录级别，Redis 总共支持四个级别：debug、verbose、notice、warning，默认为 notice|
|7|logfile stdout|日志记录方式，默认为标准输出，如果配置 Redis 为守护进程方式运行，而这里又配置为日志记录方式为标准输出，则日志将会发送给 /dev/null|
|8|databases 16|设置数据库的数量，默认数据库为0，可以使用SELECT 命令在连接上指定数据库id|
|9|save <seconds> <changes><br><br>Redis 默认配置文件中提供了三个条件：<br><br>**save 900 1**<br><br>**save 300 10**<br><br>**save 60 10000**<br><br>分别表示 900 秒（15 分钟）内有 1 个更改，300 秒（5 分钟）内有 10 个更改以及 60 秒内有 10000 个更改。|指定在多长时间内，有多少次更新操作，就将数据同步到数据文件，可以多个条件配合|
|10|rdbcompression yes|指定存储至本地数据库时是否压缩数据，默认为 yes，Redis 采用 LZF 压缩，如果为了节省 CPU 时间，可以关闭该选项，但会导致数据库文件变的巨大|
|11|dbfilename dump.rdb|指定本地数据库文件名，默认值为 dump.rdb|
|12|dir ./|指定本地数据库存放目录|
|13|slaveof <masterip> <masterport>|设置当本机为 slave 服务时，设置 master 服务的 IP 地址及端口，在 Redis 启动时，它会自动从 master 进行数据同步|
|14|masterauth <master-password>|当 master 服务设置了密码保护时，slave 服务连接 master 的密码|
|15|requirepass foobared|设置 Redis 连接密码，如果配置了连接密码，客户端在连接 Redis 时需要通过 AUTH <password> 命令提供密码，默认关闭|
|16|maxclients 128|设置同一时间最大客户端连接数，默认无限制，Redis 可以同时打开的客户端连接数为 Redis 进程可以打开的最大文件描述符数，如果设置 maxclients 0，表示不作限制。当客户端连接数到达限制时，Redis 会关闭新的连接并向客户端返回 max number of clients reached 错误信息|
|17|maxmemory <bytes>|指定 Redis 最大内存限制，Redis 在启动时会把数据加载到内存中，达到最大内存后，Redis 会先尝试清除已到期或即将到期的 Key，当此方法处理 后，仍然到达最大内存设置，将无法再进行写入操作，但仍然可以进行读取操作。Redis 新的 vm 机制，会把 Key 存放内存，Value 会存放在 swap 区|
|18|appendonly no|指定是否在每次更新操作后进行日志记录，Redis 在默认情况下是异步的把数据写入磁盘，如果不开启，可能会在断电时导致一段时间内的数据丢失。因为 redis 本身同步数据文件是按上面 save 条件来同步的，所以有的数据会在一段时间内只存在于内存中。默认为 no|
|19|appendfilename appendonly.aof|指定更新日志文件名，默认为 appendonly.aof|
|20|appendfsync everysec|指定更新日志条件，共有 3 个可选值：<br><br>- **no**：表示等操作系统进行数据缓存同步到磁盘（快）<br>- **always**：表示每次更新操作后手动调用 fsync() 将数据写到磁盘（慢，安全）<br>- **everysec**：表示每秒同步一次（折中，默认值）|
|21|vm-enabled no|指定是否启用虚拟内存机制，默认值为 no，简单的介绍一下，VM 机制将数据分页存放，由 Redis 将访问量较少的页即冷数据 swap 到磁盘上，访问多的页面由磁盘自动换出到内存中（在后面的文章我会仔细分析 Redis 的 VM 机制）|
|22|vm-swap-file /tmp/redis.swap|虚拟内存文件路径，默认值为 /tmp/redis.swap，不可多个 Redis 实例共享|
|23|vm-max-memory 0|将所有大于 vm-max-memory 的数据存入虚拟内存，无论 vm-max-memory 设置多小，所有索引数据都是内存存储的(Redis 的索引数据 就是 keys)，也就是说，当 vm-max-memory 设置为 0 的时候，其实是所有 value 都存在于磁盘。默认值为 0|
|24|vm-page-size 32|Redis swap 文件分成了很多的 page，一个对象可以保存在多个 page 上面，但一个 page 上不能被多个对象共享，vm-page-size 是要根据存储的 数据大小来设定的，作者建议如果存储很多小对象，page 大小最好设置为 32 或者 64bytes；如果存储很大大对象，则可以使用更大的 page，如果不确定，就使用默认值|
|25|vm-pages 134217728|设置 swap 文件中的 page 数量，由于页表（一种表示页面空闲或使用的 bitmap）是在放在内存中的，，在磁盘上每 8 个 pages 将消耗 1byte 的内存。|
|26|vm-max-threads 4|设置访问swap文件的线程数,最好不要超过机器的核数,如果设置为0,那么所有对swap文件的操作都是串行的，可能会造成比较长时间的延迟。默认值为4|
|27|glueoutputbuf yes|设置在向客户端应答时，是否把较小的包合并为一个包发送，默认为开启|
|28|hash-max-zipmap-entries 64<br>hash-max-zipmap-value 512|指定在超过一定的数量或者最大的元素超过某一临界值时，采用一种特殊的哈希算法|
|29|activerehashing yes|指定是否激活重置哈希，默认为开启（后面在介绍 Redis 的哈希算法时具体介绍）|
|30|include /path/to/local.conf|指定包含其它的配置文件，可以在同一主机上多个Redis实例之间使用同一份配置文件，而同时各个实例又拥有自己的特定配置文件|
# 4. 数据类型
Redis中的value可以取如下类型，其中string是最基本的类型：
- string（字符串）
- hash（哈希）
- list（列表）
- set（集合）
- **zset(sorted set：有序集合):**  类似于集合，但是每个元素都有一个分数（score）与之关联。
- **位图（Bitmaps）：** 基于字符串类型，可以对每个位进行操作。
- **超日志（HyperLogLogs）：** 用于基数统计，可以估算集合中的唯一元素数量。
- **地理空间（Geospatial）：** 用于存储地理位置信息。
- **发布/订阅（Pub/Sub）：** 一种消息通信模式，允许客户端订阅消息通道，并接收发布到该通道的消息。
- **流（Streams）：** 用于消息队列和日志存储，支持消息的持久化和时间排序。
- **模块（Modules）：** Redis 支持动态加载模块，可以扩展 Redis 的功能。

---

## 4.1 String（字符串）
- redis **最基本的类型**，最大能存储 512MB。string 类型是二进制安全的。意思是 redis 的 string 可以包含任何数据，比如jpg图片或者序列化的对象。

常用命令：
- `SET key value`：设置键的值。
- `GET key`：获取键的值。
- `INCR key`：将键的值加 1。
- `DECR key`：将键的值减 1。
- `APPEND key value`：将值追加到键的值之后。

---

## 4.2 Hash（哈希）

- Redis hash 是一个 string 类型的 field 和 value 的映射表，hash 特别适合用于存储对象。每个哈希最多可以存储 2^32 - 1 个键值对。

常用命令：

- `HSET key field value`：设置哈希表中字段的值。
- `HGET key field`：获取哈希表中字段的值。
- `HGETALL key`：获取哈希表中所有字段和值。
- `HDEL key field`：删除哈希表中的一个或多个字段。

---

## 4.3 List（列表）

- Redis 列表是简单的字符串列表，按照插入顺序排序。列表最多可以存储 2^32 - 1 个元素。

常用命令：

- `LPUSH key value`：将值插入到列表头部。
- `RPUSH key value`：将值插入到列表尾部。
- `LPOP key`：移出并获取列表的第一个元素。
- `RPOP key`：移出并获取列表的最后一个元素。
- `LRANGE key start stop`：获取列表在指定范围内的元素。

---

## 4.4 Set（集合）

- Redis 的 Set 是 string 类型的无序集合。集合是通过哈希表实现的，所以添加，删除，查找的复杂度都是 O(1)。集合中最大的成员数为 232 - 1(4294967295, 每个集合可存储40多亿个成员)。

常用命令：

- `SADD key value`：向集合添加一个或多个成员。
- `SREM key value`：移除集合中的一个或多个成员。
- `SMEMBERS key`：返回集合中的所有成员。
- `SISMEMBER key value`：判断值是否是集合的成员。

---

## 4.5 zset(sorted set：有序集合)

- 每个元素都会关联一个double类型的分数。redis正是通过分数来为集合中的成员进行从小到大的排序。zset的成员是唯一的,但分数(score)却可以重复。

常用命令：

- `ZADD key score value`：向有序集合添加一个或多个成员，或更新已存在成员的分数。
- `ZRANGE key start stop [WITHSCORES]`：返回指定范围内的成员。
- `ZREM key value`：移除有序集合中的一个或多个成员。
- `ZSCORE key value`：返回有序集合中，成员的分数值。
---

## 4.6 其他高级数据类似

### 4.6.1 HyperLogLog

- 用于基数估计算法的数据结构。
- 常用于统计唯一值的近似值。

### 4.6.2 Bitmaps

- 位数组，可以对字符串进行位操作。
- 常用于实现布隆过滤器等位操作。

### 4.6.3 Geospatial Indexes

- 处理地理空间数据，支持地理空间索引和半径查询。

### 4.6.4 Streams

- 日志数据类型，支持时间序列数据。
- 用于消息队列和实时数据处理。

# 5. 命令
- redis 命令可以写在 `.redis` 或 `.txt` 文件中，然后通过 `redis-cli` 批量执行
## 5.1 键（key）
```
<命令> <键名>
```

| 序号  | 命令                                           | 解释                                                                                |
| --- | -------------------------------------------- | --------------------------------------------------------------------------------- |
| 1   | **DEL**                                      | 删除                                                                                |
| 2   | **DUMP**                                     | 序列化给定 key ，返回该数据的二进制序列化结果。                                                        |
| 3   | **EXISTS**                                   | 是否存在                                                                              |
| 4   | **EXPIRE** key **seconds**                   | 为给定 key 设置过期时间，以秒计。                                                               |
| 5   | **EXPIREAT** key **timestamp**               | 为给定 key 设置过期时间，参数是 UNIX 时间戳。                                                      |
| 6   | **PEXPIRE** key **milliseconds**             | 设置 key 的过期时间，以毫秒计。                                                                |
| 7   | **PEXPIREAT** key **milliseconds-timestamp** | 设置 key 过期时间的时间戳(unix timestamp) 以毫秒计                                              |
| 8   | **KEYS**  **pattern**                        | 查找所有符合给定模式( pattern)的 key 。                                                       |
| 9   | **MOVE** key **db**                          | 将当前数据库的 key 移动到给定的数据库 db 当中。                                                      |
| 10  | **PERSIST** key                              | 移除 key 的过期时间，key 将持久保持。                                                           |
| 11  | **PTTL** key                                 | 以毫秒为单位返回 key 的剩余的过期时间。                                                            |
| 12  | **TTL** key                                  | 以秒为单位，返回给定 key 的剩余生存时间(TTL, time to live)。                                        |
| 13  | **RANDOMKEY**                                | 从当前数据库中随机返回一个 key 。                                                               |
| 14  | **RENAME** key **newkey**                    | 修改 key 的名称                                                                        |
| 15  | **RENAMENX** key **newkey**                  | 仅当 newkey 不存在时，将 key 改名为 newkey 。                                                 |
| 16  | SCAN cursor [MATCH pattern] [COUNT count]    | 迭代数据库中的数据库键。cursor游标，第一次从 0 开始。MATCH pattern指定匹配模式，可选。COUNT count每次迭代希望返回的元素数量，可选 |
| 17  | **TYPE** key                                 | 返回 key 所储存的值的类型。                                                                  |
更多命令请参考：[https://redis.io/commands](https://redis.io/commands)

## 5.2 字符串（String）

|序号|命令|解释|
|---|---|---|
|1|SET key value|设置指定 key 的值。|
|2|GET key|获取指定 key 的值。|
|3|GETRANGE key start end|返回 key 中字符串值的指定子字符串。|
|4|GETSET key value|将给定 key 的值设为 value，并返回 key 的旧值。|
|5|GETBIT key offset|获取 key 所储存的字符串值在指定偏移量上的位（bit）。|
|6|MGET key1 [key2 ...]|获取一个或多个给定 key 的值。|
|7|SETBIT key offset value|设置或清除 key 所储存的字符串值在指定偏移量上的位（bit）。|
|8|SETEX key seconds value|设置 key 的值，并将过期时间设为指定秒数。|
|9|SETNX key value|只有当 key 不存在时，才设置 key 的值。|
|10|SETRANGE key offset value|从指定偏移量开始，用 value 覆写 key 原来存储的字符串值。|
|11|STRLEN key|返回 key 所储存的字符串值的长度。|
|12|MSET key1 value1 [key2 value2 ...]|同时设置一个或多个 key-value 键值对。|
|13|MSETNX key1 value1 [key2 value2 ...]|当且仅当所有给定的 key 都不存在时，同时设置多个 key-value 键值对。|
|14|PSETEX key milliseconds value|设置 key 的值，并将过期时间设为指定毫秒数。|
|15|INCR key|将 key 中储存的数字值加 1。|
|16|INCRBY key increment|将 key 所储存的值增加指定的整数。|
|17|INCRBYFLOAT key increment|将 key 所储存的值增加指定的浮点数。|
|18|DECR key|将 key 中储存的数字值减 1。|
|19|DECRBY key decrement|将 key 所储存的值减去指定的整数。|
|20|APPEND key value|如果 key 已存在且为字符串，将指定 value 追加到原字符串末尾。|
## 5.3 哈希（Hash）
|序号|命令|解释|
|---|---|---|
|1|HDEL key field1 [field2 ...]|删除哈希表中一个或多个指定字段。|
|2|HEXISTS key field|查看哈希表 key 中指定的字段是否存在。|
|3|HGET key field|获取哈希表中指定字段的值。|
|4|HGETALL key|获取哈希表中所有字段及其对应的值。|
|5|HINCRBY key field increment|为哈希表中指定字段的整数值增加指定增量。|
|6|HINCRBYFLOAT key field increment|为哈希表中指定字段的浮点数值增加指定增量。|
|7|HKEYS key|获取哈希表中的所有字段。|
|8|HLEN key|获取哈希表中字段的数量。|
|9|HMGET key field1 [field2 ...]|获取哈希表中一个或多个指定字段的值。|
|10|HMSET key field1 value1 [field2 value2 ...]|同时设置哈希表中多个 field-value（域-值）对。|
|11|HSET key field value|将哈希表中指定字段的值设为 value。|
|12|HSETNX key field value|只有当指定字段不存在时，才设置该字段的值。|
|13|HVALS key|获取哈希表中所有字段对应的值。|
|14|HSCAN key cursor [MATCH pattern] [COUNT count]|迭代哈希表中的键值对。cursor 为游标，第一次从 0 开始；MATCH 指定匹配模式；COUNT 指定每次迭代希望返回的元素数量。|


## 5.4 列表（List）

|序号|命令|解释|
|---|---|---|
|1|LPUSH key value1 [value2 ...]|将一个或多个元素插入列表头部（左侧）。|
|2|RPUSH key value1 [value2 ...]|将一个或多个元素插入列表尾部（右侧）。|
|3|LPOP key|移除并返回列表的第一个元素。|
|4|RPOP key|移除并返回列表的最后一个元素。|
|5|LRANGE key start stop|获取列表指定范围内的元素。|
|6|LLEN key|获取列表长度。|
|7|LINDEX key index|通过索引获取列表中的元素。|
|8|LREM key count value|移除列表中指定数量的指定元素。|
|9|LSET key index value|通过索引设置列表中元素的值。|
|10|LTRIM key start stop|只保留列表指定范围内的元素，其余元素删除。|
|11|BLPOP key [key ...] timeout|移出并返回列表第一个元素；列表为空时阻塞等待。|
|12|BRPOP key [key ...] timeout|移出并返回列表最后一个元素；列表为空时阻塞等待。|



## 5.5 集合（Set）

|序号|命令|解释|
|---|---|---|
|1|SADD key member1 [member2 ...]|向集合添加一个或多个成员。|
|2|SMEMBERS key|获取集合中的所有成员。|
|3|SREM key member1 [member2 ...]|移除集合中一个或多个成员。|
|4|SISMEMBER key member|判断指定元素是否属于集合。|
|5|SCARD key|获取集合的成员数量。|
|6|SPOP key|随机移除并返回一个集合成员。|
|7|SRANDMEMBER key [count]|随机返回一个或多个集合成员，不删除成员。|
|8|SINTER key1 [key2 ...]|返回多个集合的交集。|
|9|SUNION key1 [key2 ...]|返回多个集合的并集。|
|10|SDIFF key1 [key2 ...]|返回多个集合的差集。|
|11|SSCAN key cursor [MATCH pattern] [COUNT count]|使用游标迭代集合中的元素。|



## 5.6 有序集合（Sorted Set）

|序号|命令|解释|
|---|---|---|
|1|ZADD key score member|向有序集合添加成员，并设置其分数；成员存在时更新分数。|
|2|ZRANGE key start stop [WITHSCORES]|按索引范围返回成员，默认按分数从低到高排列。|
|3|ZREVRANGE key start stop [WITHSCORES]|按索引范围返回成员，按分数从高到低排列。|
|4|ZSCORE key member|获取指定成员的分数。|
|5|ZRANK key member|获取指定成员的升序排名。|
|6|ZREVRANK key member|获取指定成员的降序排名。|
|7|ZREM key member [member ...]|移除有序集合中的一个或多个成员。|
|8|ZCARD key|获取有序集合的成员数量。|
|9|ZCOUNT key min max|统计指定分数范围内的成员数量。|
|10|ZINCRBY key increment member|为指定成员的分数增加指定增量。|
|11|ZRANGEBYSCORE key min max|按分数范围返回成员。|
|12|ZSCAN key cursor [MATCH pattern] [COUNT count]|使用游标迭代有序集合中的成员及其分数。|



## 5.7 HyperLogLog
- Redis 在 2.8.9 版本添加了 HyperLogLog 结构。用来做基数统计的算法，HyperLogLog 的优点是，在输入元素的数量或者体积非常非常大时，计算基数所需的空间总是固定 的、并且是很小的。
	- 对于数据集 {1, 3, 5, 7, 5, 7, 8}， 基数集为 {1, 3, 5 ,7, 8}, 基数(不重复元素)为5。 基数估计就是在误差可接受的范围内，快速计算基数。
- 但是，因为 HyperLogLog 只会根据输入元素来计算基数，而不会储存输入元素本身，所以 HyperLogLog 不能像集合那样，返回输入的各个元素。

|序号|命令|解释|
|---|---|---|
|1|PFADD key element [element ...]|向 HyperLogLog 中添加一个或多个元素。|
|2|PFCOUNT key [key ...]|返回一个或多个 HyperLogLog 的基数估算值（不重复元素数量）。|
|3|PFMERGE destkey sourcekey [sourcekey ...]|将多个 HyperLogLog 合并为一个。|



## 5.8 发布订阅（Pub/Sub）
- Redis 发布订阅 (pub/sub) 是一种消息通信模式：发送者 (pub) 发送消息，订阅者 (sub) 接收消息。Redis 客户端可以订阅任意数量的频道。

|序号|命令|解释|
|---|---|---|
|1|SUBSCRIBE channel [channel ...]|订阅一个或多个频道。|
|2|UNSUBSCRIBE channel [channel ...]|取消订阅指定频道。|
|3|PUBLISH channel message|向指定频道发布消息。|
|4|PSUBSCRIBE pattern [pattern ...]|订阅一个或多个符合指定模式的频道。|
|5|PUNSUBSCRIBE pattern [pattern ...]|取消订阅符合指定模式的频道。|
|6|PUBSUB subcommand [argument ...]|查看发布订阅系统的状态。|


## 5.9 事务（Transactions）
一次打包多个命令， 并且带有以下三个重要的保证：

- 批量操作在发送 EXEC 命令前被放入队列缓存。
- 收到 EXEC 命令后进入事务执行，事务中任意命令执行失败，其余的命令依然被执行。
- 在事务执行过程，其他客户端提交的命令请求不会插入到事务执行命令序列中。

一个事务从开始到执行会经历以下三个阶段：

- 开始事务。
- 命令入队。
- 执行事务。
Redis 事务不具备传统关系型数据库完整的 ACID 特性

|序号|命令|解释|
|---|---|---|
|1|MULTI|标记一个事务块的开始。|
|2|EXEC|执行事务块中的所有命令。|
|3|DISCARD|取消事务，放弃执行事务块中的所有命令。|
|4|WATCH key [key ...]|监视一个或多个 Key，如果事务执行前这些 Key 被修改，则事务执行失败。|
|5|UNWATCH|取消对所有 Key 的监视。|


## 5.10 脚本（Scripting）
- Redis 支持使用 Lua 编写脚本，然后提供了一些命令
- 你可以把 Lua 理解为：将一部分原本需要在 Python/Java 中执行的 Redis 操作逻辑，下沉到 Redis 服务器内部执行。

|序号|命令|解释|
|---|---|---|
|1|EVAL script numkeys key [key ...] arg [arg ...]|执行 Lua 脚本。|
|2|EVALSHA sha1 numkeys key [key ...] arg [arg ...]|通过脚本 SHA1 摘要执行已缓存的 Lua 脚本。|
|3|SCRIPT LOAD script|将 Lua 脚本加载到脚本缓存中，但不立即执行。|
|4|SCRIPT EXISTS sha1 [sha1 ...]|检查指定脚本是否存在于脚本缓存中。|
|5|SCRIPT FLUSH|清空脚本缓存。|
|6|SCRIPT KILL|终止当前正在运行的 Lua 脚本。|

## 5.11 连接（Connection）
- 用于连接 redis 服务

|序号|命令|解释|
|---|---|---|
|1|AUTH password|验证连接 Redis 所需的密码。|
|2|PING|检查 Redis 服务是否正常运行。|
|3|ECHO message|打印指定字符串。|
|4|SELECT index|切换到指定编号的数据库。|
|5|QUIT|关闭当前客户端连接。|


## 5.12 服务器（Server）
- 用于管理 redis 服务

|序号|命令|解释|
|---|---|---|
|1|INFO [section]|获取 Redis 服务器的信息和统计数据。|
|2|DBSIZE|返回当前数据库中 Key 的数量。|
|3|SAVE|同步保存当前数据库的数据到磁盘。|
|4|BGSAVE|在后台异步保存当前数据库的数据到磁盘。|
|5|BGREWRITEAOF|在后台异步执行 AOF 文件重写。|
|6|CONFIG GET parameter|获取指定配置参数的值。|
|7|CONFIG SET parameter value|修改 Redis 配置参数，无需重启。|
|8|CLIENT LIST|获取连接到 Redis 服务器的客户端列表。|
|9|FLUSHDB|删除当前数据库中的所有 Key。|
|10|FLUSHALL|删除 Redis 所有数据库中的所有 Key。|
|11|SLOWLOG subcommand [argument]|管理 Redis 慢查询日志。|
|12|SHUTDOWN [NOSAVE] [SAVE]|关闭 Redis 服务，可指定是否保存数据。|

注意

`FLUSHDB` 和 `FLUSHALL` 会删除数据，学习时谨慎执行。

## 5.13 GEO
- 存储地理位置信息，并对存储的信息进行操作，

|序号|命令|解释|
|---|---|---|
|1|GEOADD key longitude latitude member|将指定经纬度和地点名称添加到地理位置集合中。|
|2|GEOPOS key member [member ...]|获取一个或多个地点的经纬度坐标。|
|3|GEODIST key member1 member2 [unit]|计算两个地点之间的距离，可指定米、千米等单位。|
|4|GEORADIUS key longitude latitude radius unit|根据指定经纬度和半径，查找指定范围内的地点。|
|5|GEORADIUSBYMEMBER key member radius unit|以集合中已有的某个地点为中心，查找指定范围内的地点。|
|6|GEOHASH key member [member ...]|获取一个或多个地点的 GeoHash 编码。|


## 5.14 Stream

|序号|命令|解释|
|---|---|---|
|1|XADD key ID field value [field value ...]|向 Stream 添加一条消息，Stream 不存在时自动创建。|
|2|XREAD [COUNT count] [BLOCK milliseconds] STREAMS key ID|读取 Stream 中的消息，支持阻塞读取。|
|3|XRANGE key start end [COUNT count]|按消息 ID 范围获取消息。|
|4|XREVRANGE key end start [COUNT count]|按消息 ID 倒序获取消息。|
|5|XLEN key|获取 Stream 中的消息数量。|
|6|XDEL key ID [ID ...]|删除 Stream 中指定 ID 的消息。|
|7|XTRIM key MAXLEN count|修剪 Stream，限制消息数量。|
|8|XGROUP CREATE key groupname ID|创建消费者组。|
|9|XREADGROUP GROUP group consumer STREAMS key ID|通过消费者组读取消息。|
|10|XACK key group ID [ID ...]|确认消费者组中的消息已经处理完成。|
|11|XPENDING key group|查看消费者组中待确认的消息。|
|12|XCLAIM key group consumer min-idle-time ID|将待处理消息转移给指定消费者。|

## 各部分核心用途对照

|部分|核心用途|最重要的命令|
|---|---|---|
|List|双端插入、删除、队列|LPUSH、RPOP|
|Set|去重、交并差集|SADD、SINTER|
|ZSet|排序、排行榜|ZADD、ZRANGE|
|HyperLogLog|大规模去重计数|PFADD、PFCOUNT|
|Pub/Sub|消息发布与订阅|PUBLISH、SUBSCRIBE|
|Transactions|批量执行命令|MULTI、EXEC|
|Scripting|Lua 脚本执行|EVAL|
|Connection|客户端连接管理|PING、SELECT|
|Server|服务管理与监控|INFO、CONFIG|
|GEO|地理位置查询|GEOADD、GEODIST|
|Stream|消息队列|XADD、XREADGROUP、XACK|

https://zhuanlan.zhihu.com/p/419224864