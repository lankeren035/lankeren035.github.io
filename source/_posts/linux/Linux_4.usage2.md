---
title: Linux入门-3使用操作2
date: 2024-03-03 14:00:00
toc: true
tags: [Linux]
categories: [Linux]

---

#

<!--more-->



# 3 Linux实用操作

## 3.6 ip与主机

- 查看ip

```bash
ifconfig #查看ip
yum -y install net-tools #如果无法使用ifconfig
```

- 查看主机名

```bash
hostname #查看主机名
hostnamectl set-hostname 主机名 #修改主机名
```

通过修改`     C:\Windows\System32\drivers\etc\hosts  `（linux在`/etc/hosts`）中的配置就可以实现通过主机名访问-

- 在VMware中设置固定ip

  1. 在vmware中配置ip地址网关和网段

     ![](D:\blog\themes\yilia\source\img\linux\usage\1.png)

     ![](img/linux/usage/1.png)

     ![](D:\blog\themes\yilia\source\img\linux\usage\2.png)

     ![](img/linux/usage/2.png)

     ![](D:\blog\themes\yilia\source\img\linux\usage\3.png)

     ![](img/linux/usage/3.png)

     ![](D:\blog\themes\yilia\source\img\linux\usage\4.png)

     ![](img/linux/usage/4.png)

  2. 在linux中手动修改配置文件

     ```bash
     vim /etc/sysconfig/network-scripts/ifcfg-ens33
     ```

     做如下修改：

     ![](D:\blog\themes\yilia\source\img\linux\usage\5.png)

     ![](img/linux/usage/5.png)

     ```bash
     systemctl restart network #重启网卡
     ```

## 3.7 网络传输

  - ping命令

    ```bash
    ping [-c 数字] ip或主机名 #检查指定服务器是否可联通
    ```

    - -c：检查次数

  - wget命令

    ```bash
    wget [-b] url #文件下载
    tail -f wget-log #监控后台下载进度
    ```

    - -b：后台下载，会将日志写入当前工作目录的wget-log文件

    - 如果下载未完成，请及时清理未完成的不可用文件

  - curl命令

    发送http网络请求，可用于下载文件、获取信息等

    ```bash
    curl [-O] url
    ```

    - -O：下载文件

  - 端口

    | 端口        | 作用                              |
    | ----------- | --------------------------------- |
    | 1~1023      | 公认端口                          |
    | 1024~49151  | 注册端口，松散的绑定一些程序\服务 |
    | 49152~65535 | 不绑定固定程序，临时使用          |
    |             |                                   |

    - 查看端口占用

      - nmap命令

        ```bash
        yum -y install nmap #安装nmap
        nmap ip #查看端口占用
        ```

      - netstat命令

        ```bash
        yum -y install net-tools #安装netstat
        netstat -anp | grep 端口号 #查看端口被谁占用
        ```

## 3.8 进程管理

- 查看进程

  ```bash
  ps [-e -f] #查看进程信息
  ```

  - 例子

    ```bash
    ps -ef | grep tail #查找tail命令的进程信息
    ```

    

  - -e：全部进程

  - -f：以完全格式化的形式展示信息

    | 列名  | 解释                                          |
    | ----- | --------------------------------------------- |
    | UID   | 用户ID                                        |
    | PID   | 进程号                                        |
    | PPID  | 父进程                                        |
    | C     | cpu占用率                                     |
    | STIME | 启动时间                                      |
    | TTY   | 启动此进程的终端序号，如显示?，表示非终端启动 |
    | TIME  | 占用cup时间                                   |
    | CMD   | 进程名<br>启动路径<br>启动命令                |

  - 关闭进程

    ```bash
    kill [-9] 进程ID #关闭进程
    ```

    - -9：强制关闭，不询问进程

  

## 3.9 主机状态

- 查看系统资源占用

  ```bash
  top [-p -d -c -n -b -i -u] #查看系统资源占用
  ```

  ![](D:\blog\themes\yilia\source\img\linux\usage\6.png)

  ![](img/linux/usage/6.png)

  | 行   | 解释                                                         |
  | ---- | ------------------------------------------------------------ |
  | 1    | 时间<br>up：启动了6分钟<br>users：2个用户登录<br>load：1、5、15分钟负载 |
  | 2    | tasks：175个进程<br>running：1个进程在运行<br>sleeping：174个在睡眠<brL>0个停止进行，0个僵尸进程 |
  | 3    | cpu：cpu使用率<br>us：用户cpu使用率<br>sy：系统cpu使用率<br>ni：高优先级进程占用CPU时间百分比<br>id：空闲CPU率<br>wa：IO等待CPU占用率<br>hi：CPU硬件中断率<br>si：CPU软件中断率<br>st：强制等待占用CPU率 |
  | 4    | Kib Mem：物理内存<br>total：总量<br>free：空闲<br/>used：使用<br/>buff/cache：buff和cache占用 |
  | 5    | KibSwap：虚拟内存（交换空间）<br/>total：总量<br/>free：空闲<br/>used：使用<br/>buff/cache：buff和cache占用 |
  | 6    | PR：进程优先级，越小越高<br>NI：负值表示高优先级，正表示低优先级<br>VIRT：进程使用虚拟内存，单位KB<br>RES：进程使用物理内存，单位KB<br/>SHR：进程使用共享内存，单位KB<br/>S：进程状态（S休眠，R运行，Z僵死状态，N负数优先级，I空闲状态）<br/>%MEM：进程占用内存率<br/>TIME+：进程使用CPU时间总计，单位10毫秒<br/>COMMAND：进程的命令或名称或程序文件路径 |

  ![](D:\blog\themes\yilia\source\img\linux\usage\7.png)

  ![](img/linux/usage/7.png)

  - 在top界面下：

    ![](D:\blog\themes\yilia\source\img\linux\usage\8.png)

    ![](img/linux/usage/8.png)

- 磁盘信息监控

  - df命令查看磁盘使用

    ```bash
    df [-h]
    ```

    - -h显示单位

  - iostat查看cpu、磁盘信息

    ```bash
    iostat [-x][数字1][数字2]
    ```

    - -x：显示更多信息

    - 数字1：刷新间隔

    - 数字2：刷新次数

      ![](D:\blog\themes\yilia\source\img\linux\usage\9.png)

      ![](img/linux/usage/9.png)

      - rrqm/s： 每秒这个设备相关的读取请求有多少被Merge了（当系统调用需要读取数据的时候，VFS将请求发到各个FS，如果FS发现不同的读取请求读取的是相同Block的数据，FS会将这个请求合并Merge, 提高IO利用率, 避免重复调用）；

      - wrqm/s： 每秒这个设备相关的写入请求有多少被Merge了。

      - rsec/s： 每秒读取的扇区数；sectors

      - wsec/： 每秒写入的扇区数。

      - rKB/s： 每秒发送到设备的读取请求数

      - wKB/s： 每秒发送到设备的写入请求数

      - avgrq-sz  平均请求扇区的大小

      - avgqu-sz  平均请求队列的长度。毫无疑问，队列长度越短越好。  

      - await：  每一个IO请求的处理的平均时间（单位是微秒毫秒）。

      - svctm   表示平均每次设备I/O操作的服务时间（以毫秒为单位）

      - %util：  磁盘利用率

- 网络状态监控

  ```bash
  sar -n DEV 数字1 数字2
  ```

  - -n：查看网络

  - DEV：查看网络接口

  - 数字1：刷新间隔

  - 数字2：查看次数

    ![](D:\blog\themes\yilia\source\img\linux\usage\10.png)

    ![](img/linux/usage/10.png)

    - IFACE 本地网卡接口的名称

    - rxpck/s 每秒钟接受的数据包

    - txpck/s 每秒钟发送的数据包

    - rxKB/S 每秒钟接受的数据包大小，单位为KB

    - txKB/S 每秒钟发送的数据包大小，单位为KB

    - rxcmp/s 每秒钟接受的压缩数据包

    - txcmp/s 每秒钟发送的压缩包

    - rxmcst/s 每秒钟接收的多播数据包