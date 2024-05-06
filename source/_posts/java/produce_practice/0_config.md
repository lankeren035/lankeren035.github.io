---
title: 生产实习-0 开发环境准备

date: 2024-3-1 08:00:00

tags: [生产实习,java,springboot]

categories: [java]

comment: true

toc: true



---

#
<!--more-->

# 0. 开发环境准备

## 0.1 安装java

### 0.1.1 下载安装

- 下载地址： [Archived OpenJDK GA Releases (java.net)](https://jdk.java.net/archive/) 

  ![](D:/blog/themes/yilia/source/img/java/produce_practice/0/1.png)

  ![](img/java/produce_practice/0/1.png)

- 下载后解压到你想放到的目录（比如我是D:/jdk-17

### 0.1.2 配置环境变量

1. windows搜索：环境变量。选择`编辑系统环境变量`

2. 新建环境变量： ![](D:/blog/themes/yilia/source/img/java/produce_practice/0/2.png)

   ![](img/java/produce_practice/0/2.png) 

3. 输入变量名：`JAVA_HOME`，变量值（根据你存放的路径）：`D:\jdk-17`，确定。

4. 双击path，选择新建，输入`%JAVA_HOME%\bin`，确定，这样就将java的bin目录放到系统环境变量了，可以直接使用java命令了。（javac命令？？？）

5. 重新启动终端（更新环境变量）输入java回车，可以看到有反应，说明ok了。



## 0.2 安装docker

| 方案                    | 要求               |
| ----------------------- | ------------------ |
| 本地下载mysql           | 本地没下载过数据库 |
| 虚拟机 + mysql          |                    |
| 虚拟机 + docker + mysql |                    |
| 服务器 + docker + mysql |                    |

- 本次实验采用服务器 + docker + mysql

### 0.2.1 下载ubuntu镜像（使用虚拟机）

-  [Get Ubuntu Server | Download | Ubuntu](https://ubuntu.com/download/server) 

    ![](D:/blog/themes/yilia/source/img/java/produce_practice/0/4.png)

    ![](img/java/produce_practice/0/4.png) 

### 0.2.2 安装虚拟机

- 略

### 0.2.3 安装docker

- 服务器中输入

    ```bash
    sudo apt install docker.io
    ```

    如果命令无效，提示用snap安装，则用snap

    ，输入y是确认。安装完成收输入whereis docker可以查看docker位置
    
- 服务器中输入

    ```bash
    - sudo docker search mysql
    ```

### 0.2.4 安装mysql

- 服务器中输入：

  ```bash
  sudo docker pull mysql:8.0.34
  ```

- 查看镜像文件：

  ```bash
  sudo docker images
  ```

  可以看到镜像文件中有一个我们刚刚拉取的镜像

- 将镜像封装成容器运行：(这里将docker的3306端口映射到本地的3306，你可以改，数据库密码123456)

  ```bash
  sudo docker run -itd --name mysql8034 -p 3306:3306 -e MYSQL_ROOT_PASSWORD=123456 mysql:8.0.34
  ```

- 查看容器列表

  ```bash
  sudo docker ps -a
  ```

  看到刚刚的容器就说明ok了。

- 可视化工具：navicat、Mysql workbench（使用较低版本的navicat连接的时候可能会发现连接错误，需要更新navicat版本）

## 0.3 安装idea

- 下载地址（划到下面下载社区版）：https://www.jetbrains.com/zh-cn/idea/download

- 配置国内镜像：

  - 设置内搜索：maven

    ![](D:/blog/themes/yilia/source/img/java/produce_practice/0/5.png)

    ![](img/java/produce_practice/0/5.png) 

  - 创建C:\Users\123\.m2\settings.xml

    ```xml
    <settings xmlns="http://maven.apache.org/SETTINGS/1.0.0"
              xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
              xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0
                                  http://maven.apache.org/xsd/settings-1.0.0.xsd">
        <mirrors>
            <mirror>
                <id>alimaven</id>
                <name>aliyun maven</name>
                <url>http://maven.aliyun.com/nexus/content/groups/public/</url>
                <mirrorOf>central</mirrorOf>
            </mirror>
        </mirrors>
    </settings>
    ```

  - 重新打开设置，将刚刚的勾勾上

    ![](D:/blog/themes/yilia/source/img/java/produce_practice/0/5.png)

    ![](img/java/produce_practice/0/5.png) 

- 检验国内镜像配置

  - 进入网址（建议收藏）： [Maven Repository: Search/Browse/Explore (mvnrepository.com)](https://mvnrepository.com/) 

  - 搜索fastjson2：

    ![](D:/blog/themes/yilia/source/img/java/produce_practice/0/6.png)

    ![](img/java/produce_practice/0/6.png) 

    ![](D:/blog/themes/yilia/source/img/java/produce_practice/0/7.png)

    ![](img/java/produce_practice/0/7.png) 

    ![](D:/blog/themes/yilia/source/img/java/produce_practice/0/8.png)

    ![](img/java/produce_practice/0/8.png) 

    - 复制：

    ```xml
    <!-- https://mvnrepository.com/artifact/com.alibaba.fastjson2/fastjson2 -->
    <dependency>
        <groupId>com.alibaba.fastjson2</groupId>
        <artifactId>fastjson2</artifactId>
        <version>2.0.47</version>
    </dependency>
    
    ```

    - 将上面的内容粘贴到项目的pom.xml的<dependences>标签下（无则自己建），更新maven：

    ![](D:/blog/themes/yilia/source/img/java/produce_practice/0/9.png)

    ![](img/java/produce_practice/0/9.png) 

    - 此时刚刚粘贴的依赖不是红色字体了，说明下载ok，然后输入json可以看到有代码自动补全：

      ![](D:/blog/themes/yilia/source/img/java/produce_practice/0/10.png)

      ![](img/java/produce_practice/0/10.png) 

- 统一编码：（搜索encoding或编码，还有搜索控制台：编码）

  ![](D:/blog/themes/yilia/source/img/java/produce_practice/0/11.png)

  ![](img/java/produce_practice/0/11.png) 

- 避免其他插件在控制台输出中文乱码：

  ![](D:/blog/themes/yilia/source/img/java/produce_practice/0/12.png)

  ![](img/java/produce_practice/0/12.png) 

  输入：

  ```
  -Xmx4096m
  -Dfile.encoding=UTF-8
  ```