---
title: 生产实习-2 引入静态资源

date: 2024-3-4 08:00:00

tags: [生产实习,java,springboot]

categories: [java]

comment: true

toc: true



---

#
<!--more-->

# 2. 引入静态资源

## 2.1 引入静态资源

1. 将静态文件粘贴到static目录

   ![](D:/blog/themes/yilia/source/img/java/produce_practice/2/1.png)

   ![](img/java/produce_practice/2/1.png)

2. 查看程序访问路径：8085端口

   ![](D:/blog/themes/yilia/source/img/java/produce_practice/2/2.png)

   ![](img/java/produce_practice/2/2.png)

3. 修改程序配置文件的端口配置为8085

   ![](D:/blog/themes/yilia/source/img/java/produce_practice/2/3.png)

   ![](img/java/produce_practice/2/3.png)

4. 运行并访问 ：http://127.0.0.1:8085/pages/login/index.html，能显示登录页面就ok

   

## 2.2 连接数据库

1. 创建数据库myweb并创建两个表并插入一条数据

   ```sql
   CREATE TABLE IF NOT EXISTS `myweb`.`user_info` (
     `id` VARCHAR(40) NOT NULL,
     `username` VARCHAR(255) NULL,
     `password` VARCHAR(255) NULL,
     `start_time` DATETIME NULL,
     `stop_time` DATETIME NULL,
     `status` VARCHAR(255) NULL,
     `created_by` VARCHAR(255) NULL,
     `creation_date` DATETIME NULL,
     `last_update_by` VARCHAR(255) NULL,
     `last_update_date` DATETIME NULL,
     PRIMARY KEY (`id`))
   ENGINE = InnoDB
   
   INSERT INTO user_info (id, username,password)
   VALUES ('0', 'admin', '1');
   ```
   
2. 更改配置文件中数据库名：

    ![](D:/blog/themes/yilia/source/img/java/produce_practice/2/4.png)

   ![](img/java/produce_practice/2/4.png) 

3. 所有工程文件都在main/java/com/lyingedu.questionnaire下。在这里创建如下目录结构：

   ![](D:/blog/themes/yilia/source/img/java/produce_practice/2/5.png)

   ![](img/java/produce_practice/2/5.png)

4. 使用dbgen工具将数据库导入工程：

  - 用idea打开dbgen工具
  
  - 对myweb.xml做如下修改：
  
    ![](D:/blog/themes/yilia/source/img/java/produce_practice/2/6.png)
  
    ![](img/java/produce_practice/2/6.png)
  
  - 修改配置文件，运行（注意数据库地址端口用户名密码都不能错）
  
    ![](D:/blog/themes/yilia/source/img/java/produce_practice/2/7.png)
  
    ![](img/java/produce_practice/2/7.png)
  
  - 运行完成后会发现在主项目的questionnaire下多出了一个dbmap文件夹：
  
    ![](D:/blog/themes/yilia/source/img/java/produce_practice/2/8.png)
  
    ![](img/java/produce_practice/2/8.png)

5. 数据库实体用户在生成后需要对这些对实体映射进行一些配置（不然无法在开发过程中引用）

   - 指定数据库配置文件的位置：（注意不要把路径中的com/lyingedu写成了com.lyingedu ！

     ![](D:/blog/themes/yilia/source/img/java/produce_practice/2/9.png)

     ![](img/java/produce_practice/2/9.png)

   - 在主入口映射接口文件，运行成功则配置ok

      ![](D:/blog/themes/yilia/source/img/java/produce_practice/2/10.png)

      ![](img/java/produce_practice/2/10.png)  