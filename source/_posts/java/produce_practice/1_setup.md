---
title: 生产实习-1 系统结构搭建

date: 2024-3-2 08:00:00

tags: [生产实习,java,springboot]

categories: [java]

comment: true

toc: true


---

#
<!--more-->

# 1. 系统结构搭建

## 1.1 下载项目

1. 查看项目规范文档中的工具版本：
   - 使用spring boot3.1.0

![](D:/blog/themes/yilia/source/img/java/produce_practice/1.png)

![](img/java/produce_practice/1.png)

2. [进入网址](https://start.spring.io)

   - 按文档选择，其中spring boot没有这个版本，后面再改。

     ![](D:/blog/themes/yilia/source/img/java/produce_practice/3.png)

     ![](img/java/produce_practice/3.png)

     点击`生成`,网站就会帮我们下载一个工程源文件。将该文件夹解压后就是你的项目文件夹。

3. 打开里面的pom.xml文件，将里面的spring boot版本改成文档需要的3.1.0版本

   ![](D:/blog/themes/yilia/source/img/java/produce_practice/4.png)

   ![](img/java/produce_practice/4.png)

4. 这些基础依赖里只有mybatis明确指定了版本号。我们调低了框架的主版本号，为了防止不兼容，我们需要调整mybatis版本号。

   ![](D:/blog/themes/yilia/source/img/java/produce_practice/5.png)

   ![](img/java/produce_practice/5.png)

5. 访问之前的[第三方依赖仓库](https://mvnrepository.com)查看主框架的依赖版本号。

   搜索spring boot的依赖版本

   - 搜索：

   ![](D:/blog/themes/yilia/source/img/java/produce_practice/6.png)

   ![](img/java/produce_practice/6.png)

   - 点击3.1.0版本：

   ![](D:/blog/themes/yilia/source/img/java/produce_practice/7.png)

   ![](img/java/produce_practice/7.png)

   - 发现依赖的spring web版本为6.0.9：

   ![](D:/blog/themes/yilia/source/img/java/produce_practice/8.png)

   ![](img/java/produce_practice/8.png)

   

   搜索mybatis starter查询他的依赖版本

    ![](D:/blog/themes/yilia/source/img/java/produce_practice/9.png)

    ![](img/java/produce_practice/9.png)

   - 搜索

      ![](D:/blog/themes/yilia/source/img/java/produce_practice/10.png)

      ![](img/java/produce_practice/10.png)

   - 点击上面看到的版本：3.0.3

      ![](D:/blog/themes/yilia/source/img/java/produce_practice/11.png)

      ![](img/java/produce_practice/11.png)

   - 点击对应版本：

      ![](D:/blog/themes/yilia/source/img/java/produce_practice/12.png)

      ![](img/java/produce_practice/12.png)

     发现对应的spring context版本是6.1.0，与之前主框架的6.0.9不一致。

      ![](D:/blog/themes/yilia/source/img/java/produce_practice/13.png)

      ![](img/java/produce_practice/13.png)

   - 需要将mybatis starter版本降低，经查询3.0.2版本依赖的spring context也是6.0.9，因此可以将pom.xml中对应版本改为3.0.2

## 1.2 运行项目

- 用idea打开刚刚下载的项目会自动下载依赖。可以看到下面有很多依赖项：

 ![](D:/blog/themes/yilia/source/img/java/produce_practice/14.png)

![](img/java/produce_practice/14.png)

- 现在暂时不连接数据库，需要注释掉mybatis和mysql连接器，否则会出错。

![](D:/blog/themes/yilia/source/img/java/produce_practice/15.png)

![](img/java/produce_practice/15.png)

- 刷新maven后依赖更新：

![](D:/blog/themes/yilia/source/img/java/produce_practice/16.png)

![](img/java/produce_practice/16.png)

- 找到主程序，运行，记住程序端口：

![](D:/blog/themes/yilia/source/img/java/produce_practice/17.png)

![](img/java/produce_practice/17.png)

- 在static下新建一个index.html测试一下：

![](D:/blog/themes/yilia/source/img/java/produce_practice/18.png)

![](img/java/produce_practice/18.png)

- 重新运行，输入网址，测试成功：

 ![](D:/blog/themes/yilia/source/img/java/produce_practice/19.png)

 ![](img/java/produce_practice/19.png)  

## 1.3 连接数据库

- 使用阿里云服务器的mysql，第二天发现被攻击了，数据库丢失，所以要做好备份，或者即使关闭端口。可以参考[数据库恢复部分](##1.4)不一定有用。

- 解除之前的注释，刷新maven

   ![](D:/blog/themes/yilia/source/img/java/produce_practice/21.png)

   ![](img/java/produce_practice/21.png)

- 将resources/application.properties重命名：application.yml，配置数据库信息：

   ![](D:/blog/themes/yilia/source/img/java/produce_practice/20.png)

   ![](img/java/produce_practice/20.png)

- 注意：如果你将数据库配置那一段删掉，那么运行的时候会出错，但是就算你数据库配置写错了，运行也不会报错，你需要通过一个测试单元来尝试连接一下数据库才能发现是否能够连接成功。

- 在数据库中创建test表，test1列，写入数据：`hello`，在src/test/java/下面有个测试文件，写入如下代码后运行：

  ```java
  package com.lyingedu.questionnaire;
  
  import org.junit.jupiter.api.Test;
  import org.springframework.beans.factory.annotation.Autowired;
  import org.springframework.boot.test.context.SpringBootTest;
  import org.springframework.jdbc.core.JdbcTemplate;
  import org.springframework.jdbc.core.RowMapper;
  
  import java.sql.ResultSet;
  import java.sql.SQLException;
  
  @SpringBootTest
  class QuestionnaireApplicationTests {
  
  	@Autowired
  	private JdbcTemplate jdbcTemplate;
  
  	@Test
  	void contextLoads() {
  		// 定义一个RowMapper来处理查询结果
  		RowMapper<String> rowMapper = new RowMapper<String>() {
  			@Override
  			public String mapRow(ResultSet rs, int rowNum) throws SQLException {
  				// 获取test1列的值
  				return rs.getString("test1");
  			}
  		};
  
  		// 执行查询并获取结果
  		String result = jdbcTemplate.queryForObject("SELECT test1 FROM test LIMIT 1", rowMapper);
  
  		// 打印结果
  		System.out.println("查询结果: " + result);
  	}
  }
  
  ```

  如果能够输出结果，说明连接成功。

   ![](D:/blog/themes/yilia/source/img/java/produce_practice/23.png)

   ![](img/java/produce_practice/23.png)

  

## 1.4 数据库修复

- 如果你的远程数据库被攻击了或者被删了，你可以参考以下操作（不一定有用）

  - 查找mysql的binlog文件：连接数据库后执行sql语句

    ```sql
    SHOW BINARY LOGS;
    ```

    ![](D:/blog/themes/yilia/source/img/java/produce_practice/35.png)

    ![](img/java/produce_practice/35.png)

    发现有多个日志文件

  - 使用sql语句查看日志信息

    ```sql
    show binlog events in 'binlog.000002';
    ```

  - 根据日志文件名查找所在目录：在服务器中运行（root用户）

    ```bash
    find / -name "binlog.000001"
    ```

    然后cd打开目录

  - 将日志文件变成sql文件：

    ```bash
    mysqlbinlog binlog.000001 > binlog1.sql
    ```

    如果提示没有mysqlbinlog，你需要根据提示下载：

    ```
    apt install mysql-server-core-8.0
    ```

    显示的带*的就是已经有的功能，选择那个前面是空的带log的功能

  - 将转换好的sql文件复制到docker中

    ```bash
    docker cp binlog1.sql 容器名:/路径
    ```

    容器名通过以下命令查看

    ```bash
    docker ps -a
    ```

  - 进入docker：

    ```
    docker exec -it 容器名 bash
    mysql -u 数据库用户名 -p < sql文件路径
    ```

  - [参考](https://developer.aliyun.com/article/515402)

    