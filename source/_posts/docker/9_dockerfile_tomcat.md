---
title: 9- Dockerfile制作tomcat镜像

date: 2024-8-28 13:00:00

tags: [docker]

categories: [docker]

comment: true

toc: true





---

#

<!--more-->

# 9- Dockerfile制作tomcat镜像



## 1. 准备tomcat压缩包，jdk压缩包

```shell
wget https://download.java.net/java/GA/jdk22.0.2/c9ecb94cd31b495da20a27d4581645e8/9/GPL/openjdk-22.0.2_linux-x64_bin.tar.gz
wget https://dlcdn.apache.org/tomcat/tomcat-9/v9.0.93/bin/apache-tomcat-9.0.93.tar.gz
```



## 2. 编写dockerfile

```shell
touch Dockerfile
vim Dockerfile
```

- 写入：

```dockerfile
FROM centos:7
MAINTAINER admin<123456@qq.com>

# 将当前目录的readme.txt复制到镜像中
COPY readme.txt /usr/local/readme.txt

# 使用ADD会自动解压
ADD openjdk-22.0.2_linux-x64_bin.tar.gz /usr/local/
ADD apache-tomcat-9.0.93.tar.gz /usr/local

# 禁用默认的 CentOS 源，并替换为阿里云源
RUN sed -i 's|^mirrorlist=|#mirrorlist=|g' /etc/yum.repos.d/CentOS-Base.repo && \
    sed -i 's|^#baseurl=http://mirror.centos.org|baseurl=http://mirrors.aliyun.com|g' /etc/yum.repos.d/CentOS-Base.repo && \
    yum clean all && \
    yum makecache fast
RUN yum -y install vim

# 配置jdk，apache等环境变量
ENV MYPATH /usr/local
WORKDIR $MYPATH

#如果你用的jdk版本较新，需要注意classpath环境变量
ENV JAVA_HOME /usr/local/jdk-22.0.2
ENV CLASSPATH $JAVA_HOME/lib
ENV CATALINA_HOME /usr/local/apache-tomcat-9.0.93
ENV CATALINA_BASH /usr/local/apache-tomcat-9.0.93
ENV PATH $PATH:$JAVA_HOME/bin:$CATALINA_HOME/lib:$CATALINA_HOME/bin

EXPOSE 8080

# tail 实时追踪文件的变化
# 需要注意路径问题，是logs不是bin/logs
CMD /usr/local/apache-tomcat-9.0.93/bin/startup.sh && tail -F /usr/local/apache-tomcat-9.0.93/logs/catalina.out
```

- 将其命名为`Dockerfile`的好处就是构建的时候会自动寻找这个文件，不用使用`-f`选项了

  ```shell
  mkdir -p tomcat_volumes/test tomcat_volumes/logs
  docker build -t mytomcat .
  docker run -d -p 3355:8080 --name mytomcat01 -v /home/ke/test/tomcat_volumes/test:/usr/local/apache-tomcat-9.0.93/webapps/test -v /home/ke/test/tomcat_volumes/logs:/usr/local/apache-tomcat-9.0.93/logs mytomcat
  curl localhost:3355
  ```

- 浏览器输入：`ip:3355`即可查看tomcat主页。

- 创建主页：

  ```shell
  vim tomcat_volumes/test/index.jsp
  ```

  ```jsp
  <%@ page language="java" contentType="text/html; charset=UTF-8"
      pageEncoding="UTF-8"%>
  <!DOCTYPE html>
  <html>
      <head>
          <meta charset="utf-8">
          <title>hello world</title>
      </head>
      <body>
          Hello World!<br/>
          <%
          	System.out.println("-------my test web logs------");
          %>
      </body>
  </html>
  ```

- 访问：`ip:3355/test`