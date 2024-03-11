---
title: 生产实习-6 单元测试与log日志输出

date: 2024-3-5 12:00:00

tags: [生产实习,java,springboot]

categories: [java]

comment: true

toc: true



---

#
<!--more-->



# 6. 单元测试与log日志输出



## 6.1 配置日志文件

- 将logback-spring.xml放到resources下，做如下修改：

  ![](D:/blog/themes/yilia/source/img/java/produce_practice/5/2.png)

  ![](img/java/produce_practice/5/2.png)

- 假设你需要在每次登录的时候都记录日志，那在controller中找到登录方法所在的类，然后在类上面加上@Slf4j的注释，可以发现结构中多了一个log：

    ![](D:/blog/themes/yilia/source/img/java/produce_practice/5/3.png)

    ![](img/java/produce_practice/5/3.png)

- 在登录方法中加入日志记录：

  ```java
      public HttpResponseEntity userLogin(@RequestBody UserInfo userInfo) { //requestbody注解，将请求的json数据转换为对象
  
          log.info("用户" + userInfo.getUsername() + "登录");
  ```

  然后运行项目，登录后可以看到控制台有输出登录信息，并且log目录下也会有日志记录。

## 6.2 单元测试

- 在src/test下可以找到测试类

- 在pom.xml的dependences标签下加入依赖：（记得刷新maven）这样就可以使用@RunWith注解，它可以使用springboot的依赖注入

  ```xml
  		<dependency>
  			<groupId>junit</groupId>
  			<artifactId>junit</artifactId>
  			<version>4.12</version>
  		</dependency>
  ```

- 测试代码：

  ```java
  package com.lyingedu.questionnaire;
  
  import com.lyingedu.questionnaire.beans.HttpResponseEntity;
  import com.lyingedu.questionnaire.biz.user.controller.UserController;
  import com.lyingedu.questionnaire.common.utils.UUIDUtil;
  import com.lyingedu.questionnaire.dbmap.entities.UserInfo;
  import jakarta.annotation.Resource;
  import lombok.extern.slf4j.Slf4j;
  import org.junit.Test;
  import org.junit.runner.RunWith;
  import org.springframework.boot.test.context.SpringBootTest;
  import org.springframework.test.context.junit4.SpringRunner;
  
  import java.util.Date;
  import java.util.List;
  
  @Slf4j
  @RunWith(SpringRunner.class) //这样可以用springboot的依赖注入
  @SpringBootTest
  public class QuestionnaireApplicationTests {
      @Resource
      private UserController userController;
  
      /*
      * 测试查询功能
       */
      @Test //注意使用junit，而不是junit.api
      public void testQueryUserInfoList(){
          UserInfo userInfo = new UserInfo();
          HttpResponseEntity httpResponseEntity = userController.queryUserList(userInfo);
          if ("666".equals(httpResponseEntity.getCode())) {
              log.info(">>queryUserList用户列表查询测试成功");
          }
      }
  
      /*
      * 测试用户登录
       */
      @Test
      public void testUserLogin(){
          UserInfo userInfo = new UserInfo();
          userInfo.setUsername("admin");
          userInfo.setPassword("12");
          HttpResponseEntity httpResponseEntity = userController.queryUserList(userInfo);
          if ("666".equals(httpResponseEntity.getCode())) {
              log.info(">>selectUserInfo用户"+userInfo.getUsername()+"登录测试成功");
          }else {
              log.info(">>selectUserInfo用户登录测试失败");
          }
      }
  
      /*
      * 测试创建用户
       */
      @Test
      public void testCreateUser(){
          UserInfo userInfo = new UserInfo();
          userInfo.setId(UUIDUtil.getOneUUID());
          userInfo.setUsername("test005");
          userInfo.setPassword("123");
          userInfo.setStatus("1");
          userInfo.setCreatedBy("anyone");
          userInfo.setCreationDate(new Date());
          userInfo.setStartTime(new Date());
          userInfo.setStopTime(new Date());
          userInfo.setLastUpdateBy("anyone");
          userInfo.setLastUpdateDate(new Date());
  
          HttpResponseEntity httpResponseEntity = userController.addUserInfo(userInfo);
          if("666".equals(httpResponseEntity.getCode())){
              log.info(">>adduserinfo插入用户测试成功");
          }
  
      }
  
      /*
      * 测试修改用户
       */
      @Test
      public void testUpdUser(){
          UserInfo userInfo = new UserInfo();
          userInfo.setUsername("test005");
          userInfo.setPassword("123");
  
          HttpResponseEntity httpResponseEntity = userController.queryUserList(userInfo);
  
          if("666".equals(httpResponseEntity.getCode())){//查询成功
              List list=(List) httpResponseEntity.getData();
              if(list.size() !=0){
                  UserInfo data = (UserInfo) list.get(0);
                  data.setUsername("test006");
                  HttpResponseEntity httpResponseEntity1 = userController.modifyUserInfo(data);
                  if("666".equals(httpResponseEntity1.getCode())){
                      log.info(">>update修改用户更新测试成功");
                  }
              }
  
          }
      }
  
      /*
      * 测试删除用户
       */
      @Test
      public void testDeleteUser(){
          UserInfo userInfo = new UserInfo();
          userInfo.setUsername("test006");
          userInfo.setPassword("123");
  
          HttpResponseEntity httpResponseEntity = userController.queryUserList(userInfo);
  
          if("666".equals(httpResponseEntity.getCode())){//查询成功
              List list=(List) httpResponseEntity.getData();
              if(list.size() !=0){
                  UserInfo data = (UserInfo) list.get(0);
  
                  HttpResponseEntity httpResponseEntity1 = userController.deleteUserById(data);
                  if("666".equals(httpResponseEntity1.getCode())){
                      log.info(">>delete用户删除测试成功");
                  }
              }
  
          }
      }
  }
  
  ```

  