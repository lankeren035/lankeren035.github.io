---
title: 生产实习-3 用户模块基础代码

date: 2024-3-4 11:00:00

tags: [生产实习,java,springboot]

categories: [java]

comment: true

toc: true




---

#
<!--more-->

# 3. 用户模块基础代码

- spring boot运行总体流程：前端发起请求 -> 后端controller -> 后端service -> mapper -> sql
- 我们从后往前写，先写service

## 3.1 实现service

- service层是一个接口，实现类的形式。

  ![](D:/blog/themes/yilia/source/img/java/produce_practice/3/1.png)

  ![](img/java/produce_practice/3/1.png)

  1. 首先在biz/user/service下创建UserService。然后根据文档中表格信息实现接口：

     ```java
     package com.lyingedu.questionnaire.biz.user.service;
     
     import com.lyingedu.questionnaire.dbmap.entities.UserInfo;
     
     import java.util.List;
     
     public interface UserService {
     
         /**
          * 增加用户
          * @param userInfo
          * @return
          */
         int addUserInfo(UserInfo userInfo);
     
         /**
          * 修改用户
          * @param userInfo
          * @return
          */
         int modifyUserInfo(UserInfo userInfo);
     
     
         /**
          * 删除用户
          * @param id
          * @return
          */
         int deleteUserById(String id);
     
         /**
          * 查询用户列表
          * @return
          */
         List<UserInfo> queryUserList(UserInfo userInfo);
     }
     
     ```

  2. 在biz/user/service下创建UserServiceImpl。编写service实现类

     - 先写public class UserServiceImpl implements UserService然后鼠标放到下划线，点击实现）
     - 然后在类上面写注释：@Service让springboot知道这是一个操作数据库的service
- 写入UserInfoMapper属性，属性需要在实现类里面说明。@resource注解可以让springboot自动给他注入依赖，否则UserInfoMapper找不到
  
     ```java
     package com.lyingedu.questionnaire.biz.user.service;
     
     import com.lyingedu.questionnaire.dbmap.entities.UserInfo;
     import com.lyingedu.questionnaire.dbmap.imapper.UserInfoMapper;
     import jakarta.annotation.Resource;
     import org.springframework.stereotype.Service;
     
     import java.util.List;
     
     @Service //让spring boot知道这是操作数据库的service
     public class UserServiceImpl implements UserService{
     
         @Resource //resource注解可以让springboot自动给他注入依赖，否则UserInfoMapper找不到
         private UserInfoMapper userInfoMapper; //属性需要在实现类中说明
     
         @Override
         public int addUserInfo(UserInfo userInfo) {
             //TODO
             return 0;
         }
     
         @Override
         public int modifyUserInfo(UserInfo userInfo) {
             return 0;
         }
     
         @Override
         public int deleteUserById(String id) {
             return 0;
         }
     
    @Override
         public List<UserInfo> queryUserList(UserInfo userInfo) {
             return null;
         }
     }
     
    ```
    
     

## 3.２ 实现controller

 ![](D:/blog/themes/yilia/source/img/java/produce_practice/3/2.png)![](D:/blog/themes/yilia/source/img/java/produce_practice/3/3.png)

![](img/java/produce_practice/3/2.png)

![](img/java/produce_practice/3/3.png)

1. 在biz/user/controller下创建UserController类，并注解@RestController，使用rest的接口风格；注解@RequestMapping("/admin")，使用request请求

2. HttpResponseEntity是一个通用的给前端返回结果的类，因此我们在questionnaire目录下创建一个beans目录存放一些通用的bean，在beans里面创建HttpResponseEntity.java

   ```java
   package com.lyingedu.questionnaire.beans;
   
   import lombok.Data;
   
   import java.io.Serializable;
   
   @Data //lombook的data注解可以省略get set方法
   public class HttpResponseEntity implements Serializable {
       private String code;//状态码
       private Object data;//返回数据
       private String message;//状态消息
   
       public HttpResponseEntity(){ //默认初始信息
           this.code="0";
           this.message="操作失败";
       }
   
   
   }
   
   ```

   - 注意：由于代码中使用了lombok，你需要在idea中下载lombok插件并启用其注解处理（一般idea会提示你）

3. 继续完善controller：

   ```java
   package com.lyingedu.questionnaire.biz.user.controller;
   
   import com.lyingedu.questionnaire.beans.HttpResponseEntity;
   import com.lyingedu.questionnaire.biz.user.service.UserService;
   import com.lyingedu.questionnaire.dbmap.entities.UserInfo;
   import com.lyingedu.questionnaire.dbmap.imapper.UserInfoMapper;
   import jakarta.annotation.Resource;
   import org.springframework.web.bind.annotation.RequestMapping;
   import org.springframework.web.bind.annotation.RestController;
   
   @RestController //rest的接口风格
   @RequestMapping("/admin") //使用request请求
   public class UserController {
   
       @Resource
       private UserService userService;
   
       /**
        * 用户列表
        * @param userInfo
        * @return
        */
       public HttpResponseEntity queryUserList(UserInfo userInfo){
   
           //TODO
           return null;
       }
   
       /**
        * 增加用户
        * @param userInfo
        * @return
        */
       public HttpResponseEntity addUserInfo(UserInfo userInfo) {
           //TODO
           return null;
       }
   
       /**
        * 修改用户
        * @param userInfo
        * @return
        */
       public HttpResponseEntity modifyUserInfo(UserInfo userInfo) {
   
           //TODO
           return null;
       }
   
       /**
        * 删除用户
        * @param userInfo
        * @return
        */
       public HttpResponseEntity deleteUserById(UserInfo userInfo) {
           //TODO
           return null;
       }
   
       /**
        * 用户登录
        * @param userInfo
        * @return
        */
       public HttpResponseEntity userLogin(UserInfo userInfo) {
   
           //TODO
           return null;
       }
   }
   
   ```

   