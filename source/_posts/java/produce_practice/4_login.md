---
title: 生产实习-4 登录及创建用户功能开发

date: 2024-3-5 09:00:00

tags: [生产实习,java,springboot]

categories: [java]

comment: true

toc: true

---

#
<!--more-->



# 4. 登录及创建用户功能开发

## 4.1 登录功能

### 4.1.1 查看前端代码

- 首先在resources/static/pages/login/index.html查看登录按钮的动作：发现是调用了onLogin函数。

- 进入./index.js查看该函数：

  ![](D:/blog/themes/yilia/source/img/java/produce_practice/4/1.png)

  ![](img/java/produce_practice/4/1.png)

### 4.1.2 编写UserController.userLogin方法（control层）

- 然后我们到biz/user/controller/UserController，其中有

  ```java
  @RequestMapping("/admin") //使用request请求
  ```

  对应/admin路径。然后我们找到其中的userLogin方法，进行编写：

  ```java
      @PostMapping("/userLogin")
      public HttpResponseEntity userLogin(@RequestBody UserInfo userInfo) { //requestbody注解，将请求的json数据转换为对象
  
          HttpResponseEntity httpResponseEntity = new HttpResponseEntity();
          //TODO
          List<UserInfo> userInfos = userService.queryUserList(userInfo); //使用service层的方法查询用户信息
  
          if(userInfos.size() != 0) {//如果查询到用户信息则状态码为666，否则为默认值
              httpResponseEntity.setCode("666");
              httpResponseEntity.setData(userInfos.get(0));
          }
          return httpResponseEntity;//传给前端
      }
  ```

- 然后进行调试，可以先在该方法内打个断点，然后调试，进入 http://127.0.0.1:8085/pages/login/index.html，输入用户名密码，确认后可以看到代码运行到了断点处，可以查看一下userInfo变量是否能获得。

  - 注意：如果你输入了localhost而非127.0.0.1你可能会发现代码没有运行到断点处，无法查看变量。在浏览器打开检查页面。

    ![](D:/blog/themes/yilia/source/img/java/produce_practice/4/2.png)

    ![](img/java/produce_practice/4/2.png)

### 4.1.3 编写userserviceImpl.queryUserList方法（service层）

- 具体查询是在userserviceImpl.queryUserList中实现的：

  ```java
      /**
       * 查询用户列表
       * @param userInfo
       * @return
       */
      @Override
      public List<UserInfo> queryUserList(UserInfo userInfo) {//1. 用userInof查询数据库
  
          UserInfoExample userInfoExample = new UserInfoExample(); //3 在dbmap/entities里面是有userinfoexample的
          UserInfoExample.Criteria userInfoCriteria = userInfoExample.createCriteria(); //4 然后用这个example创建一个条件
          userInfoCriteria.andUsernameEqualTo(userInfo.getUsername()) //根据这两个条件进行查询
                          .andPasswordEqualTo(userInfo.getPassword());
  
  
          return userInfoMapper.selectByExample(userInfoExample);//2 使用userInfoMapper里面自带一些方法，使用里面的selectbyexample查询
      }
  }
  ```

### 4.1.4 调试结果

- 然后继续在usercontroller的返回那一行打个断点，继续调试，发现确认用户名密码后发生错误，调试终止。这是因为一些xml文件放在了java包下面，无法编译到工程中，需要在项目根目录的pom.xml的<build>中加入以下代码：

  ```xml
  	<resources>
  			<resource>
  				<directory>src/main/java</directory>
  				<includes>
  					<include>**/*.xml</include>
  				</includes>
  				<filtering>false</filtering>
  			</resource>
  			<resource>
  				<directory>src/main/resources</directory>
  			</resource>
  		</resources>
  ```

  ![](D:/blog/themes/yilia/source/img/java/produce_practice/4/3.png)

  ![](img/java/produce_practice/4/3.png)

  然后刷新maven即可。

- 此时发现userInfos的id是0，说明查询数据库成功了（id值来自数据库）

  ![](D:/blog/themes/yilia/source/img/java/produce_practice/4/4.png)

  ![](img/java/produce_practice/4/4.png)

  - 然后点击调试控制台的恢复程序按钮

     ![](D:/blog/themes/yilia/source/img/java/produce_practice/4/5.png)

    ![](img/java/produce_practice/4/5.png) 

    发现前端页面跳转：

    ![](D:/blog/themes/yilia/source/img/java/produce_practice/4/6.png)

    ![](img/java/produce_practice/4/6.png)

## 4.2 创建用户功能

### 4.2.1 查看前端代码

- 发现路径是admin/addUserInfo，参照4.1的经验，我们接下来要找control层的addUserInfo方法

  ![](D:/blog/themes/yilia/source/img/java/produce_practice/4/7.png)

  ![](img/java/produce_practice/4/7.png) 



### 4.2.2 编写UserController.userLogin方法（control层）

```java
    public HttpResponseEntity addUserInfo(@RequestBody UserInfo userInfo) {

        HttpResponseEntity httpResponseEntity = new HttpResponseEntity();
        int count = userService.addUserInfo(userInfo); //返回受影响的行数
        if(count > 0) { //插入成功则状态码为666，否则为默认错误值
            httpResponseEntity.setCode("666");
        }
        return httpResponseEntity;
    }
```

### 4.2.3 编写userserviceImpl.addUserInfo方法（service层）

- 由于数据库设计问题，每次插入数据都需要一个userid，userid不是自动编号的，需要自己写入，而前端没有写入，因此我们还需要一个在后端生成userid的方法在后端。为此我们创建src/main/java/com/lyingedu/questionnaire/common/utils/UUIDUtil.java:

  ```java
  package com.lyingedu.questionnaire.common.utils;
  
  
  import java.util.UUID;
  
  public class UUIDUtil {
  
      /**
       *获取一个UUID
       */
      public static String getOneUUID(){
          //获取UUID
          String s = UUID.randomUUID().toString();
          //去掉“-”符号
          return s.substring(0,8)+s.substring(9,13)+s.substring(14,18)+s.substring(19,23)+s.substring(24);
      }
      /**
       * 获得指定数目的UUID
       * @param number int 需要获得的UUID数量
       * @return String[] UUID数组
       */
      public static String[] getUUID(int number){
          if(number < 1){
              return null;
          }
          String[] ss = new String[number];
          for(int i=0;i<number;i++){
              ss[i] = getOneUUID();
          }
          return ss;
      }
  }
  
  ```

- 然后我们再来编写userserviceImpl.addUserInfo方法：

  ```java
      public int addUserInfo(UserInfo userInfo) {
  
          userInfo.setId(UUIDUtil.getOneUUID());
          userInfo.setStatus("1");//1有效，0无效
          
          return userInfoMapper.insert(userInfo);
      }
  ```

- 最后还需要将index.js里面的两句判断状态码从1改成自己的666：

  ```js
  if (res.code === "666") {
  ```

  