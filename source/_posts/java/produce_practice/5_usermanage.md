---
title: 生产实习-5 用户管理功能

date: 2024-3-5 11:00:00

tags: [生产实习,java,springboot]

categories: [java]

comment: true

toc: true


---

#
<!--more-->



# 5. 用户管理功能

## 5.1 用户列表功能

- 查看pages/user/index.js可以看到需要完成的函数是：queryUserList

### 5.1.1 control层queryUserList

```java
    @PostMapping("/queryUserList")
    public HttpResponseEntity queryUserList(@RequestBody UserInfo userInfo){

        HttpResponseEntity httpResponseEntity = new HttpResponseEntity();

        List<UserInfo> userInfos = userService.queryUserList(userInfo);
        httpResponseEntity.setCode("666");
        httpResponseEntity.setData(userInfos);

        return httpResponseEntity;
    }
```

### 5.1.2 server层queryUserList

```java
    public List<UserInfo> queryUserList(UserInfo userInfo) {//1. 用userInof查询数据库

        UserInfoExample userInfoExample = new UserInfoExample(); //3 在dbmap/entities里面是有userinfoexample的

        if(userInfo.getUsername()!=null){//登录
            UserInfoExample.Criteria userInfoCriteria = userInfoExample.createCriteria(); //4 然后用这个example创建一个条件
            userInfoCriteria.andUsernameEqualTo(userInfo.getUsername()) //根据这两个条件进行查询
                    .andPasswordEqualTo(userInfo.getPassword());
        }
        //没有条件的就是全查

        return userInfoMapper.selectByExample(userInfoExample);//2 使用userInfoMapper里面自带一些方法，使用里面的selectbyexample查询
    }
```

## 5.2 用户编辑

- 首先在浏览器检查源码，查看编辑用户按钮调用的是哪个函数（handleEdit)：

  ![](D:/blog/themes/yilia/source/img/java/produce_practice/5/1.png)

  ![](img/java/produce_practice/5/1.png)

  - 在pages/user/index.js里面也可以看到编辑按钮对应的是handleEdit，然后在handleEdit里面发现又跳转到创建用户页。
  - 然后去createUser/index.html里面看到创建用户按钮对应index.js下的handleCreateUser函数，在handleCreateUser函数里面判断，如果传入了user.id说明是要修改，没传入则是要新建，在修改部分找到路径：/admin/modifyUserInfo。

### 5.2.1 control层modifyUserInfo

```java
    @PostMapping("/modifyUserInfo")
    public HttpResponseEntity modifyUserInfo(@RequestBody UserInfo userInfo) {

        HttpResponseEntity httpResponseEntity = new HttpResponseEntity();
        userService.modifyUserInfo(userInfo);
        httpResponseEntity.setCode("666");
        return httpResponseEntity;
    }
```

### 5.2.2 service层modifyUserInfo

```java
    public int modifyUserInfo(UserInfo userInfo) {
        //根据主键修改，只修改提供的部分，不提供的部分不修改，不带selective的方法对于不提供的部分默认改成null
        return userInfoMapper.updateByPrimaryKeySelective(userInfo);
    }
```

## 5.3 删除用户

- 同理，在user/index.js里面找到删除按钮，对应deleteUser函数，在deleteUser函数中找到路径admin/deleteUserinfo，所以要写deleteUserinfo函数

### 5.3.1 control层deleteUserinfo

```java
    @PostMapping("/deleteUserinfo")
    public HttpResponseEntity deleteUserById(@RequestBody UserInfo userInfo) {
        
        userService.deleteUserById(userInfo.getId());
        HttpResponseEntity httpResponseEntity = new HttpResponseEntity();
        httpResponseEntity.setCode("666");
        
        return httpResponseEntity;
    }
```



### 5.3.2 server层deleteUserinfo

```java
    public int deleteUserById(String id) {
        return userInfoMapper.deleteByPrimaryKey(id);
        
    }
```