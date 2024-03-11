---
title: 生产实习-7 项目功能开发

date: 2024-3-8 12:00:00

tags: [生产实习,java,springboot]

categories: [java]

comment: true

toc: true
---

#
<!--more-->

# 7 项目功能开发
![](D:/blog/themes/yilia/source/img/java/produce_practice/6/0.png)

![](img/java/produce_practice/6/0.png)

![](D:/blog/themes/yilia/source/img/java/produce_practice/6/1.png)

![](img/java/produce_practice/6/1.png)

## 7.1 项目实体创建

- 创建ProjectInfo（之前用dbgan程序生成过），还需要在数据库创建表：

  ```sql
  CREATE TABLE IF NOT EXISTS `myweb`.`project_info` (
    `id` VARCHAR(45) NOT NULL,
    `user_id` VARCHAR(45) NULL,
    `project_name` VARCHAR(45) NULL,
    `project_content` VARCHAR(45) NULL,
    `created_by` VARCHAR(45) NULL,
    `creation_date` DATETIME NULL,
    `last_updated_by` VARCHAR(45) NULL,
    `last_update_date` DATETIME NULL,
    PRIMARY KEY (`id`))
  ENGINE = InnoDB
  ```

## 7.2 数据库映射创建

- 创建ProjectInfoMapper接口（之前dbgan程序生成过xml文件）

## 7.3 创建功能接口

- 先写service层的接口：ProjectService：

  ![](D:/blog/themes/yilia/source/img/java/produce_practice/6/3.png)

  ![](img/java/produce_practice/6/3.png)

  ```java
  package com.lyingedu.questionnaire.biz.user.service;
  
  import com.lyingedu.questionnaire.dbmap.entities.ProjectInfo;
  
  import java.util.List;
  
  public interface ProjectService {
      /**
       * 增加项目
       * @param projectInfo
       * @return
       */
      int addProjectInfo(ProjectInfo projectInfo, String UserId);
  
      /**
       * 修改项目
       * @param projectInfo
       * @return
       */
      int modifyProjectInfo(ProjectInfo projectInfo);
  
      /**
       * 删除项目
       * @param id
       * @return
       */
      int deleteProjectById(String id);
  
      /**
       * 查询项目列表
       * @return
       */
      List<ProjectInfo> queryProjectList(ProjectInfo projectInfo);
  }
  
  ```

- 再写service实现类：ProjectServiceImpl:

  ```java
  package com.lyingedu.questionnaire.biz.user.service;
  
  import com.lyingedu.questionnaire.dbmap.entities.ProjectInfo;
  import com.lyingedu.questionnaire.dbmap.imapper.ProjectInfoMapper;
  import jakarta.annotation.Resource;
  import org.springframework.stereotype.Service;
  
  import java.util.List;
  
  @Service
  public class ProjectServiceImpl implements ProjectService {
      
      @Resource
      private ProjectInfoMapper projectInfoMapper;
  
      @Override
      public int addProjectInfo(ProjectInfo projectInfo, String UserId) {
          return 0;
      }
  
      @Override
      public int modifyProjectInfo(ProjectInfo projectInfo) {
          return 0;
      }
  
      @Override
      public int deleteProjectById(String id) {
          return 0;
      }
  
      @Override
      public List<ProjectInfo> queryProjectList(ProjectInfo projectInfo) {
          return null;
      }
  
  
  }
  
  ```

- 最后写control层：

  ![](D:/blog/themes/yilia/source/img/java/produce_practice/6/2.png)

  ![](img/java/produce_practice/6/2.png)

  ```java
  package com.lyingedu.questionnaire.biz.user.controller;
  
  import com.lyingedu.questionnaire.beans.HttpResponseEntity;
  import com.lyingedu.questionnaire.biz.user.service.ProjectService;
  import com.lyingedu.questionnaire.dbmap.entities.ProjectInfo;
  import jakarta.annotation.Resource;
  import lombok.extern.slf4j.Slf4j;
  import org.springframework.web.bind.annotation.RequestBody;
  import org.springframework.web.bind.annotation.RestController;
  
  
  @RestController //rest的接口风格
  @Slf4j
  public class ProjectController {
      @Resource
      private ProjectService projectService;
  
      public HttpResponseEntity queryProjectList(@RequestBody ProjectInfo projectInfo){
          return null;
      }
  
      public HttpResponseEntity addProjectInfo(@RequestBody ProjectInfo projectInfo){
          return null;
      }
  
      public HttpResponseEntity modifyProjectInfo(@RequestBody ProjectInfo projectInfo){
          return null;
      }
  
      public HttpResponseEntity deleteProjectById(@RequestBody String id){
          return null;
      }
  
  }
  
  ```

  

## 7.3 项目列表功能

- 先查看对应的前端代码得到路径

### 7.3.1 control层

- ProjectController.queryProjectList：

```java
    @PostMapping("/queryProjectList")
    public HttpResponseEntity queryProjectList(@RequestBody ProjectInfo projectInfo){

        HttpResponseEntity httpResponseEntity = new HttpResponseEntity();

        List<ProjectInfo> projectInfos = projectService.queryProjectList(projectInfo);
        if(projectInfos.size() == 1) {
            httpResponseEntity.setCode("666");
        }
        httpResponseEntity.setData(projectInfos);

        return httpResponseEntity;
    }
```



### 7.3.2 service层

- ProjectServiceImpl.queryProjectList：

```java
    public List<ProjectInfo> queryProjectList(ProjectInfo projectInfo) {

        ProjectInfoExample projectInfoExample = new ProjectInfoExample();
        ProjectInfoExample.Criteria projectInfoCriteria = projectInfoExample.createCriteria();
        projectInfoCriteria.andCreatedByEqualTo(projectInfo.getCreatedBy());
        return projectInfoMapper.selectByExample(projectInfoExample);
    }
```

### 7.3.3 数据库映射

- 实现selectByEanmple：

- imapper函数接口（已经生成过了）：

  ```java
  package com.lyingedu.questionnaire.dbmap.imapper;
  
  import com.lyingedu.questionnaire.dbmap.entities.ProjectInfo;
  import com.lyingedu.questionnaire.dbmap.entities.ProjectInfoExample;
  import java.util.List;
  import org.apache.ibatis.annotations.Param;
  
  public interface ProjectInfoMapper {
      /**
       * This method was generated by MyBatis Generator.
       * This method corresponds to the database table project_info
       *
       * @mbg.generated Tue Mar 05 14:17:07 CST 2024
       */
      long countByExample(ProjectInfoExample example);
  
      /**
       * This method was generated by MyBatis Generator.
       * This method corresponds to the database table project_info
       *
       * @mbg.generated Tue Mar 05 14:17:07 CST 2024
       */
      int deleteByExample(ProjectInfoExample example);
  
      /**
       * This method was generated by MyBatis Generator.
       * This method corresponds to the database table project_info
       *
       * @mbg.generated Tue Mar 05 14:17:07 CST 2024
       */
      int deleteByPrimaryKey(String id);
  
      /**
       * This method was generated by MyBatis Generator.
       * This method corresponds to the database table project_info
       *
       * @mbg.generated Tue Mar 05 14:17:07 CST 2024
       */
      int insert(ProjectInfo row);
  
      /**
       * This method was generated by MyBatis Generator.
       * This method corresponds to the database table project_info
       *
       * @mbg.generated Tue Mar 05 14:17:07 CST 2024
       */
      int insertSelective(ProjectInfo row);
  
      /**
       * This method was generated by MyBatis Generator.
       * This method corresponds to the database table project_info
       *
       * @mbg.generated Tue Mar 05 14:17:07 CST 2024
       */
      List<ProjectInfo> selectByExample(ProjectInfoExample example);
  
      /**
       * This method was generated by MyBatis Generator.
       * This method corresponds to the database table project_info
       *
       * @mbg.generated Tue Mar 05 14:17:07 CST 2024
       */
      ProjectInfo selectByPrimaryKey(String id);
  
      /**
       * This method was generated by MyBatis Generator.
       * This method corresponds to the database table project_info
       *
       * @mbg.generated Tue Mar 05 14:17:07 CST 2024
       */
      int updateByExampleSelective(@Param("row") ProjectInfo row, @Param("example") ProjectInfoExample example);
  
      /**
       * This method was generated by MyBatis Generator.
       * This method corresponds to the database table project_info
       *
       * @mbg.generated Tue Mar 05 14:17:07 CST 2024
       */
      int updateByExample(@Param("row") ProjectInfo row, @Param("example") ProjectInfoExample example);
  
      /**
       * This method was generated by MyBatis Generator.
       * This method corresponds to the database table project_info
       *
       * @mbg.generated Tue Mar 05 14:17:07 CST 2024
       */
      int updateByPrimaryKeySelective(ProjectInfo row);
  
      /**
       * This method was generated by MyBatis Generator.
       * This method corresponds to the database table project_info
       *
       * @mbg.generated Tue Mar 05 14:17:07 CST 2024
       */
      int updateByPrimaryKey(ProjectInfo row);
  }
  ```

  

- xmapper执行sql（已经生成过了）：

  ```java
    <select id="selectByExample" parameterType="com.lyingedu.questionnaire.dbmap.entities.ProjectInfoExample" resultMap="BaseResultMap">
      <!--
        WARNING - @mbg.generated
        This element is automatically generated by MyBatis Generator, do not modify.
        This element was generated on Tue Mar 05 14:17:07 CST 2024.
      -->
      select
      <if test="distinct">
        distinct
      </if>
      <include refid="Base_Column_List" />
      from project_info
      <if test="_parameter != null">
        <include refid="Example_Where_Clause" />
      </if>
      <if test="orderByClause != null">
        order by ${orderByClause}
      </if>
    </select>
  ```

  

### 7.3.3 测试

- 登陆后会显示当前用户创建的问卷

## 7.4 新建项目功能

- 查看前端代码得到路径

### 7.4.1 control层

```java
    @PostMapping("/addProjectInfo")
    public HttpResponseEntity addProjectInfo(@RequestBody ProjectInfo projectInfo) {

        HttpResponseEntity httpResponseEntity = new HttpResponseEntity();
        int count = projectService.addProjectInfo(projectInfo, projectInfo.getCreatedBy()); //返回受影响的行数 //用户名要唯一
        if(count == 1) { //插入成功则状态码为666，否则为默认错误值
            httpResponseEntity.setCode("666");
        }
        return httpResponseEntity;
    }
```

### 7.4.2 service层

```java
    public int addProjectInfo(ProjectInfo projectInfo, String userName) {

        projectInfo.setId(UUIDUtil.getOneUUID());
        projectInfo.setCreatedBy(userName);
        return projectInfoMapper.insert(projectInfo);
    }
```



## 7.5 项目修改

### 7.5.1 control层

```java
    @PostMapping("/modifyProjectInfo")
    public HttpResponseEntity modifyProjectInfo(@RequestBody ProjectInfo projectInfo) {

        HttpResponseEntity httpResponseEntity = new HttpResponseEntity();
        projectService.modifyProjectInfo(projectInfo);
        httpResponseEntity.setCode("666");
        httpResponseEntity.setMessage("修改成功");
        return httpResponseEntity;
    }
```

### 7.5.2 service层

```java
    public int modifyProjectInfo(ProjectInfo projectInfo) {
        return projectInfoMapper.updateByPrimaryKeySelective(projectInfo)
    }
```



## 7.6 项目删除

### 7.6.1 control层

```java
    @PostMapping("/deleteProjectById")
    public HttpResponseEntity deleteProjectById(@RequestBody ProjectInfo projectInfo) {

        HttpResponseEntity httpResponseEntity = new HttpResponseEntity();
        projectService.deleteProjectById(projectInfo.getId());
        httpResponseEntity.setCode("666");
        httpResponseEntity.setMessage("删除成功");
        return httpResponseEntity;
    }
```

### 7.6.2 service层

```java
    public int deleteProjectById(String id) {
        return projectInfoMapper.deleteByPrimaryKey(id);
    }
```

