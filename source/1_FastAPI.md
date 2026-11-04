---
title: 1_FastAPI
date: 2026-11-01
tags:
comment: true
toc: true
published: true
permalink:
hexo-path:
---
#
<!--more-->

# 1. Web编程

Web网站由三个核心部分组成：前端程序、服务端程序、数据库。

- 前端程序：负责前端页面的展示，网页由 HTML(负责网页的结构)、CSS(负责网页的表现)、JavaScript(负责网页的动作) 三个部分组成 。
- 服务端程序：可以基于Python中的Django、Flask 或 FastAPI来进行开发。
- 数据库：负责数据的存储与管理

# 2. FastAPI
[https://fastapi.org.cn](https://fastapi.org.cn/)
- API接口：应用程序编程接口(Application Programming Interface)，就是对外提供的**功能入口，供别人来调用**（比如：天气查询的API接口）。FastAPI就是一个Web框架，用于基于标准python类型提示构建API
```python
from fastapi import FastAPI

# 1. 创建实例
app = FastAPI()

# 2. 定义API接口
@app.get('/')
def root():
	return {'message': "hello world"}
@app.get('/users')
def get_users():
	return [
		{'id':1, 'name':'Alice', 'age':10},
		{'id':2, 'name':'Bob', 'age':11}
	]

# 3. 启动FastAPI服务器
if __name__ == '__main__':
	import uvicorn
	uvicorn.run(app, host='0.0.0.0', port=8000)
# 或者命令行启动
# fastapi dev 'xxx.py'
# uvicorn 文件名:实例名（app） --reload
```

# 3. Restful
- 遵循REST架构风格的API接口服务，而REST（REpresentational State Transfer），表述性状态转换，它是一种软件架构风格。不是什么强制性规则。
- 传统非restful的方式来定义url时不规范，比如添加用户这个url，有的开发定义成`/user/add`有的定义成`/user/adduser`：

| 传统风格url                                        | 请求方式     | 含义            |
| ---------------------------------------------- | -------- | ------------- |
| http://localhost:8000/user/getById?id=1<br>    | GET<br>  | 查询id为1的用户<br> |
| http://localhost:8000/user/saveUser<br>        | POST<br> | 新增用户<br>      |
| http://localhost:8000/user/updateUser<br>      | POST<br> | 修改用户<br>      |
| http://localhost:8000/user/deleteUser?id=1<br> | GET<br>  | 删除id为1的用户<br> |
- rest风格：URL定位资源，HTTP动词描述操作


| rest风格url                                 | 请求方式       | 含义                |
| ----------------------------------------- | ---------- | ----------------- |
| http://localhost:8000/users/1<br><br><br> | GET<br>    | 查询id为1的用户<br>     |
| http://localhost:8000/users/1<br><br>     | DELETE<br> | 删除id为1的用户<br><br> |
| http://localhost:8000/users<br><br>       | POST<br>   | 新增用户<br><br>      |
| http://localhost:8000/users<br><br>       | PUT<br>    | 修改用户<br><br>      |
