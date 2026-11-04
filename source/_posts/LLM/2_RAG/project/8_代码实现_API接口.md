---
title: 8_代码实现_API接口
date: 2026-10-27
tags:
  - 项目
categories:
  - 项目
comment: true
toc: true
published: true
permalink: llm/2_RAG/project
hexo-path:
---
#
<!--more-->

# 1. 模块设计
## 1.1 模块定位
- 已解决：所有后端任务
	- 存储结构
	- 文档入库
	- 单次检索
	- RAG Pipeline
	- rag tool
	- Agent 运行基础与状态管理
	- Agent会话编排
- 当前：实现 **API 接口层**，通过 FastAPI 把之前的后端业务能力暴露为 HTTP 接口，处理参数校验、用户认证与权限、调用 Chat Service、会话查询、文档上传/删除，以及 SSE 响应。
## 1.2 模块流程设计
### 1.2.1 用户聊天链路
```
前端 Vue / Pinia
    ↓ HTTP POST + Bearer Token
FastAPI app.py
    ↓ api/router.py 路由分发
/chat 或 /chat/stream
    ↓ Depends(get_current_user) 验证 JWT + 查询 User
ChatRequest 校验 message / session_id
    ↓
同步 /chat                  流式 /chat/stream
    ↓                              ↓
chat_with_agent()         chat_with_agent_stream()
    ↓                              ↓
完整 ChatResponse         StreamingResponse (text/event-stream)
    ↓                              ↓
JSON 响应                 rag_step / content / trace / hitl_request / [DONE]
    └──────────────┬───────────────┘
                   ↓
           前端展示 AI 回复
```
### 1.2.2 登录与会话管理链路
```
 注册 /auth/register ──→ 校验角色邀请码 → Hash 密码 → PostgreSQL User → JWT  
 登录 /auth/login ─────→ 校验账号密码 ────────────────────────→ JWT  
                    ↓  
 前端 Authorization: Bearer <token>  
                    ↓  
              get_current_user()  
                    ↓  
 查询当前用户信息 /auth/me  
 查询会话列表     GET /sessions  
 查询会话消息     GET /sessions/{session_id}  
 删除本人会话     DELETE /sessions/{session_id}
```

### 1.2.3 管理员文档管理链路
```
管理员 Bearer Token  
         ↓ require_admin()  
 GET /documents ───────────────────────→ Milvus 查询文件列表  
 POST /documents/upload/async  
         ↓ 文件类型校验 + 上传到 data/documents/  
         ↓ 创建 job_id  
         ↓ BackgroundTasks(_process_upload_job)  
         ↓ 清理同名文档 → 文档解析与三级分块  
         ↓ L1/L2 → ParentChunkStore (PostgreSQL + Redis)  
         ↓ L3 → Embedding → Milvus  
         ↓ 更新内存中的 UploadJobManager  
         └── GET /documents/upload/jobs/{job_id} 轮询进度  
 ​  
 DELETE /documents/delete/async/{filename}  
         ↓ 创建 job_id + BackgroundTasks(_process_delete_job)  
         ↓ 删除 Milvus → 删除 PostgreSQL / Redis 父块  
         └── GET /documents/delete/jobs/{job_id} 轮询进度
```
 
## 1.3 模块结构设计
根据 API 层的职责，将整个模块划分为以下 **6 个部分**：

1. **应用初始化与路由注册**：`app.py`、`api/router.py`，创建 FastAPI 应用、注册中间件和业务路由，统一 HTTP 请求入口。
2. **用户认证与权限管理**：`routes/auth.py`、`infra/auth.py`，完成用户注册、登录、JWT 身份识别及管理员权限校验。
3. **聊天接口**：`routes/chat.py`，校验聊天请求并调用第 7 模块的同步/流式 Chat Service，返回 JSON 或 SSE。
4. **会话管理接口**：`routes/sessions.py`，查询当前用户的会话列表、历史消息，以及删除指定会话。
5. **文档管理接口**：`routes/documents.py`，提供文档列表、同步/后台上传、同步/后台删除和任务进度查询；复用第 2 模块的入库能力。
6. **API 公共资源与任务状态**：`api/resources.py`、`jobs/upload_jobs.py`，封装文件存储、格式校验、跨存储删除与上传/删除任务状态管理，供文档接口内部调用。
## 1.4 文件结构设计
```python
backend/
├── app.py                         # FastAPI 程序入口、中间件、路由和前端静态资源
├── env.py                         # 启动时加载项目根目录 .env
├── api/
│   ├── router.py                  # 汇总四组业务路由
│   ├── resources.py               # 文档上传目录、组件复用、删除辅助函数
│   └── routes/
│       ├── auth.py                # 注册、登录、当前用户
│       ├── chat.py                # 同步聊天、SSE 流式聊天
│       ├── sessions.py            # 会话列表、消息、删除
│       └── documents.py           # 文档列表、上传、删除、后台任务查询
├── infra/
│   ├── auth.py                    # 认证依赖、JWT、PBKDF2、角色权限
│   └── database.py                # SessionLocal / init_db（第 1 模块已有）
├── schemas/
│   ├── auth.py                    # Auth 请求/响应 Schema
│   ├── chat.py                    # Chat/Session Schema（第 6 模块已有）
│   └── documents.py               # Document/Job Schema
├── chat/
│   ├── service.py                 # 第 7 模块：真正执行 Agent 和 HITL
│   └── storage.py                 # 第 6 模块：聊天消息存储
└── jobs/
    └── upload_jobs.py            # 上传/删除进度状态（进程内存）
```
## 1.5 API 接口清单

### 1.5.1 应用初始化与路由注册

- 负责创建 FastAPI 应用并注册路由。

|内部入口|所在文件|作用|
|---|---|---|
|`create_app()`|`backend/app.py`|创建应用，注册启动事件、中间件、业务路由和静态文件服务|
|`router.include_router(...)`|`backend/api/router.py`|统一注册 auth、sessions、chat、documents 四组路由|

### 1.5.2 用户认证与权限管理

| 接口               | 方法   | 权限   | 功能          | 核心调用                                          |
| ---------------- | ---- | ---- | ----------- | --------------------------------------------- |
| `/auth/register` | POST | 无需登录 | 注册用户        | `resolve_role()`、密码哈希、创建 User                 |
| `/auth/login`    | POST | 无需登录 | 用户登录并生成 JWT | `authenticate_user()`、`create_access_token()` |
| `/auth/me`       | GET  | 已登录  | 获取当前用户信息    | `get_current_user()`                          |

### 1.5.3 聊天接口

|接口|方法|权限|功能|核心调用|
|---|---|---|---|---|
|`/chat`|POST|已登录|同步聊天，返回完整 JSON 响应|`chat_with_agent()`|
|`/chat/stream`|POST|已登录|流式聊天，通过 SSE 实时返回进度和回答|`chat_with_agent_stream()`|

两个接口都使用 `ChatRequest` 校验请求，区别在于同步返回 `ChatResponse`，流式返回 `StreamingResponse`。

### 1.5.4 会话管理接口

| 接口                       | 方法     | 权限  | 功能           | 核心调用                             |
| ------------------------ | ------ | --- | ------------ | -------------------------------- |
| `/sessions`              | GET    | 已登录 | 获取当前用户的会话列表  | `storage.list_session_infos()`   |
| `/sessions/{session_id}` | GET    | 已登录 | 获取指定会话的历史消息  | `storage.get_session_messages()` |
| `/sessions/{session_id}` | DELETE | 已登录 | 删除指定会话及其聊天记录 | `storage.delete_session()`       |

所有会话操作均基于当前登录用户的身份进行隔离。

### 1.5.5 文档管理接口

| 接口                                   | 方法     | 功能             | 核心调用                                   |
| ------------------------------------ | ------ | -------------- | -------------------------------------- |
| `/documents`                         | GET    | 获取知识库文档列表      | `milvus_manager.query()`               |
| `/documents/upload/async`            | POST   | 创建后台文档上传任务     | `BackgroundTasks(_process_upload_job)` |
| `/documents/upload/jobs/{job_id}`    | GET    | 查询指定上传任务的状态和进度 | 上传任务状态查询                               |
| `/documents/upload/jobs`             | GET    | 查询所有上传任务       | 上传任务列表查询                               |
| `/documents/delete/async/{filename}` | DELETE | 创建后台文档删除任务     | `BackgroundTasks(_process_delete_job)` |
| `/documents/delete/jobs/{job_id}`    | GET    | 查询指定删除任务的状态和进度 | 删除任务状态查询                               |
| `/documents/upload`                  | POST   | 同步执行文档上传与入库    | 文档入库模块                                 |
| `/documents/{filename}`              | DELETE | 同步删除指定文档及关联数据  | 跨存储删除函数                                |

这里有两种任务执行方式：

- **同步接口**：在本次 HTTP 请求中完成上传或删除，再返回结果。
    
- **后台任务接口**：先返回 `job_id`，由 `BackgroundTasks` 执行任务，前端通过任务查询接口获取进度。

### 1.5.6 API 公共资源与任务状态

- 不独立提供 HTTP 接口，主要作为文档管理模块的内部依赖。

|内部函数|所在文件|作用|
|---|---|---|
|`is_supported_document()`|`api/resources.py`|检查文档类型是否受支持|
|`ensure_upload_dir()`|`api/resources.py`|确保文档上传目录存在|
|`save_upload_file()`|`api/resources.py`|保存上传的文件|
|`delete_document_transactionally()`|`api/resources.py`|删除 Milvus 向量及 PostgreSQL/Redis 父块数据|
|`_process_upload_job()`|`routes/documents.py`|执行后台文档解析、分块和入库|
|`_process_delete_job()`|`routes/documents.py`|执行后台文档删除|
|`UploadJobManager.create_job()`|`jobs/upload_jobs.py`|创建任务并初始化状态|
|`UploadJobManager.get_job()`|`jobs/upload_jobs.py`|查询任务状态|
|`UploadJobManager.update_step()`|`jobs/upload_jobs.py`|更新任务执行进度|

这些函数为上层 HTTP 接口提供公共能力，避免文档上传、删除及任务查询接口重复实现底层逻辑。
# 2. 模块实现
## 2.1 应用初始化与路由注册
| 内部入口                         | 所在文件                    | 作用                                     |
| ---------------------------- | ----------------------- | -------------------------------------- |
| `create_app()`               | `backend/app.py`        | 创建应用，注册启动事件、中间件、业务路由和静态文件服务            |
| `router.include_router(...)` | `backend/api/router.py` | 统一注册 auth、sessions、chat、documents 四组路由 |
### 2.1.1 应用入口
**执行流程：** 加载 `.env` → 创建 FastAPI → 注册启动时初始化数据库 → 安装 CORS 和禁止缓存中间件 → 注册业务路由 → 有前端构建产物时挂载静态文件 → 返回 `app`。
> 中间件：在 HTTP 请求到达业务函数之前，先经过的一层公共处理程序。例如前端发送一个聊天请求，通常顺序是：前端->FastAPI/chat接口->chat_with_agent后端函数->返回回答。现在假设我们希望：每次请求都记录一下处理耗时。如果没有中间件，就要在每个接口里面写计时代码。有了中间件，就可以统一处理。中间件可以：在业务接口执行之前处理请求，在业务接口返回响应之后处理响应。

>`CORSMiddleware`（跨域资源共享中间件），作用是：控制浏览器中的前端页面能否跨域访问后端 API。假设 SuperMew 部署在本地，但是前后端部署在不同端口，浏览器就认为它们属于不同的源（Origin）。当前端请求后端 `fetch("http://localhost:8000/chat")` 时，浏览器会进行 CORS 检查。如果后端没有明确允许这个来源，浏览器可能阻止前端读取响应。
>`CORSMiddleware` 就负责告诉浏览器：哪些前端来源可以访问后端接口。比如实际部署后，只允许官网 `https://app.supermew.com` 跨域访问后端 API。如果用户已经登录官网，此时又打开一个恶意网站 `https://evil.com`，该网站的 JavaScript 试图利用用户已有的登录凭据请求 SuperMew 后端，读取用户聊天记录，那么浏览器会根据 CORS 规则阻止恶意网站读取响应。
>**需要注意：CORS 限制的是浏览器中的跨域响应读取，而不是阻止请求到达服务器。** Python、curl 等客户端不受 CORS 限制，因此后端仍然需要 JWT 身份认证、权限校验等安全机制。

### 2.1.2 路由注册





### 1.5.2 用户认证与权限管理

| 接口               | 方法   | 权限   | 功能          | 核心调用                                          |
| ---------------- | ---- | ---- | ----------- | --------------------------------------------- |
| `/auth/register` | POST | 无需登录 | 注册用户        | `resolve_role()`、密码哈希、创建 User                 |
| `/auth/login`    | POST | 无需登录 | 用户登录并生成 JWT | `authenticate_user()`、`create_access_token()` |
| `/auth/me`       | GET  | 已登录  | 获取当前用户信息    | `get_current_user()`                          |

### 1.5.3 聊天接口

|接口|方法|权限|功能|核心调用|
|---|---|---|---|---|
|`/chat`|POST|已登录|同步聊天，返回完整 JSON 响应|`chat_with_agent()`|
|`/chat/stream`|POST|已登录|流式聊天，通过 SSE 实时返回进度和回答|`chat_with_agent_stream()`|

两个接口都使用 `ChatRequest` 校验请求，区别在于同步返回 `ChatResponse`，流式返回 `StreamingResponse`。

### 1.5.4 会话管理接口

| 接口                       | 方法     | 权限  | 功能           | 核心调用                             |
| ------------------------ | ------ | --- | ------------ | -------------------------------- |
| `/sessions`              | GET    | 已登录 | 获取当前用户的会话列表  | `storage.list_session_infos()`   |
| `/sessions/{session_id}` | GET    | 已登录 | 获取指定会话的历史消息  | `storage.get_session_messages()` |
| `/sessions/{session_id}` | DELETE | 已登录 | 删除指定会话及其聊天记录 | `storage.delete_session()`       |

所有会话操作均基于当前登录用户的身份进行隔离。

### 1.5.5 文档管理接口

| 接口                                   | 方法     | 功能             | 核心调用                                   |
| ------------------------------------ | ------ | -------------- | -------------------------------------- |
| `/documents`                         | GET    | 获取知识库文档列表      | `milvus_manager.query()`               |
| `/documents/upload/async`            | POST   | 创建后台文档上传任务     | `BackgroundTasks(_process_upload_job)` |
| `/documents/upload/jobs/{job_id}`    | GET    | 查询指定上传任务的状态和进度 | 上传任务状态查询                               |
| `/documents/upload/jobs`             | GET    | 查询所有上传任务       | 上传任务列表查询                               |
| `/documents/delete/async/{filename}` | DELETE | 创建后台文档删除任务     | `BackgroundTasks(_process_delete_job)` |
| `/documents/delete/jobs/{job_id}`    | GET    | 查询指定删除任务的状态和进度 | 删除任务状态查询                               |
| `/documents/upload`                  | POST   | 同步执行文档上传与入库    | 文档入库模块                                 |
| `/documents/{filename}`              | DELETE | 同步删除指定文档及关联数据  | 跨存储删除函数                                |

这里有两种任务执行方式：

- **同步接口**：在本次 HTTP 请求中完成上传或删除，再返回结果。
    
- **后台任务接口**：先返回 `job_id`，由 `BackgroundTasks` 执行任务，前端通过任务查询接口获取进度。

### 1.5.6 API 公共资源与任务状态

- 不独立提供 HTTP 接口，主要作为文档管理模块的内部依赖。

|内部函数|所在文件|作用|
|---|---|---|
|`is_supported_document()`|`api/resources.py`|检查文档类型是否受支持|
|`ensure_upload_dir()`|`api/resources.py`|确保文档上传目录存在|
|`save_upload_file()`|`api/resources.py`|保存上传的文件|
|`delete_document_transactionally()`|`api/resources.py`|删除 Milvus 向量及 PostgreSQL/Redis 父块数据|
|`_process_upload_job()`|`routes/documents.py`|执行后台文档解析、分块和入库|
|`_process_delete_job()`|`routes/documents.py`|执行后台文档删除|
|`UploadJobManager.create_job()`|`jobs/upload_jobs.py`|创建任务并初始化状态|
|`UploadJobManager.get_job()`|`jobs/upload_jobs.py`|查询任务状态|
|`UploadJobManager.update_step()`|`jobs/upload_jobs.py`|更新任务执行进度|

这些函数为上层 HTTP 接口提供公共能力，避免文档上传、删除及任务查询接口重复实现底层逻辑。