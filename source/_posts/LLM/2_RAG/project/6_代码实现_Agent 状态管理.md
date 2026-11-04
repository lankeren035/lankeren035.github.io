---
title: 6_代码实现_Agent 状态管理
date: 2026-10-25
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
- 已解决：
	- 存储结构
	- 文档入库
	- 单次检索
	- RAG Pipeline
	- rag tool
- 当前：构造Agent模块，先完成Agent 运行基础与状态管理：定义共享数据结构、管理一次 Query 的运行状态、持久化会话历史、创建绑定 Tool 的 Agent。
## 1.2 模块流程设计
```
                     用户发送 Query
                           ↓
                    加载当前会话数据
                    ├── 历史聊天记录
                    ├── 长期记忆摘要
                    └── 待处理的 HITL （判断当前query是不是对HITL的补充信息）
                           ↓
                    创建本轮请求 ctx（记录本次query执行过程的临时状态）
                           ↓
                 判断是否需要恢复 HITL
                    ↙             ↘
                  否               是
                  ↓                ↓
            构建 Agent 上下文    结合用户补充信息
            ├── 历史消息        拼接恢复 RAG 检索
            ├── 长期记忆             ↓
            └── 当前 Query       重新判断证据
                  ↓                ↓
              创建 Agent           │
                  ↓                │
             LLM 推理与决策         │
             ├── 直接回答           │
             ├── Knowledge Tool    │
             │      └── RAG        │
             └── Weather Tool      │
                  ↓                │
             获取工具执行结果        │
                  ↓                │
             LLM 生成回答           │
                  └────────┬───────┘
                           ↓
                    检查本轮结果
                   ↙          ↘
               正常回答       需要 HITL
                  ↓              ↓
             按条件更新记忆   生成澄清问题
                  ↓              ↓
                  │         保存待恢复 HITL 状态
                  ↓              ↓
             保存用户消息、AI 消息
                  和会话信息
                       ↓
                  关闭本轮 ctx
                       ↓
                   返回给用户
                   ↙        ↘
               正常答案    澄清问题
                             ↓
                       等待下一轮 Query
                             ↓
                       重新进入对话流程
```
## 1.3 数据结构设计
- 针对上述的Agent流程，我们需要设计几个数据结构，有些数据结构我们前面已经用到了，只是我们之前没有定义，现在在这里都给定义：
	- RAG相关：主要用于记录、校验和规范化 RAG 执行轨迹，不直接参与检索计算。
		- RetrievedChunk：检索得到的 Evidence 数据格式
		- RagTraceFields：定义 RAG 检索、评分、Rewrite 和路由等执行轨迹的公共字段。后面的父问题trace和子问题trace都继承自这里。
		- RagSubTrace：记录复杂问题中单个子问题的 RAG 执行轨迹。
		- RagTrace：记录完整 RAG 执行轨迹，包括主流程和各子问题的执行信息。
	- HITL相关：用于实现 HITL 跨轮恢复，而不是记录 RAG Trace。
		- HitlResumeState：保存 RAG 恢复所需的状态，供用户补充信息后继续检索。这个是在RAG层用到的。
		- PendingHitlState：用于保存跨轮待处理的 HITL 信息，包括澄清问题、用户已补充的内容和 `HitlResumeState`，以便下一轮继续处理。这个是在chat层用到的。
	- Chat 相关：负责单轮请求状态管理、聊天接口的请求与响应、历史消息查询以及会话管理。
		- `ChatRequestContext`：管理单次 Query 的临时状态，包括工具调用额度、RAG Trace、HITL 恢复信息和流式输出。
		- `ChatRequest`：定义用户发送给 Chat API 的请求格式。
		- `ChatResponse`：定义 Chat API 的响应格式。
		- `MessageInfo`：定义单条历史聊天消息的数据格式。
		- `SessionMessagesResponse`：定义查询某个会话历史消息时的返回格式。
		- `SessionInfo`：记录单个会话的基本信息。
		- `SessionListResponse`：定义查询用户会话列表时的返回格式。
		- `SessionDeleteResponse`：定义删除会话后的返回格式。
> 之前数据库设计的时候设计了消息相关的数据结构啊，为什么这里又有如MessageInfo相关的数据结构呢？
> 之前的数据库数据结构（SQLAlchemy ORM）负责数据持久化存储，而这里的 Chat 数据结构（Pydantic Schema）负责API 请求校验、响应格式定义和数据序列化，两者分别服务于存储层和接口层。
### 1.3.1 RAG数据结构
- 这里的四个数据结构都是为了RAG Trace 的记录、校验和规范化服务的，不直接参与检索计算。
#### 1）检索结果Chunk数据结构

| 字段             | 类型                   | 作用           |
| -------------- | -------------------- | ------------ |
| `filename`     | `str`                | 来源文件名，必填     |
| `page_number`  | `Optional[str / int] | 来源页码         |
| `text`         | `Optional[str]`      | 证据文本         |
| `score`        | `Optional[float]`    | 原始检索分数       |
| `rrf_rank`     | `Optional[int]`      | 融合检索结果中的排序位置 |
| `rerank_score` | `Optional[float]`    | Rerank 相关性分数 |
#### 2）RAG Trace数据结构

| 字段                               | 类型                                       | 作用                          |
| -------------------------------- | ---------------------------------------- | --------------------------- |
| 工具及查询                            |                                          |                             |
| `tool_used`                      | `Optional[bool]`                         | 是否使用了知识库工具                  |
| `tool_name`                      | `Optional[str]`                          | 工具名称                        |
| `query`                          | `Optional[str]`                          | 检索 Query                    |
| 查询重写                             |                                          |                             |
| `rewrite_method`                 | `Optional[Literal["step_back", "hyde"]]` | 查询重写方法                      |
| `rewritten_query`                | `Optional[str]`                          | 重写后的 Query                  |
| `step_back_question`             | `Optional[str]`                          | Step-back 生成的问题             |
| `hyde_document`                  | `Optional[str]`                          | HyDE 生成的假设文档                |
| `retrieval_stage`                | `Optional[str]`                          | 当前检索阶段                      |
| 证据质量判断                           |                                          |                             |
| `route`                          | `Optional[str]`                          | RAG 最终路由                    |
| `retrieval_status`               | `Optional[str]`                          | 证据检索状态                      |
| `evidence_relevance`             | `Optional[str]`                          | 证据相关性                       |
| `evidence_answerability`         | `Optional[str]`                          | 证据可回答性                      |
| `evidence_ambiguity`             | `Optional[str]`                          | 问题歧义情况                      |
| `evidence_confidence`            | `Optional[float]`                        | 证据评分置信度                     |
| `evidence_reason`                | `Optional[str]`                          | 证据评分原因                      |
| HITL                             |                                          |                             |
| `missing_slots`                  | `Optional[List[str]]`                    | 缺失的关键条件                     |
| `hitl_prompt`                    | `Optional[str]`                          | 向用户提出的澄清问题                  |
| `hitl_options`                   | `Optional[List[str]]`                    | 用户可选的澄清选项                   |
| `hitl_resumed`                   | `Optional[bool]`                         | 是否执行了 HITL 恢复               |
| `hitl_answer`                    | `Optional[str]`                          | 用户补充的信息                     |
| `hitl_resume_strategy`           | `Optional[str]`                          | HITL 恢复策略                   |
| `hitl_resume_from_status`        | `Optional[str]`                          | 恢复前的检索状态                    |
| `hitl_resume_from_route`         | `Optional[str]`                          | 恢复前的路由                      |
| `hitl_targeted_retrieved_chunks` | `Optional[List[RetrievedChunk]]`         | HITL 恢复时检索到的证据              |
| Rerank                           |                                          |                             |
| `rerank_enabled`                 | `Optional[bool]`                         | 是否配置启用 Rerank               |
| `rerank_applied`                 | `Optional[bool]`                         | 本次是否尝试执行 Rerank             |
| `rerank_model`                   | `Optional[str]`                          | Rerank 模型                   |
| `rerank_endpoint`                | `Optional[str]`                          | Rerank API 地址               |
| `rerank_error`                   | `Optional[str]`                          | Rerank 错误信息                 |
| `rerank_timeout_seconds`         | `Optional[float]`                        | Rerank 超时时间                 |
| `rerank_min_score`               | `Optional[float]`                        | 最低相关性阈值                     |
| `post_rerank_count`              | `Optional[int]`                          | Rerank 后的候选数量               |
| `post_threshold_count`           | `Optional[int]`                          | 阈值过滤后的候选数量                  |
| `retrieval_empty`                | `Optional[bool]`                         | 最终检索结果是否为空                  |
| 检索配置及过程                          |                                          |                             |
| `retrieval_mode`                 | `Optional[str]`                          | Hybrid、Dense Fallback 等检索模式 |
| `retrieval_pipeline`             | `Optional[str]`                          | 检索处理流程标识                    |
| `candidate_k`                    | `Optional[int]`                          | 初始召回候选数量                    |
| `candidate_k_source`             | `Optional[str]`                          | 候选数量的配置来源                   |
| `candidate_k_config_error`       | `Optional[str]`                          | 候选数量配置错误                    |
| `retrieval_candidate_multiplier` | `Optional[int]`                          | 候选数量扩展倍数                    |
| `retrieval_top_k`                | `Optional[int]`                          | 最终检索 TopK                   |
| `recall_count`                   | `Optional[int]`                          | 初始召回结果数量                    |
| `post_merge_candidate_count`     | `Optional[int]`                          | Auto-Merging 后的候选数量         |
| `candidate_count`                | `Optional[int]`                          | 进入精排的候选数量                   |
| `leaf_retrieve_level`            | `Optional[int]`                          | 初始检索的 Chunk 层级              |
| Auto-Merging                     |                                          |                             |
| `auto_merge_enabled`             | `Optional[bool]`                         | 是否启用 Auto-Merging           |
| `auto_merge_applied`             | `Optional[bool]`                         | 本次是否发生合并                    |
| `auto_merge_threshold`           | `Optional[int]`                          | 触发父块合并的子块数量阈值               |
| `auto_merge_replaced_chunks`     | `Optional[int]`                          | 合并时被替换的子块数量                 |
| `auto_merge_steps`               | `Optional[int]`                          | 实际发生合并的层级步骤数                |
| 检索证据                             |                                          |                             |
| `retrieved_chunks`               | `Optional[List[RetrievedChunk]]`         | 最终记录的检索证据                   |
| `initial_retrieved_chunks`       | `Optional[List[RetrievedChunk]]`         | 初始检索证据                      |
| `rewrite_retrieved_chunks`       | `Optional[List[RetrievedChunk]]`         | 重写后检索证据                     |
| 复杂问题处理                           |                                          |                             |
| `complexity`                     | `Optional[str]`                          | 问题复杂度                       |
| `complexity_reason`              | `Optional[str]`                          | 复杂度判断原因                     |
| `sub_questions`                  | `Optional[List[str]]`                    | 拆解出的子问题                     |
| `sub_agent_count`                | `Optional[int]`                          | 子 Agent 数量                  |
| `synthesis_merged_count`         | `Optional[int]`                          | Synthesis 合并后的证据数量          |
#### 3）RagSubTrace
- 继承 `RagTraceFields`

#### 4）RagTrace
- 在公共字段基础上增加 `sub_traces`


### 1.3.2 HITL数据结构
#### 1）HitlResumeState
- 保存 RAG 恢复快照，供用户补充信息后恢复检索。

|字段|类型|作用|
|---|---|---|
|`question`|`str`|原始问题，不能为空|
|`route`|`Literal["clarify", "scope_select"]`|触发 HITL 的路由|
|`retrieval_status`|`Literal["needs_clarification", "needs_scope_selection"]`|恢复前的检索状态|
|`rewrite_count`|`int`|已执行的 Rewrite 次数，默认 `0`，不得为负数|
|`complexity`|`Optional[Literal["simple", "complex"]]`|原问题复杂度，默认 `None`|
|`complexity_reason`|`Optional[str]`|复杂度判断原因，默认 `None`|
|`sub_questions`|`List[str]`|原问题拆分出的子问题，默认空列表，最多 4 个|

#### 2）PendingHitlState

记录跨轮待处理的 HITL 任务，保存用户需要补充的信息和 RAG 恢复快照。

|字段|类型|作用|
|---|---|---|
|`id`|`str`|HITL 任务 ID，不能为空|
|`original_question`|`str`|最初的用户问题，不能为空|
|`prompt`|`str`|向用户提出的澄清问题，不能为空|
|`options`|`List[str]`|可供用户选择的选项，默认空列表|
|`route`|`Literal["clarify", "scope_select"]`|HITL 类型|
|`retrieval_status`|`Literal["needs_clarification", "needs_scope_selection"]`|当前待澄清的检索状态|
|`answers`|`List[str]`|用户之前已补充的信息，默认空列表|
|`resume_state`|`HitlResumeState`|RAG 恢复快照|
|`created_at`|`str`|HITL 状态创建时间|

### 1.3.3 chat 数据结构

#### 1）ChatRequestContext
- 管理单次 Query 的临时运行状态，包括当前用户与会话、Knowledge Tool 调用额度、RAG Trace、HITL 恢复信息和流式输出事件。

| 字段                           | 类型                                    | 作用                             |
| ---------------------------- | ------------------------------------- | ------------------------------ |
| `user_id`                    | `str`                                 | 当前用户 ID                        |
| `session_id`                 | `str`                                 | 当前会话 ID                        |
| `output_queue`               | `Optional[asyncio.Queue]`             | 流式输出事件队列，默认 `None`             |
| `loop`                       | `Optional[asyncio.AbstractEventLoop]` | 流式输出所使用的事件循环，默认 `None`         |
| `_lock`                      | `threading.RLock`                     | 保证状态读写的线程安全                    |
| `_active`                    | `bool`                                | 请求是否有效，默认 `True`               |
| `_rag_trace`                 | `Optional[dict]`                      | 暂存规范化 RAG Trace 和 HITL 恢复状态    |
| `_knowledge_tool_slots_used` | `int`                                 | 本轮 Knowledge Tool 已调用次数，默认 `0` |
| `_started_at`                | `float`                               | 本轮上下文创建时的单调时钟时间                |
| `_last_step_at`              | `Optional[float]`                     | 上一次 RAG 进度事件的时间，默认 `None`      |
#### 2）ChatRequest
- 定义用户调用 Chat API 时提交的请求数据，包括用户消息和会话 ID。

|字段|类型|作用|
|---|---|---|
|`message`|`str`|用户输入的 Query|
|`session_id`|`Optional[str]`|会话 ID，默认 `"default_session"`|

#### 3）ChatResponse
- 定义 Chat API 返回的响应数据，包括 Agent 生成的回答及本轮 RAG Trace。

|字段|类型|作用|
|---|---|---|
|`response`|`str`|Agent 生成的回答|
|`rag_trace`|`Optional[RagTrace]`|本轮 RAG 执行轨迹，默认 `None`|

#### 4）MessageInfo
- 定义单条历史聊天消息的数据格式，包括消息类型、内容、时间以及关联的 RAG Trace。

|字段|类型|作用|
|---|---|---|
|`type`|`str`|消息类型|
|`content`|`str`|消息内容|
|`timestamp`|`str`|消息时间|
|`rag_trace`|`Optional[RagTrace]`|消息关联的 RAG Trace，默认 `None`|

#### 5）SessionMessagesResponse
- 定义查询某个会话历史消息时的返回格式，包含多条 MessageInfo。

|字段|类型|作用|
|---|---|---|
|`messages`|`List[MessageInfo]`|历史聊天消息列表|

#### 6）SessionInfo
- 定义单个会话的基本信息，包括会话 ID、标题、更新时间和消息数量。

|字段|类型|作用|
|---|---|---|
|`session_id`|`str`|会话 ID|
|`title`|`Optional[str]`|会话标题，默认 `None`|
|`updated_at`|`str`|最后更新时间|
|`message_count`|`int`|会话消息数量|

#### 7）SessionListResponse
- 定义查询用户会话列表时的返回格式，包含多个 `SessionInfo`。

|字段|类型|作用|
|---|---|---|
|`sessions`|`List[SessionInfo]`|用户会话信息列表|

#### 8）SessionDeleteResponse
- 定义删除会话操作的返回格式，包括被删除的会话 ID 和操作结果信息。

|字段|类型|作用|
|---|---|---|
|`session_id`|`str`|被删除的会话 ID|
|`message`|`str`|删除操作的结果信息|


## 1.4 文件结构设计
```python
backend/
├── schemas/
│   └── chat.py                 # 对话/RAG Trace/HITL 公共数据结构及规范化
│
└── chat/
    ├── request_context.py     # 单次 Query 状态、额度控制与流式事件
    ├── storage.py             # 会话与消息的 PostgreSQL / Redis 存取
    └── runtime.py             # LLM 初始化、System Prompt、Agent 创建与工具注册
```

# 2. 模块实现
## 2.1 数据结构实现
本模块需要定义三类数据结构：
1. RAG 数据结构：规范检索证据及主问题、子问题的执行轨迹。
2. HITL 数据结构：保存 RAG 恢复快照及跨轮待澄清状态。
3. 对话请求上下文：管理单次 Query 的临时运行状态。
### 2.1.1 RAG 数据结构
文件：`backend/schemas/chat.py`
```python
# 定义公共基类，禁止传入未定义的字段。
class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

# 检索出来的evidence
class RetrievedChunk(StrictSchema):
    filename: str                             # 来源文件名，必填
    page_number: Optional[str | int] = None  # 来源页码
    text: Optional[str] = None                # Evidence 文本
    score: Optional[float] = None             # 原始检索分数
    rrf_rank: Optional[int] = None            # 融合检索后的排序位置
    rerank_score: Optional[float] = None      # Rerank 相关性分数

# 记录检索路径
class RagTraceFields(StrictSchema):
    # 1. 工具及查询信息
    tool_used: Optional[bool] = None          # 是否使用知识库工具
    tool_name: Optional[str] = None           # 工具名称
    query: Optional[str] = None               # 检索 Query

    # 2. Query Rewrite
    rewrite_method: Optional[Literal["step_back", "hyde"]] = None  # 重写方式
    rewritten_query: Optional[str] = None     # 重写后的 Query
    step_back_question: Optional[str] = None  # Step-back 生成的问题
    hyde_document: Optional[str] = None       # HyDE 生成的假设文档
    retrieval_stage: Optional[str] = None     # 当前检索阶段

    # 3. Evidence Grading
    route: Optional[str] = None               # RAG 最终路由
    retrieval_status: Optional[str] = None    # 证据检索状态
    evidence_relevance: Optional[str] = None  # 证据相关性
    evidence_answerability: Optional[str] = None  # 证据可回答性
    evidence_ambiguity: Optional[str] = None  # 问题歧义情况
    evidence_confidence: Optional[float] = None   # 评分置信度
    evidence_reason: Optional[str] = None     # 证据评分原因

    # 4. HITL 相关执行记录
    missing_slots: Optional[List[str]] = None  # 缺失的关键条件
    hitl_prompt: Optional[str] = None          # 向用户提出的澄清问题
    hitl_options: Optional[List[str]] = None   # 用户可选择的选项
    hitl_resumed: Optional[bool] = None        # 是否执行过 HITL 恢复
    hitl_answer: Optional[str] = None          # 用户补充的信息
    hitl_resume_strategy: Optional[str] = None # HITL 恢复策略
    hitl_resume_from_status: Optional[str] = None # 恢复前的检索状态
    hitl_resume_from_route: Optional[str] = None  # 恢复前的路由
    hitl_targeted_retrieved_chunks: Optional[List[RetrievedChunk]] = None  # 恢复时检索的证据

    # 5. Rerank
    rerank_enabled: Optional[bool] = None     # 是否配置启用 Rerank
    rerank_applied: Optional[bool] = None     # 本次是否尝试执行 Rerank
    rerank_model: Optional[str] = None        # Rerank 模型名称
    rerank_endpoint: Optional[str] = None     # Rerank API 地址
    rerank_error: Optional[str] = None        # Rerank 错误信息
    rerank_timeout_seconds: Optional[float] = None  # Rerank 超时时间
    rerank_min_score: Optional[float] = None  # 最低相关度阈值
    post_rerank_count: Optional[int] = None   # Rerank 后的候选数量
    post_threshold_count: Optional[int] = None  # 阈值过滤后的数量
    retrieval_empty: Optional[bool] = None   # 最终检索结果是否为空

    # 6. Retrieval
    retrieval_mode: Optional[str] = None      # Hybrid / Dense Fallback 等模式
    retrieval_pipeline: Optional[str] = None  # 检索流程标识
    candidate_k: Optional[int] = None         # 初始召回候选数量
    candidate_k_source: Optional[str] = None  # 候选数量配置来源
    candidate_k_config_error: Optional[str] = None  # 候选数量配置错误
    retrieval_candidate_multiplier: Optional[int] = None  # 召回候选扩展倍数
    retrieval_top_k: Optional[int] = None     # 最终检索 TopK
    recall_count: Optional[int] = None        # 初始召回数量
    post_merge_candidate_count: Optional[int] = None  # Auto-Merging 后的候选数量
    candidate_count: Optional[int] = None     # 进入精排的候选数量
    leaf_retrieve_level: Optional[int] = None # 初始检索的 Chunk 层级

    # 7. Auto-Merging
    auto_merge_enabled: Optional[bool] = None  # 是否启用 Auto-Merging
    auto_merge_applied: Optional[bool] = None  # 本次是否发生合并
    auto_merge_threshold: Optional[int] = None # 父块合并阈值
    auto_merge_replaced_chunks: Optional[int] = None  # 被替换的子块数量
    auto_merge_steps: Optional[int] = None    # 实际发生合并的层级步骤数

    # 8. 检索证据
    retrieved_chunks: Optional[List[RetrievedChunk]] = None  # 最终记录的 Evidence
    initial_retrieved_chunks: Optional[List[RetrievedChunk]] = None  # 初次检索的 Evidence
    rewrite_retrieved_chunks: Optional[List[RetrievedChunk]] = None  # Rewrite 后的 Evidence

    # 9. 复杂问题处理
    complexity: Optional[str] = None          # 问题复杂度
    complexity_reason: Optional[str] = None   # 复杂度判断原因
    sub_questions: Optional[List[str]] = None # 拆解出的子问题
    sub_agent_count: Optional[int] = None     # 子 Agent 数量
    synthesis_merged_count: Optional[int] = None  # Synthesis 合并后的证据数量

# 子问题的rag trace, 直接继承
class RagSubTrace(RagTraceFields):
    pass

# 父问题的rag trace
class RagTrace(RagTraceFields):
    # 额外保存多个子问题的执行轨迹
    sub_traces: Optional[List[RagSubTrace]] = None
```
- 再提供几个RAG Trace 字段规范化函数
```python
### 4 个 RAG Trace 规范化函数
# 过滤无效 Chunk，只保留 RetrievedChunk 定义的字段，并进行数据校验
def _normalize_chunks(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    # 获取 RetrievedChunk 中声明的字段
    fields = RetrievedChunk.model_fields
    return [
        RetrievedChunk.model_validate(
            # 只保留 RetrievedChunk 定义的字段
            {key: item[key] for key in fields if key in item}
        ).model_dump(exclude_none=True)  # 校验后转回字典，去除 None 字段
        for item in value
        if isinstance(item, dict) and item.get("filename") # 只处理具有有效 filename 的字典
    ]

# 过滤 Trace 中未定义的字段
def _normalize_trace_fields(trace: dict, fields: dict) -> dict:
    # 只保留指定 Schema 定义的字段
    normalized = {key: trace[key] for key in fields if key in trace}
    for key in (  # 找出 Trace 中的四类检索证据
        "retrieved_chunks",
        "initial_retrieved_chunks",
        "rewrite_retrieved_chunks",
        "hitl_targeted_retrieved_chunks",
    ):
        if key in normalized:# 如果存在，则调进行规范化
            normalized[key] = _normalize_chunks(normalized[key])
    return normalized

# 将子问题 Trace 转换为符合 RagSubTrace 的字典
def normalize_rag_sub_trace(trace: dict | None) -> Optional[dict]:
    # 输入空字典，返回 None
    if not isinstance(trace, dict) or not trace:
        return None
    # 根据 RagSubTrace 定义的字段进行过滤和 Chunk 规范化
    normalized = _normalize_trace_fields(trace, RagSubTrace.model_fields)
    # 使用 Pydantic 校验，并转回普通字典
    return RagSubTrace.model_validate(normalized).model_dump(exclude_none=True)

# 将主问题 Trace 转换为符合 RagTrace 的字典，同时规范化所有子问题 Trace
def normalize_rag_trace(trace: dict | None) -> Optional[dict]:
    if not isinstance(trace, dict) or not trace:
        return None
    normalized = _normalize_trace_fields(trace, RagTrace.model_fields)
    if "sub_traces" in normalized:
        sub_traces = normalized["sub_traces"] if isinstance(normalized["sub_traces"], list) else []
        normalized["sub_traces"] = [
            item
            for item in (
                normalize_rag_sub_trace(sub_trace)
                for sub_trace in sub_traces
                if isinstance(sub_trace, dict)
            )
            if item is not None
        ]
    return RagTrace.model_validate(normalized).model_dump(exclude_none=True)
```
### 2.1.2 HITL数据结构
文件：`backend/schemas/chat.py`
```python
# 保存RAG Pipeline 的恢复快照，供用户补充信息后继续检索。
class HitlResumeState(StrictSchema):
    question: str = Field(min_length=1)   # 原始问题，不允许为空字符串
    route: Literal["clarify", "scope_select"]  # 触发 HITL 的路由
    retrieval_status: Literal[    # HITL 前的检索状态
        "needs_clarification",
        "needs_scope_selection"
    ]
    rewrite_count: int = Field(default=0, ge=0)    # 已执行的 Query Rewrite 次数
    complexity: Optional[Literal["simple", "complex"]] = None    # 原问题复杂度
    complexity_reason: Optional[str] = None    # 原问题复杂度判断的原因
    sub_questions: List[str] = Field(    # 原问题拆分出的子问题
        default_factory=list,
        max_length=4
    )

# 保存跨轮待处理的 HITL 任务，供下一轮 Chat Service 识别并恢复
class PendingHitlState(StrictSchema):
    id: str = Field(min_length=1)    # 当前待处理 HITL 任务的唯一 ID
    original_question: str = Field(min_length=1)    # 最初触发 HITL 的用户问题
    prompt: str = Field(min_length=1)    # 需要向用户提出的澄清问题
    options: List[str] = Field(default_factory=list)    # 可供用户选择的候选选项，默认空列表
    route: Literal["clarify", "scope_select"]    # HITL 类型：缺失条件 / 多候选范围
    retrieval_status: Literal[    # 当前待恢复的检索状态
        "needs_clarification",
        "needs_scope_selection"
    ]
    answers: List[str] = Field(default_factory=list)    # 用户之前已经补充的信息，默认空列表
    resume_state: HitlResumeState    # 保存第四模块生成的 RAG 恢复快照
    created_at: str    # HITL 任务创建时间
```

### 2.1.3 chat数据结构
文件：`backend/schemas/chat.py`
```python
# Chat API 请求数据结构
class ChatRequest(StrictSchema):
    message: str                              # 用户输入的 Query
    session_id: Optional[str] = "default_session"  # 会话 ID，默认使用 default_session


# Chat API 响应数据结构
class ChatResponse(StrictSchema):
    response: str                             # Agent 生成的回答
    rag_trace: Optional[RagTrace] = None      # 本轮 RAG 执行轨迹，默认 None


# 单条历史聊天消息的数据结构
class MessageInfo(StrictSchema):
    type: str                                 # 消息类型，如 human、ai
    content: str                              # 消息内容
    timestamp: str                            # 消息时间
    rag_trace: Optional[RagTrace] = None      # 该消息关联的 RAG 执行轨迹


# 会话历史消息查询的响应数据结构
class SessionMessagesResponse(StrictSchema):
    messages: List[MessageInfo]               # 当前会话的历史消息列表


# 单个会话的基本信息
class SessionInfo(StrictSchema):
    session_id: str                           # 会话唯一标识
    title: Optional[str] = None               # 会话标题，默认 None
    updated_at: str                           # 会话最后更新时间
    message_count: int                        # 会话中的消息数量


# 会话列表查询的响应数据结构
class SessionListResponse(StrictSchema):
    sessions: List[SessionInfo]               # 用户的会话信息列表


# 删除会话接口的响应数据结构
class SessionDeleteResponse(StrictSchema):
    session_id: str                           # 被删除的会话 ID
    message: str                              # 删除操作的结果信息
```
#### 对话上下文数据结构
文件：`backend/chat/request_context.py`
- 前面定义的都是数据格式，这里定义的是一个运行时管理逻辑。之前的 Schema 类主要用于定义、校验和传递数据，而 `ChatRequestContext` 是包含属性和操作方法的运行状态管理类，负责动态管理单次 Query 执行过程中不断变化的临时状态。
##### 1）属性设计

| 属性                           | 作用                                    |
| ---------------------------- | ------------------------------------- |
| `user_id`                    | 标识当前请求所属用户                            |
| `session_id`                 | 标识当前请求所属会话                            |
| `output_queue`               | 后端 RAG 与 Chat Service 之间传递流式进度事件的异步队列 |
| `loop`                       | 保存当前事件循环，支持 RAG 工作线程安全地向队列投递事件        |
| `_lock`                      | 可重入线程锁，保护多个线程对请求状态的并发读写               |
| `_active`                    | 标记本轮请求是否仍然有效，防止请求结束后继续写入状态            |
| `_rag_trace`                 | 暂存本轮 RAG Trace 及可能存在的 HITL 恢复状态       |
| `_knowledge_tool_slots_used` | 记录本轮 Knowledge Tool 已占用的调用额度          |
| `_started_at`                | 记录上下文创建时间，用于计算本轮累计耗时                  |
| `_last_step_at`              | 记录上一次 RAG 进度事件的时间，用于计算事件间隔耗时          |

##### 2）方法设计

| 方法                              | 参数                                            | 作用                                 |
| ------------------------------- | --------------------------------------------- | ---------------------------------- |
| `for_stream()`                  | `user_id`、`session_id`、`output_queue`         | 创建流式请求上下文，自动绑定当前 Event Loop        |
| `for_sync()`                    | `user_id`、`session_id`                        | 创建非流式请求上下文，不需要消息队列和 Event Loop     |
| `emit_rag_step()`               | `icon`、`label`、`detail`、`group`、`group_label` | 将 RAG 执行进度和耗时信息写入异步队列，供前端实时展示      |
| `store_rag_trace()`             | `rag_trace`、`hitl_resume_state`               | 校验并暂存本轮 RAG Trace，以及可选的 HITL 恢复状态  |
| `take_rag_trace()`              | 无                                             | 取出并清空暂存的 RAG 结果，供 Chat Service 处理  |
| `peek_rag_trace()`              | 无                                             | 读取 RAG 结果但不清空，用于检查执行状态和 HITL       |
| `reset_knowledge_tool_budget()` | 无                                             | 清零 Knowledge Tool 调用计数             |
| `acquire_knowledge_tool_slot()` | 无                                             | 申请工具调用额度，每轮最多成功申请一次                |
| `close()`                       | 无                                             | 标记请求结束，停止后续进度发送和 Trace 写入，解除流式资源引用 |

##### 3）生命周期设计

```
用户发送 Query
      ↓
创建 ChatRequestContext
├── for_sync()       # 非流式
└── for_stream()     # 流式
      ↓
Agent / Tool / RAG 执行
├── acquire_knowledge_tool_slot()  # 控制工具调用
├── emit_rag_step()                # 发送执行进度
└── store_rag_trace()              # 暂存执行结果
      ↓
Chat Service 处理结果
├── peek_rag_trace()   # 查看状态
└── take_rag_trace()   # 取走结果
      ↓
持久化消息及必要的 HITL 状态
      ↓
close()                # 结束本轮 ctx
```

设计原则： 每次 Query 独立创建一个 `ChatRequestContext`；其属性只服务于当前请求，跨轮聊天记录和 HITL 状态由 Chat Service 通过 Storage 持久化，不依赖复用 `ctx` 对象。
```python
@dataclass
class ChatRequestContext:
    """
    管理单次 Query 执行期间的临时运行状态
    Request-owned state shared explicitly across agent tools and RAG nodes.
    """

    user_id: str    #用户id
    session_id: str    #绘画id
    # 后端 RAG 和 Chat Service 之间传递实时进度的临时队列
    output_queue: Optional[asyncio.Queue] = None 
    loop: Optional[asyncio.AbstractEventLoop] = None  # asyncio 事件循环，用于安全地向流式队列投递消息

    # 内部运行状态，会随着本轮 Query 的执行动态变化。
    _lock: threading.RLock = field(default_factory=threading.RLock) # 可重入线程锁，保护多个线程对共享状态的读写
    _active: bool = True  # 请求是否仍处于有效状态，标记本轮 Query 是否仍在执行
    _rag_trace: Optional[dict] = None  # 暂存本轮 RAG Trace，以及可能存在的 HITL 恢复状态
    _knowledge_tool_slots_used: int = 0  # 本轮 Knowledge Tool 调用次数
    _started_at: float = field(default_factory=time.monotonic)  # 记录本轮开始时间
    _last_step_at: Optional[float] = None  # 上一次发送 RAG 进度事件的时间，用于计算阶段耗时

    @classmethod # 初始化类，流式输出
    def for_stream(
        cls,
        *,
        user_id: str,
        session_id: str,
        output_queue: asyncio.Queue,
    ) -> ChatRequestContext:
        return cls(
            user_id=user_id,
            session_id=session_id,
            output_queue=output_queue,
            loop=asyncio.get_running_loop(),
        )

    @classmethod
    def for_sync( # 初始化类，非流式输出
        cls,
        *,
        user_id: str,
        session_id: str,
    ) -> ChatRequestContext:
        return cls(user_id=user_id, session_id=session_id)

    def emit_rag_step( # 发送rag执行进度
        self,
        icon: str,
        label: str,
        detail: str = "",
        *,
        group: Optional[str] = None,
        group_label: Optional[str] = None,
    ) -> None:
        '''
        当 RAG Pipeline 执行检索、Rerank、证据评分等操作时，
        将执行进度写入 output_queue，供 Chat Service 通过 SSE 实时推送给前端。
        '''
        with self._lock: # 1. 获取线程锁，保护 ctx 的共享状态
            if not self._active: # 请求已经结束，不再发送进度
                return
            if self.output_queue is None or self.loop is None: # 非流式请求没有队列和 Event Loop，无需发送
                return
            now = time.monotonic()
            # 如果是第一次发送进度，使用请求开始时间, 否则使用上一次发送进度的时间
            last_step_at = self._last_step_at or self._started_at
            elapsed_ms = max(int((now - self._started_at) * 1000), 0) # 从请求开始到现在的累计耗时，单位毫秒
            stage_elapsed_ms = max(int((now - last_step_at) * 1000), 0) # 距离上一次进度事件的耗时，单位毫秒
            self._last_step_at = now
            queue = self.output_queue
            loop = self.loop

        step = {
            "icon": icon, # 步骤图标
            "label": label,# 步骤名称
            "detail": detail,# 步骤描述
            "elapsed_ms": elapsed_ms,# 本轮累计耗时
            "stage_elapsed_ms": stage_elapsed_ms,# 距上次事件的耗时
        }
        if group: # 如果指定了分组，添加分组标识
            step["group"] = group
        if group_label:
            step["group_label"] = group_label

        try:# 3. 将事件投递到异步队列
            if not loop.is_closed():
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"type": "rag_step", "step": step},
                )
        except Exception:
            logger.exception("Failed to emit RAG step")

    def store_rag_trace(self, rag_trace: dict, hitl_resume_state: Optional[dict] = None) -> None:
        '''
        将 RAG Pipeline 返回的 rag_trace 进行规范化，
        并暂存到 ChatRequestContext；
        如果触发 HITL，还会保存用于后续恢复检索的 HitlResumeState。
        '''
        current_trace = normalize_rag_trace(rag_trace)
        if not current_trace:
            return
        with self._lock:
            if self._active:
                self._rag_trace = {"rag_trace": current_trace}
                if hitl_resume_state:
                    self._rag_trace["hitl_resume_state"] = HitlResumeState.model_validate(
                        hitl_resume_state
                    ).model_dump()

    def take_rag_trace(self) -> Optional[dict]:
        '''
        从 ChatRequestContext 中取出暂存的 RAG Trace 和 HITL 恢复状态，同时清空内部数据
        '''
        with self._lock:
            context = self._rag_trace
            self._rag_trace = None
            return context

    def peek_rag_trace(self) -> Optional[dict]:
        # 查看本轮 RAG 执行结果
        with self._lock:
            return self._rag_trace

    def reset_knowledge_tool_budget(self) -> None:
        # 重置知识库工具调用额度
        with self._lock:
            self._knowledge_tool_slots_used = 0

    def acquire_knowledge_tool_slot(self) -> bool:
        # 申请一次 Knowledge Tool 调用额度
        with self._lock:
            if self._knowledge_tool_slots_used >= 1:
                return False
            self._knowledge_tool_slots_used += 1
            return True

    def close(self) -> None:
        '''
        标记当前 Query 已结束，停止接收新的 RAG Trace 和进度事件，
        并解除流式输出相关资源的引用。
        '''
        with self._lock:
            self._active = False
            self.output_queue = None
            self.loop = None
```

## 2.2 会话存储管理
`backend/chat/storage.py`
遵照我们的流程设计，会话存储管理部分体现在如下节点：
- 加载当前会话数据：从 PostgreSQL / Redis 读取历史聊天记录、长期记忆摘要和待处理的 HITL 状态。
- 保存用户消息、AI 消息和会话信息：持久化聊天消息、RAG Trace、长期记忆摘要和待处理的 HITL 状态。
### 1）方法设计

| 方法                             | 参数                                                                | 作用                                                         |
| ------------------------------ | ----------------------------------------------------------------- | ---------------------------------------------------------- |
| `_messages_cache_key()`        | `user_id`、`session_id`                                            | 生成指定会话的 Redis 消息缓存 Key                                     |
| `_sessions_cache_key()`        | `user_id`                                                         | 生成用户会话列表的 Redis 缓存 Key                                     |
| `_to_langchain_messages()`     | `records`                                                         | 将数据库消息字典转换为 LangChain Message 对象，供 Agent 使用                |
| `_normalize_message_records()` | `records`                                                         | 规范化历史消息中的 RAG Trace，保留其他消息字段                               |
| `save()`                       | `user_id`、`session_id`、`messages`、`metadata`、`extra_message_data` | 将完整会话消息、RAG Trace 和元数据保存到 PostgreSQL，并更新 Redis 缓存          |
| `load()`                       | `user_id`、`session_id`                                            | 优先从 Redis 加载历史消息，未命中则查询 PostgreSQL，返回 LangChain Message 列表 |
| `load_with_meta()`             | `user_id`、`session_id`                                            | 加载历史消息及会话元数据，包括长期记忆摘要和待处理 HITL                             |
| `list_sessions()`              | `user_id`                                                         | 获取指定用户的所有会话 ID                                             |
| `list_session_infos()`         | `user_id`                                                         | 获取用户所有会话的摘要信息，包括标题、更新时间和消息数量                               |
| `get_session_messages()`       | `user_id`、`session_id`                                            | 获取指定会话的完整历史消息，包括内容、时间戳和 RAG Trace                          |
| `delete_session()`             | `user_id`、`session_id`                                            | 删除指定会话及关联消息，并清理 Redis 中对应的缓存                               |

### 2）存储设计

| 存储组件                                         | 数据                         | 作用                    |
| -------------------------------------------- | -------------------------- | --------------------- |
| PostgreSQL `ChatSession`                     | 会话 ID、更新时间、`metadata_json` | 持久化会话信息、长期记忆和待处理 HITL |
| PostgreSQL `ChatMessage`                     | 消息类型、内容、时间戳、RAG Trace      | 持久化完整聊天记录             |
| Redis `chat_messages:{user_id}:{session_id}` | 指定会话的消息列表                  | 加速历史消息读取              |
| Redis `chat_sessions:{user_id}`              | 用户所有会话的摘要列表                | 加速会话列表查询              |

> 当前实现注意： `save()` 采用全量覆盖策略，会先删除指定 Session 的旧消息，再写入完整消息列表；后续可以优化为增量追加???

```python
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.db.models import ChatMessage, ChatSession, User
from backend.infra.cache import cache
from backend.infra.database import SessionLocal
from backend.schemas.chat import normalize_rag_trace


class ConversationStorage:
    """对话存储（PostgreSQL + Redis）。"""

    @staticmethod
    def _messages_cache_key(user_id: str, session_id: str) -> str:
        # 生成聊天消息的 Redis 缓存 Key
        return f"chat_messages:{user_id}:{session_id}"

    @staticmethod
    def _sessions_cache_key(user_id: str) -> str:
        # 生成用户会话列表的 Redis 缓存 Key
        return f"chat_sessions:{user_id}"

    @staticmethod
    def _to_langchain_messages(records: list[dict]) -> list:
        # 将数据库消息转换为 LangChain 消息对象
        messages = []
        for msg_data in records:
            msg_type = msg_data.get("type")
            content = msg_data.get("content", "")
            if msg_type == "human":
                messages.append(HumanMessage(content=content))
            elif msg_type == "ai":
                messages.append(AIMessage(content=content))
            elif msg_type == "system":
                messages.append(SystemMessage(content=content))
        return messages

    @staticmethod
    def _normalize_message_records(records: list[dict]) -> list[dict]:
        # 规范化历史消息中的 RAG Trace
        normalized = []
        for record in records:
            current = dict(record)
            current["rag_trace"] = normalize_rag_trace(record.get("rag_trace"))
            normalized.append(current)
        return normalized

    def save(
        self,
        user_id: str,
        session_id: str,
        messages: list, # 当前会话的完整消息列表（LangChain Message）
        metadata: dict = None,
        extra_message_data: list = None,  # 每条消息附加的数据，如 RAG Trace
    ):
        '''
        将当前会话的消息列表、RAG Trace 和会话元数据保存到 PostgreSQL，同时更新 Redis 缓存。
        '''
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return

            session = (# 2. 查询当前用户的指定聊天会话
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:# 3. 会话不存在，则创建新会话
                session = ChatSession(user_id=user.id, session_id=session_id, metadata_json=metadata or {})
                db.add(session)
                db.flush()
            elif metadata is not None:# 会话已经存在，且传入了新的元数据
                existing_meta = session.metadata_json or {}# 合并原有元数据和新元数据
                session.metadata_json = {**existing_meta, **metadata}

            # 4. 删除当前会话的全部旧消息, 这里是全量删除，后续有待优化成增量增加？？？
            db.query(ChatMessage).filter(ChatMessage.session_ref_id == session.id).delete(synchronize_session=False)

            # 5. 重新写入当前会话的完整消息列表
            serialized = []
            now = datetime.utcnow()
            for idx, msg in enumerate(messages):
                rag_trace = None
                # 根据消息下标获取对应的附加数据
                if extra_message_data and idx < len(extra_message_data):
                    extra = extra_message_data[idx] or {}
                    rag_trace = normalize_rag_trace(extra.get("rag_trace"))

                db.add(
                    ChatMessage(
                        session_ref_id=session.id,
                        message_type=msg.type,
                        content=str(msg.content),
                        timestamp=now,
                        rag_trace=rag_trace,
                    )
                )
                serialized.append(
                    {
                        "type": msg.type,
                        "content": str(msg.content),
                        "timestamp": now.isoformat(),
                        "rag_trace": rag_trace,
                    }
                )

            session.updated_at = now
            db.commit()

            # 7. 将完整聊天记录写入 Redis 缓存
            cache.set_json(self._messages_cache_key(user_id, session_id), serialized)
            # 更新某个会话列表之后，这个会话对应的一些信息，比如updated_at、message_count 就可能变化
            # redis里面用了一个字典来存储用户的所有session的一些摘要信息
            # 更新某个session之后，整个摘要列表就要全量删除
            cache.delete(self._sessions_cache_key(user_id))
        finally:
            db.close()

    def load(self, user_id: str, session_id: str) -> list:
        # 加载指定会话的历史聊天记录
        # 优先从 Redis 缓存读取历史消息
        cached = cache.get_json(self._messages_cache_key(user_id, session_id))
        if cached is not None:
            return self._to_langchain_messages(cached)

        records = self.get_session_messages(user_id, session_id)
        cache.set_json(self._messages_cache_key(user_id, session_id), records)
        return self._to_langchain_messages(records)

    def load_with_meta(self, user_id: str, session_id: str) -> tuple[list, dict]:
        """加载对话消息及会话元数据（标题、持久化笔记等）。"""
        # 在load的基础上再查询一下metadata
        messages = self.load(user_id, session_id)
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return messages, {}
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return messages, {}
            return messages, dict(session.metadata_json or {})
        finally:
            db.close()

    def list_sessions(self, user_id: str) -> list: #获取用户的所有会话 ID
        return [item["session_id"] for item in self.list_session_infos(user_id)]

    def list_session_infos(self, user_id: str) -> list[dict]:
        cached = cache.get_json(self._sessions_cache_key(user_id))
        if cached is not None:
            return cached

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return []

            sessions = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id)
                .order_by(ChatSession.updated_at.desc())
                .all()
            )
            result = []
            for s in sessions:
                count = db.query(ChatMessage).filter(ChatMessage.session_ref_id == s.id).count()
                meta = s.metadata_json or {}
                result.append(
                    {
                        "session_id": s.session_id,
                        "title": meta.get("title") or s.session_id,
                        "updated_at": s.updated_at.isoformat(),
                        "message_count": count,
                    }
                )
            cache.set_json(self._sessions_cache_key(user_id), result)
            return result
        finally:
            db.close()

    def get_session_messages(self, user_id: str, session_id: str) -> list[dict]:
        # 责获取包含 RAG Trace、时间戳的完整历史消息字典，供前端展示
        cached = cache.get_json(self._messages_cache_key(user_id, session_id))
        if cached is not None:
            normalized = self._normalize_message_records(cached)
            if normalized != cached:
                cache.set_json(self._messages_cache_key(user_id, session_id), normalized)
            return normalized

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return []
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return []

            rows = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_ref_id == session.id)
                .order_by(ChatMessage.id.asc())
                .all()
            )
            result = [
                {
                    "type": row.message_type,
                    "content": row.content,
                    "timestamp": row.timestamp.isoformat(),
                    "rag_trace": normalize_rag_trace(row.rag_trace),
                }
                for row in rows
            ]
            cache.set_json(self._messages_cache_key(user_id, session_id), result)
            return result
        finally:
            db.close()

    def delete_session(self, user_id: str, session_id: str) -> bool:
        # 删除指定会话及其聊天记录
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return False
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return False

            db.delete(session)
            db.commit()
            cache.delete(self._messages_cache_key(user_id, session_id))
            cache.delete(self._sessions_cache_key(user_id))
            return True
        finally:
            db.close()

storage = ConversationStorage()
```
## 2.3 Agent Runtime
`backend/chat/runtime.py`
初始化 LLM，配置 System Prompt，创建绑定 Knowledge Tool 和 Weather Tool 的 Agent。
```python
import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from backend.chat.request_context import ChatRequestContext
from backend.tools import get_current_weather, make_search_knowledge_base

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
FAST_MODEL = os.getenv("FAST_MODEL")
BASE_URL = os.getenv("BASE_URL")

SYSTEM_PROMPT = (
    "You are a cute cat bot that loves to help users. "
    "When responding, you may use tools to assist. "
    "Use search_knowledge_base when users ask document/knowledge questions. "
    "Do not call the same tool repeatedly in one turn. At most one knowledge tool call per turn. "
    "Once you call search_knowledge_base and receive its result, you MUST immediately produce the Final Answer based on that result. "
    "After receiving search_knowledge_base result, you MUST NOT call any tool again (including get_current_weather or search_knowledge_base). "
    "If the tool result starts with NEEDS_CLARIFICATION or NEEDS_SCOPE_SELECTION, ask the user the requested question directly and do not answer from retrieved context. "
    "If the tool result starts with NO_KNOWLEDGE, say the knowledge base does not contain reliable relevant information. "
    "If the retrieved context is insufficient, answer honestly that you don't know instead of making up facts. "
    "When answering based on retrieved chunks, you MUST cite the source chunks using their index numbers inline, for example [1] or [2][3]. "
    "Step-back questions and HyDE documents are retrieval aids only, not factual evidence. "
    "Base factual claims only on retrieved source chunks and do not reveal chain-of-thought. "
    "If you don't know the answer, admit it honestly."
)


model = init_chat_model(
    model=MODEL,
    model_provider="openai",
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.3,
    stream_usage=True,
)

fast_model = init_chat_model(
    model=FAST_MODEL,
    model_provider="openai",
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.2,
    stream_usage=True,
)


def create_agent_for_request(ctx: ChatRequestContext):
    return create_agent(
        model=model,
        tools=[
            get_current_weather,
            make_search_knowledge_base(ctx),
        ],
        system_prompt=SYSTEM_PROMPT,
    )
```