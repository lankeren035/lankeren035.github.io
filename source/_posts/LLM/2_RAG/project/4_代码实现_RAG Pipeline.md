---
title: 4_代码实现_RAG Pipeline
date: 2026-10-07
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
- 当前：实现 **RAG Pipeline**，负责根据用户问题组织完整的 RAG 流程：

```
用户问题
↓
问题复杂度判断
↓
简单问题 → 检索 → Evidence 评分 → 必要时 Query Rewrite → 再检索
复杂问题 → 子问题拆分 → 并行检索 → Evidence 合并
↓
最终 Evidence
```

它负责 **RAG 的流程编排和决策**，不负责最终 Answer 生成。

---

## 1.2 模块整体流程设计

```
                    用户问题
                       ↓
                 判断问题复杂度
                ↙             ↘
             简单问题          复杂问题
                ↓               ↓
             知识库检索        子问题拆分
                ↓               ↓
             证据质量判断      并行检索子问题
                ↓               ↓
      ┌─────────┼─────────┐   合并各子问题证据
      ↓         ↓         ↓          ↓
   证据足够   证据不足   问题有歧义    最终 Evidence
      ↓         ↓         ↓
   直接返回   Query 重写   用户补充信息
                ↓
             重新检索
                ↓
             再次判断证据
```
---

## 1.3 模块数据结构设计
在langgraph中，需要有一个RAG State作为全局变量，来控制各个节点的跳转，根据前面定义的节点和流程，设计如下字段：

| 节点/阶段        | State 字段                 | 作用                 |
| ------------ | ------------------------ | ------------------ |
| 初始输入         | `question`               | 当前需要处理的问题          |
|              | `query`                  | 当前实际用于检索的 Query    |
| `复杂度判断节点`    | `complexity`             | `simple / complex` |
|              | `complexity_reason`      | 复杂度判断原因            |
|              | `sub_questions`          | 复杂问题拆分出的子问题        |
| `知识库检索节点`    | `docs`                   | 检索得到的 Chunk        |
|              | `context`                | 格式化后的 Evidence 文本  |
| `证据质量判断节点`   | `evidence_relevance`     | Evidence 相关性       |
|              | `evidence_answerability` | Evidence 是否足够回答问题  |
|              | `evidence_ambiguity`     | 是否存在缺失条件或多候选歧义     |
|              | `evidence_confidence`    | 评分置信度              |
|              | `retrieval_status`       | 当前检索结果状态           |
|              | `route`                  | 后续处理路由             |
| HITL         | `missing_slots`          | 用户问题缺失的条件          |
|              | `hitl_prompt`            | 向用户提出的补充问题         |
|              | `hitl_options`           | 用户可选择的候选范围         |
| `查询重写节点`     | `rewrite_count`          | Query 已重写次数        |
|              | `rewrite_method`         | `Step-back / HyDE` |
|              | `rewritten_query`        | 重写后的检索 Query       |
|              | `step_back_question`     | Step-back 生成的问题    |
|              | `hyde_document`          | HyDE 生成的假设文档       |
| `子问题 RAG 节点` | `is_sub_agent`           | 标识当前 State 是否属于子问题 |
|              | `sub_results`            | 多个 Sub-Agent 的执行结果 |
| 全流程          | `rag_trace`              | 保存 RAG 执行轨迹        |
| 进度输出         | `request_context`        | 当前请求上下文            |
|              | `rag_step_group`         | RAG 进度所属分组         |
|              | `rag_step_group_label`   | 分组显示名称             |



## 1.4 文件结构设计

#### `backend/rag/`
```python
pipeline.py
│
├── run_rag_graph()                         # 普通 RAG 主入口，初始化状态并执行完整 LangGraph
│   │
│   ├── _initial_state()                   # 创建本轮 RAG 的初始 State
│   │
│   ├── rag_graph.invoke()                 # 执行已经构建好的 RAG Graph
│   │   │
│   │   ├── classify_complexity()          # 判断问题走简单 RAG 还是复杂问题拆分流程
│   │   │   ├── _emit()                    # 向外发送当前 RAG 执行进度
│   │   │   ├── _simple_question_fast_path_reason()
│   │   │   │                              # 用本地规则快速识别明显的简单问题
│   │   │   ├── _get_complexity_model()    # 获取用于复杂度判断和子问题生成的 FAST_MODEL
│   │   │   └── _route_after_complexity()  # 根据 complexity 决定进入简单流程还是复杂流程
│   │   │
│   │   ├── retrieve_initial()             # 对当前问题执行第一次知识库检索
│   │   │   ├── retrieve_documents()       # 调用第三阶段检索模块获取 Evidence
│   │   │   ├── _format_docs()             # 将结构化 Chunk 整理成供 LLM 使用的 Context
│   │   │   └── retrieval_trace_fields()   # 提取本次检索需要记录的 Trace 字段
│   │   │
│   │   ├── grade_documents_node()         # 判断当前 Evidence 是否足够回答问题
│   │   │   ├── _grade_for_no_docs()       # 检索结果为空时直接生成 no_knowledge 评分
│   │   │   ├── _get_grader_model()        # 获取 Evidence Grading 使用的模型
│   │   │   ├── _resolve_route()           # 根据评分结果决定 answer/rewrite/HITL/no_knowledge
│   │   │   └── _grade_update()            # 将评分结果整理后写回 RAG State 和 Trace
│   │   │       ├── _retrieval_status_for_route()
│   │   │       │                          # 将内部 route 转换成对外 retrieval_status
│   │   │       └── _default_hitl_prompt() # Grader 未给提示时生成默认 HITL 提示
│   │   │
│   │   ├── rewrite_question_node()        # Evidence 不足时执行一次 Query Rewrite
│   │   │   └── rewrite_query_once()       # 选择 Step-back 或 HyDE 生成新的检索 Query
│   │   │
│   │   ├── retrieve_rewritten()           # 使用重写后的 Query 重新检索知识库
│   │   │   ├── retrieve_documents()       # 执行第二次检索
│   │   │   ├── _format_docs()             # 整理第二次检索得到的 Evidence
│   │   │   └── retrieval_trace_fields()   # 更新第二次检索的 Trace 信息
│   │   │
│   │   ├── prepare_sub_questions()        # 整理复杂问题已经生成的子问题
│   │   │
│   │   ├── _fanout_sub_questions()        # 将多个子问题并行分发给 Sub-Agent
│   │   │   └── _initial_state()           # 为每个子问题创建独立的 RAG State
│   │   │
│   │   ├── rag_sub_agent()                # 对单个子问题执行检索和 Evidence 评分
│   │   │   ├── retrieve_initial()         # 检索当前子问题
│   │   │   └── grade_documents_node()     # 判断当前子问题 Evidence 是否可用
│   │   │
│   │   └── synthesis()                    # 合并多个 Sub-Agent 返回的 Evidence
│   │       ├── dedupe_documents()         # 按 chunk_id 去重不同子问题命中的重复 Evidence
│   │       ├── _format_docs()             # 生成合并后的统一 Context
│   │       └── normalize_rag_sub_trace()  # 标准化并汇总各个 Sub-Agent 的 Trace
│   │
│   ├── _is_hitl_result()                  # 判断本轮结果是否需要用户补充信息
│   └── _build_hitl_resume_state()         # 提取并保存后续 HITL 恢复需要的状态
│
└── resume_rag_from_hitl()                 # 用户补充信息后的 RAG 恢复入口
    │
    ├── _state_from_resume()               # 根据上一轮保存状态恢复新的 RAG State
    │   ├── _refined_question_for_hitl()   # 将用户补充信息拼接进原问题形成新 Query
    │   └── _initial_state()               # 创建新的基础 State
    │
    ├── _retrieve_resume_query()           # 使用补充后的问题执行针对性重新检索
    │   ├── retrieve_documents()           # 重新检索知识库
    │   ├── _format_docs()                 # 整理新的 Evidence Context
    │   └── grade_documents_node()         # 重新判断 Evidence 是否足够
    │
    ├── _is_hitl_result()                  # 判断补充后是否仍需继续 HITL
    └── _build_hitl_resume_state()         # 如果仍需补充，则保存新的恢复状态
```

---

# 2. 模块实现

## 2.1 `pipeline.py`

使用 LangGraph 编排完整 RAG 流程，对外提供两个入口：
- `run_rag_graph()`：正常问题的 RAG 执行入口。
- `resume_rag_from_hitl()`：用户补充信息后的 RAG 恢复入口。
---
### 2.1.1 定义RAG工作流
1. 创建状态图
2. 注册节点
	1. classify_complexity   # 判断简单/复杂问题
	2. prepare_sub_questions # 准备复杂问题的子问题
	3. retrieve_initial      # 第一次检索
	4. grade_documents       # Evidence 评分
	5. rewrite_question      # Query Rewrite
	6. retrieve_rewritten    # 重写后重新检索
	7. rag_sub_agent         # 执行单个子问题 RAG
	8. synthesis             # 合并多个子问题 Evidence
3. 定义入口：判断问题复杂度
4. 定义状态转移
```
                         ┌──────────────────────┐
                         │ classify_complexity  │
                         │ 判断问题复杂度       │
                         └──────────┬───────────┘
                                    │
                         _route_after_complexity()
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
                  simple                        complex
                     │                             │
                     ▼                             ▼
          ┌──────────────────┐          ┌──────────────────────┐
          │ retrieve_initial │          │ prepare_sub_questions│
          │ 第一次检索       │          │ 准备子问题           │
          └────────┬─────────┘          └──────────┬───────────┘
                   │                               │
                   │                         _fanout_sub_questions()
                   │                               │
                   ▼                               ▼
          ┌──────────────────┐          ┌──────────────────────┐
          │ grade_documents  │          │ rag_sub_agent × N    │
          │ Evidence 评分    │          │ 并行执行子问题 RAG   │
          └────────┬─────────┘          └──────────┬───────────┘
                   │                               │
          _route_after_grade()                     │
                   │                               ▼
          ┌────────┴─────────┐           ┌──────────────────┐
          │                  │           │    synthesis     │
   rewrite_question         end          │ 合并子问题证据   │
          │                  │           └────────┬─────────┘
          ▼                  ▼                    │
┌──────────────────┐       END                    ▼
│ rewrite_question │                             END
│ Query Rewrite    │
└────────┬─────────┘
         │
         ▼
┌────────────────────┐
│ retrieve_rewritten │
│ 重写后重新检索     │
└────────┬───────────┘
         │
         └────────────────────────→ grade_documents
```

```python
def build_rag_graph():
    graph = StateGraph(RAGState)

    # 节点注册
    graph.add_node("classify_complexity", classify_complexity) # 判断问题复杂度
    graph.add_node("prepare_sub_questions", prepare_sub_questions) # 复杂问题拆分
    graph.add_node("retrieve_initial", retrieve_initial) # 执行检索
    graph.add_node("grade_documents", grade_documents_node) # 判断证据知否足够回答问题
    graph.add_node("rewrite_question", rewrite_question_node) # Evidence 不足时执行一次 Query Rewrite
    graph.add_node("retrieve_rewritten", retrieve_rewritten) # 使用重写后的 Query 再次检索知识库
    graph.add_node("rag_sub_agent", rag_sub_agent) # 对子问题执行独立的检索和 Evidence 评分
    graph.add_node("synthesis", synthesis) # 合并多个 Sub-Agent 返回的 Evidence

    # 入口：复杂度分类
    graph.set_entry_point("classify_complexity")

    # 简单问题直接检索；复杂问题使用规划器一次产出的子问题。
    graph.add_conditional_edges(
        "classify_complexity",
        _route_after_complexity, # 路由函数，第一个节点执行之后，根据最新状态判断进入哪个节点。
        { # 根据路由函数的返回值，决定进入哪个节点。
            "retrieve_initial": "retrieve_initial",
            "prepare_sub_questions": "prepare_sub_questions",
        },
    )

    # 执行节点之后，路由函数根据子问题动态并行分发到多个 rag_sub_agent 节点
    graph.add_conditional_edges("prepare_sub_questions", _fanout_sub_questions)

    # 简单问题路径
    graph.add_edge("retrieve_initial", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        _route_after_grade,
        {
            "rewrite_question": "rewrite_question",
            "end": END,
        },
    )
    graph.add_edge("rewrite_question", "retrieve_rewritten")
    graph.add_edge("retrieve_rewritten", "grade_documents")

    # 并行子 Agent → 合成
    graph.add_edge("rag_sub_agent", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()


rag_graph = build_rag_graph()
```
#### 1）定义节点
##### 1-1）问题复杂度判断
1. 读取问题
2. `_simple_question_fast_path_reason`: 本地规则快速判断，如果是简单问题提前return
3. `_get_complexity_model`：无法规则判断时调用 判断模型
4. 解析 complexity / reason / sub_questions
5. `ComplexityResult`：格式化复杂度判断模型的输出
6. 校验结果，更新 RAG State
```python
def _emit(state: RAGState, icon: str, label: str, detail: str = "") -> None:
    ctx = state["request_context"]
    ctx.emit_rag_step(
        icon,
        label,
        detail,
        group=state.get("rag_step_group"),
        group_label=state.get("rag_step_group_label"),
    )

def classify_complexity(state: RAGState) -> RAGState:
    """使用 FAST_MODEL 判断问题复杂度。"""
    # 1. 从当前 RAG State 中读取用户原始问题
    question = state["question"]
     # 2. 向前端发送当前正在进行复杂度判断的进度信息
    _emit(state, "🧭", "正在分析问题复杂度...")

    # 3. 先使用本地规则判断是否是明显的简单问题，避免不必要的 LLM 调用
    fast_path_reason = _simple_question_fast_path_reason(question) # 快速规则判断是否是简单问题
    if fast_path_reason: 
        _emit(state, "⚡", "快速判断为简单问题 → 走标准 RAG 流程")
        return {"complexity": "simple", "complexity_reason": fast_path_reason}

    # 5. 本地规则无法确定时，获取用于复杂度判断的 FAST_MODEL
    model = _get_complexity_model()
    if not model:
        raise RuntimeError("FAST_MODEL is required for complexity planning")

    # 7. 调用 FAST_MODEL，并通过 ComplexityResult 约束结构化输出
    prompt = COMPLEXITY_PROMPT.format(question=question)
    result = with_structured_output(model, ComplexityResult).invoke(
        [{"role": "user", "content": prompt}]
    )
    complexity = (result.complexity or "simple").strip().lower()
    reason = (result.reason or "").strip()
    sub_questions = [ # 9. 清洗子问题，去掉空字符串，并最多保留 4 个
        item.strip()
        for item in (result.sub_questions or [])
        if item and item.strip()
    ][:4]
    if complexity not in ("simple", "complex"):
        raise ValueError(f"Unsupported complexity result: {complexity}")
    if complexity == "complex" and not sub_questions:
        raise ValueError("Complexity planner returned no sub-questions")

    if complexity == "simple":
        _emit(state, "✅", "简单问题 → 走标准 RAG 流程", f"理由: {reason[:60]}")
    else:
        _emit(state, "🔀", "复杂问题 → 将分解为子问题并行检索", f"理由: {reason[:60]}")

    # 13. 返回对 RAG State 的更新
    return {
        "complexity": complexity,
        "complexity_reason": reason,
        "sub_questions": sub_questions if complexity == "complex" else [],
    }
```
###### 1-1-1）规则判断简单问题
1. **排除明显复杂问题**：字符长度过长、包含复杂问题关键词、多个查询维度、多个问句或明显分隔符时，直接返回 `None`，交给 FAST_MODEL 判断。
2. **再识别明显简单问题**：如果包含“是什么、参数、定义、多少”等单事实查询标记，直接判为简单问题。
3. **最后判断短问题**：如果没有命中前面的复杂条件，且问题长度不超过 18 个字符，也按简单单意图问题处理；其余情况返回 `None`。
```python
_SIMPLE_QUERY_MARKERS = (
    "是什么",
    "是谁",
    "哪里",
    "何时",
    "多少",
    "是否",
    "哪个",
    "哪种",
    "属性",
    "参数",
    "规格",
    "定义",
    "含义",
    "what is",
    "who is",
    "where is",
    "when is",
    "how many",
    "which",
)

_COMPLEX_QUERY_MARKERS = (
    "比较",
    "对比",
    "区别",
    "差异",
    "优缺点",
    "优势",
    "劣势",
    "分析",
    "总结",
    "综合",
    "原因",
    "成因",
    "影响",
    "方案",
    "步骤",
    "如何",
    "为什么",
    "以及",
    "同时",
    "并且",
    "和",
    "与",
    "谁更",
    "compare",
    "versus",
    "difference",
    "different",
    "analyze",
    "summarize",
    "trade-off",
    "pros and cons",
    "why ",
    "how ",
    "complex",
)

_QUERY_DIMENSION_MARKERS = (
    "属性",
    "武器",
    "定位",
    "技能",
    "机制",
    "参数",
    "规格",
    "性能",
    "价格",
    "优点",
    "缺点",
    "作用",
)


def _simple_question_fast_path_reason(question: str) -> Optional[str]:
    """Return a reason only when a local rule can confidently classify a simple query."""
    # 去掉首尾空格并转换为小写
    normalized = re.sub(r"\s+", " ", (question or "").strip()).lower()
    # 长度超过48个字符，无法判断
    if not normalized or len(normalized) > 48:
        return None
    # 检查是否包含复杂问题标记
    if any(marker in normalized for marker in _COMPLEX_QUERY_MARKERS):
        return None
    # 检查是否包含分隔符
    if "、" in normalized:
        return None
    # 检查是否包含中文字符且空格数量大于等于2
    if re.search(r"[\u4e00-\u9fff]", normalized) and normalized.count(" ") >= 2:
        return None
    # 检查是否包含多个查询维度标记
    if sum(marker in normalized for marker in _QUERY_DIMENSION_MARKERS) >= 2:
        return None
    # 检查是否包含多个问号或分号
    if sum(normalized.count(mark) for mark in ("?", "？", ";", "；")) > 1:
        return None
    # 检查是否包含简单问题标记
    if any(marker in normalized for marker in _SIMPLE_QUERY_MARKERS):
        return "obvious_simple_fast_path:single_fact_marker"
    # 检查是否为简短的单一意图查询
    if len(normalized.rstrip("?？。.!！")) <= 18:
        return "obvious_simple_fast_path:short_single_intent"
    return None
```
###### 1-1-2）获取复杂度判断模型
```python
COMPLEXITY_PROMPT = (
    "你是一个问题复杂度规划器。请判断用户问题的复杂度。\n\n"
    "【简单问题】：事实查询、定义查询、单一信息点查询、明确的二选一问题、"
    "某个具体属性/参数/规格的查询。\n"
    "【复杂问题】：需要跨文档综合、多角度分析、比较对比、多步骤推理、"
    "需要综合多个信息源才能完整回答的问题。\n\n"
    "用户问题：{question}\n\n"
    "如果是复杂问题，请同时给出 2-4 个互不重叠、可独立检索的子问题；"
    "如果是简单问题，sub_questions 留空。"
)

def _get_complexity_model():
    """FAST_MODEL 用于问题复杂度分类和子问题分解。"""
    global _complexity_model
    if not API_KEY or not FAST_MODEL:
        return None
    if _complexity_model is None:
        _complexity_model = init_chat_model(
            model=FAST_MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,
            stream_usage=True,
        )
    return _complexity_model
```
###### 1-1-3）定义判断模型输出格式
判断模型输出：
- 复杂度
- 分类理由
- 子问题（若为复杂问题）
```python
class ComplexityResult(BaseModel):
    """问题复杂度分类结果。"""

    complexity: Literal["simple", "complex"] = Field(
        description="问题复杂度：'simple' 为简单问题，'complex' 为复杂问题"
    )
    reason: str = Field(default="", description="分类理由")
    sub_questions: List[str] = Field(
        default_factory=list,
        description="复杂问题对应的 2-4 个可独立检索子问题；简单问题留空",
        max_length=4,
    )
```
##### 1-2）准备子问题
对复杂度判断阶段已经生成的子问题进行清洗，并发送子问题准备完成的进度信息，为后续并行分发做准备。
```python
def prepare_sub_questions(state: RAGState) -> RAGState:
    """Emit the sub-questions produced by the complexity planner."""
    planned_sub_questions = [
        item.strip()
        for item in (state.get("sub_questions") or [])
        if item and item.strip()
    ]
    for i, sq in enumerate(planned_sub_questions, 1):
        _emit(state, "📌", f"子问题 {i}", f"{sq[:80]} 已加入并行检索")
    return {"sub_questions": planned_sub_questions}
```
##### 1-3）知识库检索
调用检索模块定义的检索入口，将检索结果（chunk）整理成llm的上下文，并保存本次检索的路径信息（检索模块定义的trace，包括召回、Auto-Merging、Rerank 等），返回并更新state的：
- query
- docs
- context
- rag_trace
```python
def _format_docs(docs: List[dict]) -> str:
    #将结构化 Chunk 转成供 LLM 使用的 Context
    if not docs:
        return ""
    chunks = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("filename", "Unknown")
        page = doc.get("page_number", "N/A")
        text = doc.get("text", "")
        chunks.append(f"[{i}] {source} (Page {page}):\n{text}")
    return "\n\n---\n\n".join(chunks)
    
def retrieve_initial(state: RAGState) -> RAGState:
    query = state["question"]
    _emit(state, "🔍", "正在检索知识库...", "初始检索")
    # 1. 调用检索模块
    retrieved = retrieve_documents(query, top_k=RETRIEVAL_TOP_K)
    results = retrieved.get("docs", [])
    retrieve_meta = retrieved.get("meta", {})
    # 2. 将结构化 Evidence 转成文本 Context
    context = _format_docs(results)
    _emit(
        state,
        "🧱",
        "三级分块检索",
        (
            f"叶子层 L{retrieve_meta.get('leaf_retrieve_level', 3)} 召回，"
            f"候选 {retrieve_meta.get('candidate_k', 0)}"
        ),
    )
    _emit(
        state,
        "🧩",
        "Auto-merging 合并",
        (
            f"启用: {bool(retrieve_meta.get('auto_merge_enabled'))}，"
            f"应用: {bool(retrieve_meta.get('auto_merge_applied'))}，"
            f"替换片段: {retrieve_meta.get('auto_merge_replaced_chunks', 0)}"
        ),
    )
    _emit(state, "✅", f"检索完成，找到 {len(results)} 个片段", f"模式: {retrieve_meta.get('retrieval_mode', 'hybrid')}")
    if not results:
        _emit(state, "⚠️", "无可用片段，将进入证据评分短路判断")
    rag_trace = {# 3. 保存本次检索 Trace
        "tool_used": True,
        "tool_name": "search_knowledge_base",
        "query": query,
        "retrieved_chunks": results,
        "initial_retrieved_chunks": results,
        "retrieval_stage": "initial",
        "complexity": state.get("complexity"),
        "complexity_reason": state.get("complexity_reason"),
        **retrieval_trace_fields(retrieve_meta),
    }
    return {# 4. 更新 State
        "query": query,
        "docs": results,
        "context": context,
        "rag_trace": rag_trace,
    }
```
##### 1-4）证据评分
判断当前检索到的 Evidence 是否可以支撑回答，并决定下一步 Route：
1.  获取 Evidence，如果检索到的文档为空，返回空分数
2. `_get_grader_model`：获取证据评分模型对结果进行评分
3. `EvidenceGrade`：评分模型输出规范化
4. 根据评分决定最终 Route
    - `answer`：Evidence 足够。保存Evidence，Graph结束
    - `clarify`：问题缺少必要条件，需要用户补充信息。清空Evidence，等待用户补充，Graph结束
    - `scope_select`：存在多个候选范围，需要用户选择具体范围。清空Evidence，等待用户选择范围，Graph结束
    - `no_knowledge`：没有 Evidence。清空Evidence，Graph结束
    - `rewrite`：Evidence 不足，进入重写节点。
```python
def _grade_for_no_docs() -> EvidenceGrade:
    '''没检索到文档是，给出的分数'''
    return EvidenceGrade(
        relevance="none",
        answerability="none",
        ambiguity="none",
        route="no_knowledge",
        confidence=1.0,
        reason="no_retrieved_documents",
    )

def grade_documents_node(state: RAGState) -> RAGState:
    """评估当前检索到的 Evidence，并决定后续 RAG 路由。"""
    _emit(state, "📊", "正在评估证据质量...")
    docs = state.get("docs") or []
    if not docs:# 如果没有检索到任何 Evidence，直接生成 no_knowledge 评分，不调用 LLM
        grade = _grade_for_no_docs()
    else:
        grader = _get_grader_model()
        if not grader:
            raise RuntimeError("GRADE_MODEL is required for evidence grading")
        question = state["question"]
        context = state.get("context", "")
        prompt = EVIDENCE_GRADE_PROMPT.format(question=question, context=context)
        grade = with_structured_output(grader, EvidenceGrade).invoke(
            [{"role": "user", "content": prompt}]
        )

    # 8. 根据 Evidence 评分和当前 State 决定最终路由
    route = _resolve_route(grade, state)
    # 9. 将 route 和评分结果整理成统一的 retrieval_status、HITL 信息等
    grade_update = _grade_update(grade, route)
    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update(grade_update)

    if route == "answer":
        if grade.answerability == "partial":
            _emit(state, "🟡", "保留部分相关证据", f"置信度: {grade.confidence:.2f}")
        else:
            _emit(state, "✅", "证据足够，返回检索片段", f"置信度: {grade.confidence:.2f}")
    elif route == "rewrite":
        _emit(state, "⚠️", "证据不足，将改写查询一次", f"置信度: {grade.confidence:.2f}")
    elif route in ("clarify", "scope_select"):
        _emit(state, "❓", "需要用户补充信息", grade_update["hitl_prompt"])
    else:
        _emit(state, "⛔", "知识库中未找到可用证据", grade.reason or "no_knowledge")

    update = {
        "route": route,
        "retrieval_status": grade_update["retrieval_status"],
        "evidence_relevance": grade.relevance,
        "evidence_answerability": grade.answerability,
        "evidence_ambiguity": grade.ambiguity,
        "evidence_confidence": grade.confidence,
        "missing_slots": grade.missing_slots,
        "hitl_prompt": grade_update["hitl_prompt"],
        "hitl_options": grade.hitl_options,
        "rag_trace": rag_trace,
    }

    # 13. 如果 Evidence 不能直接使用，则清空 docs 和 context
    if route in ("no_knowledge", "clarify", "scope_select"):
        if route in ("clarify", "scope_select") and docs:
            rag_trace["retrieved_chunks"] = []
        update.update({"docs": [], "context": ""})

    return update
```
###### 1-4-1）获取证据评分模型
```python
def _get_grader_model():
    '''获取证据评分模型'''
    global _grader_model
    if not API_KEY or not GRADE_MODEL:
        return None
    if _grader_model is None:
        _grader_model = init_chat_model(
            model=GRADE_MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,
            stream_usage=True,
        )
    return _grader_model
EVIDENCE_GRADE_PROMPT = (
    "你是 RAG 证据评分器。请只根据检索片段判断它们是否足以回答用户问题，"
    "不要补充片段里没有的信息。\n\n"
    "用户问题：\n{question}\n\n"
    "检索片段：\n{context}\n\n"
    "请按以下规则给出结构化结果：\n"
    "- relevance: none 表示主题不相关；weak 表示主题接近但证据弱；strong 表示主题明确相关。\n"
    "- answerability: none 表示不能回答；partial 表示有部分线索但不足以给确定答案；"
    "sufficient 表示片段能直接或组合支撑答案。\n"
    "- ambiguity: missing_slot 表示缺少角色名、版本、文件类型、模块名、产品线等关键条件；"
    "multiple_candidates 表示多个候选方向都可能相关；none 表示无明显歧义。\n"
    "- route 只能选择：answer、rewrite、clarify、scope_select、no_knowledge。\n"
    "  answer: relevance=strong 且 answerability=sufficient。\n"
    "  rewrite: 有相关信号，但像是问法、别名或泛化程度导致证据不足。\n"
    "  clarify: 缺少关键条件，需要用户补充。\n"
    "  scope_select: 多个候选方向都相关，需要用户选择。\n"
    "  no_knowledge: 无召回或主题不相关。\n"
    "- 如果 route 是 clarify 或 scope_select，请给 hitl_prompt；如果能列出选项，请给 hitl_options。"
)
```
###### 1-4-2）定义评分模型输出
```python
class EvidenceGrade(BaseModel):
    """结构化证据评分：同时判断相关性、可回答性与下一步路由。"""

    relevance: Literal["none", "weak", "strong"] = Field(
        description="检索片段与问题的主题相关性"
    )
    answerability: Literal["none", "partial", "sufficient"] = Field(
        description="检索片段是否足以回答问题"
    )
    ambiguity: Literal["none", "missing_slot", "multiple_candidates"] = Field(
        default="none",
        description="问题是否缺条件或存在多个候选方向"
    )
    route: Literal["answer", "rewrite", "clarify", "scope_select", "no_knowledge"] = Field(
        description="下一步路由"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # 当 ambiguity="missing_slot" 时，记录具体缺少哪些条件。
    missing_slots: List[str] = Field(default_factory=list)
    # 需要 HITL 时，直接给用户看的补充问题，例如：请问你指的是哪个模型？
    hitl_prompt: str = ""
    # 如果是 scope_select，可以给用户提供候选选项
    hitl_options: List[str] = Field(default_factory=list)
    reason: str = ""
```
###### 1-4-3）评分结果路由
根据评分模型的输出，决定下一步路由：
- 无证据/不相关 → `no_knowledge`
- 有歧义 → `clarify / scope_select`
- `strong + sufficient` → `answer`
- **Sub-Agent partial → 保留，交给 synthesis**
- **主流程第一次 partial → rewrite**
- **主流程 Rewrite 后仍 partial → clarify**
- Rewrite 后仍完全不可回答 → `no_knowledge`
```python
def _resolve_route(grade: EvidenceGrade, state: RAGState) -> str:
    docs = state.get("docs") or []
    rewrite_count = int(state.get("rewrite_count") or 0)
    is_sub_agent = bool(state.get("is_sub_agent"))
    route = grade.route

    if not docs or grade.relevance == "none":
        return "no_knowledge"

    if grade.ambiguity == "missing_slot":
        return "clarify"
    if grade.ambiguity == "multiple_candidates":
        return "scope_select"

    answer_is_supported = grade.relevance == "strong" and grade.answerability == "sufficient"
    if route == "answer" and answer_is_supported:
        return "answer"

    # 子问题不做二次纠错。partial 证据交给 synthesis 合并，完全不可回答则停止。
    if is_sub_agent:
        if grade.answerability in ("partial", "sufficient"):
            return "answer"
        return "no_knowledge"

    if route == "rewrite" and rewrite_count < 1:
        return "rewrite"

    if route == "rewrite" and rewrite_count >= 1:
        if grade.answerability == "partial":
            return "clarify"
        return "no_knowledge"

    if grade.answerability == "partial":
        if rewrite_count < 1:
            return "rewrite"
        return "clarify"

    if answer_is_supported:
        return "answer"

    return "no_knowledge"
```
###### 1-4-4）根据评分结果更新rag trace状态
这里更新的状态字典主要是给rag_trace，而不是langgraph
1. 根据评分结果和路由，更新当前检索状态
2. 如果需要人为补充信息，更新补充提示词

```python
def _retrieval_status_for_route(route: str, grade: EvidenceGrade) -> str:
    # 根据评分路由结果，输出当前检索状态
    if route == "answer":
        if grade.answerability == "partial":
            return "partial"
        return "answerable"
    if route == "rewrite":
        return "needs_rewrite"
    if route == "clarify":
        return "needs_clarification"
    if route == "scope_select":
        return "needs_scope_selection"
    return "no_knowledge"
    
def _default_hitl_prompt(route: str, grade: EvidenceGrade) -> str:
    # 如果评分模型判断需要补充信息，更新提示信息
    if grade.hitl_prompt:
        return grade.hitl_prompt
    if route == "scope_select":
        return "我在知识库中找到了多个可能相关的方向。你想问的是哪一个？"
    if grade.missing_slots:
        return "我找到了相关内容，但还缺少关键信息：" + "、".join(grade.missing_slots)
    return "我找到了相关内容，但证据不足以确定答案。请补充一下你具体想问的条件。"
    
def _grade_update(grade: EvidenceGrade, route: str) -> dict:
    # 评分之后，更新全局state
    # route决定LangGraph 下一步怎么走，retrieval_status 指明这次检索现在处于什么状态
    status = _retrieval_status_for_route(route, grade)
    hitl_prompt = _default_hitl_prompt(route, grade) if route in ("clarify", "scope_select") else ""
    return {
        "retrieval_status": status,
        "evidence_relevance": grade.relevance,
        "evidence_answerability": grade.answerability,
        "evidence_ambiguity": grade.ambiguity,
        "evidence_confidence": grade.confidence,
        "evidence_reason": grade.reason,
        "missing_slots": grade.missing_slots,
        "hitl_prompt": hitl_prompt,
        "hitl_options": grade.hitl_options,
        "route": route,
    }
```

##### 1-5）查询重写
1. 检查 `rewrite_count`，最多只允许 Rewrite 一次。如果超过，就更新状态：no_knowledge
2. `rewrite_query_once()`：选择重写方式（Step-back 或 HyDE）并执行（`utils.py` 定义过）
	1. 获取重写模型
	2. 选择重写方式，执行重写
	3. 规范化重写模型的输出
	4. 返回重写结果
4. 读取重写结果，包括：
	- 重写方式
	- 重写的 Query
	- 重写过程中生成的中间结果
5. 校验重写结果是否合法。
6. 更新全局状态
```python
def rewrite_question_node(state: RAGState) -> RAGState:
    question = state["question"]
    _emit(state, "✏️", "正在重写查询...")

    # 1. 重写过一次了
    rewrite_count = int(state.get("rewrite_count") or 0)
    if rewrite_count >= 1: # 如果改写次数已达到上限，返回没检索到
        rag_trace = state.get("rag_trace", {}) or {}
        rag_trace.update({
            "retrieval_status": "no_knowledge",
            "route": "no_knowledge",
            "evidence_reason": "rewrite_budget_exhausted",
        })
        _emit(state, "⛔", "改写预算已用完，停止检索")
        return {
            "route": "no_knowledge",
            "retrieval_status": "no_knowledge",
            "docs": [],
            "context": "",
            "rag_trace": rag_trace,
        }

    # 2. 选择重写方式，并执行
    _emit(state, "🧠", "选择 Step-back / HyDE 重写方式")
    rewrite = rewrite_query_once(question)
    # 3. 获取重写结果
    rewrite_method = (rewrite.get("rewrite_method") or "").strip()
    step_back_question = (rewrite.get("step_back_question") or "").strip()
    hyde_document = (rewrite.get("hyde_document") or "").strip()
    rewritten_query = (rewrite.get("rewritten_query") or "").strip()
    if rewrite_method not in ("step_back", "hyde") or not rewritten_query:
        raise ValueError("Query rewriting returned an incomplete result")
    if rewrite_method == "step_back" and (not step_back_question or hyde_document):
        raise ValueError("Step-back rewriting returned an invalid result")
    if rewrite_method == "hyde" and (not hyde_document or step_back_question):
        raise ValueError("HyDE rewriting returned an invalid result")

    method_label = "Step-back" if rewrite_method == "step_back" else "HyDE"
    _emit(state, "✅", f"已选择 {method_label} 重写", "本轮只执行这一种重写检索")

    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update({
        "rewrite_method": rewrite_method,
        "rewritten_query": rewritten_query,
        "rewrite_count": rewrite_count + 1,
    })
    if step_back_question:
        rag_trace["step_back_question"] = step_back_question
    if hyde_document:
        rag_trace["hyde_document"] = hyde_document

    # 更新全局状态
    return {
        "rewrite_method": rewrite_method,
        "rewritten_query": rewritten_query,
        "step_back_question": step_back_question,
        "hyde_document": hyde_document,
        "rewrite_count": rewrite_count + 1,
        "rag_trace": rag_trace,
    }
```

##### 1-6）重写后检索
1. 使用重写后的query检索，重写后的query形式：`原query+重写方式+新query`
```python
def retrieve_rewritten(state: RAGState) -> RAGState:
    rewrite_method = (state.get("rewrite_method") or "").strip()
    if rewrite_method not in ("step_back", "hyde"):
        raise ValueError("rewrite_method is required for rewritten retrieval")
    rewritten_query = (state.get("rewritten_query") or "").strip()
    if not rewritten_query:
        raise ValueError("rewritten_query is required for rewritten retrieval")
    method_label = "Step-back" if rewrite_method == "step_back" else "HyDE"
    _emit(state, "🔄", f"使用 {method_label} 查询重新检索...")
    retrieved = retrieve_documents(rewritten_query, top_k=RETRIEVAL_TOP_K)
    results = retrieved.get("docs", [])
    retrieve_meta = retrieved.get("meta", {})
    context = _format_docs(results)
    _emit(
        state,
        "🧱",
        f"{method_label} 三级检索",
        (
            f"L{retrieve_meta.get('leaf_retrieve_level', 3)} 召回，"
            f"候选 {retrieve_meta.get('candidate_k', 0)}，"
            f"合并替换 {retrieve_meta.get('auto_merge_replaced_chunks', 0)}"
        ),
    )
    _emit(state, "✅", f"重写检索完成，共 {len(results)} 个片段")
    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update({
        "rewrite_method": rewrite_method,
        "rewritten_query": rewritten_query,
        "retrieved_chunks": results,
        "rewrite_retrieved_chunks": results,
        "retrieval_stage": "rewritten",
        **retrieval_trace_fields(retrieve_meta),
    })
    if state.get("step_back_question"):
        rag_trace["step_back_question"] = state["step_back_question"]
    if state.get("hyde_document"):
        rag_trace["hyde_document"] = state["hyde_document"]
    return {"docs": results, "context": context, "rag_trace": rag_trace}
```

##### 1-7）执行子问题 RAG
- 这里到了第二条分支，执行复杂度判断之后，如果分析为需要拆分子问题，就会执行`准备子问题`节点，该节点执行之后就已经有了多个子问题，现在本函数就针对每个子问题进行检索（后续要给本节点和`准备子问题`节点增加一个路由函数，将多个子问题分成多个state并行处理，这里先只管一个子问题）
1. 使用当前子问题进行知识库检索（注意，这里是在节点内部手动调用了了一个节点，不会触发langgraph自动按边调用下一个节点）
2. 对当前子问题手动调用评分节点
```python
def rag_sub_agent(state: RAGState) -> RAGState:
    """Run the only reachable sub-agent path directly: retrieve → grade."""
    question = state.get("question", "")
    result = dict(state)
    # 1. 子问题检索
    result.update(retrieve_initial(result))
    # 2. 子问题评分
    result.update(grade_documents_node(result))
    trace = result.get("rag_trace") or {}
    return {
        "sub_results": [{
            "question": question,
            "docs": result.get("docs", []),
            "retrieval_status": result.get("retrieval_status") or trace.get("retrieval_status"),
            "route": result.get("route") or trace.get("route"),
            "rag_trace": trace,
        }],
    }
```
##### 1-8）合并子问题结果
1. 合并可用子问题的evidence，只保留检索状态为 `answerable / partial` 的子问题结果。
2. 合并所有子问题的 `docs`
3. `dedupe_documents()`：对合并结果去重。（utils.py定义）
4. 判断下一步路由：
	1. 如果合并后有evidence，就走路由到回答分支
		1. 再看证据完整度，如果所有有效子问题都是answerable，那么父问题状态就是answerable
		2. 只要有一个子问题是partial，父问题状态就是partial
	2. 如果没有evidence，就判断是否有需要用户介入的子问题（HITL）
		1. 优先看是否有需要用户选择（needs_scope_selection）的子问题，有则路由到scope_select（注意，HITL不是Graph节点，而是更上层，因此这里会结束Grpah）
		2. 其次看是否需要用户澄清（needs_clarification）的子问题，有则路由到clarify
	3. 如果连HITL都没有，就路由到no_knowledge
    
> 对于简单问题，如果检索状态为paritial，那么会执行重写。但是复杂问题的子问题如果为paritial，并不会重写，而是视作跟answerable一样，这里的逻辑有待优化？？？

```python
def synthesis(state: RAGState) -> RAGState:
    """合并所有子 Agent 检索到的文档，去重排序后输出最终上下文。"""
    # 1. 获取所有子问题的执行结果
    sub_results = state.get("sub_results", [])
    _emit(state, "🔬", f"正在合成 {len(sub_results)} 个子问题的检索结果...")

    # 2. 只收集 answerable / partial 的子问题 Evidence
    all_docs: List[dict] = []
    for result in sub_results:
        status = result.get("retrieval_status")
        if status not in ("answerable", "partial"):
            continue
        docs = result.get("docs", [])
        all_docs.extend(docs)

    # 3. 多个子问题可能检索到相同 Chunk，因此统一去重
    deduped = dedupe_documents(all_docs)
    for idx, item in enumerate(deduped, 1):# 去重后重新设置最终 Evidence 的顺序
        item["rrf_rank"] = idx

    # 4. 将最终结构化 Evidence 转换成 LLM 使用的 Context
    context = _format_docs(deduped)
    if deduped:
        _emit(state, "✅", f"合成完成，共 {len(deduped)} 个去重片段")
    else:
        _emit(state, "⛔", "所有子问题都没有可用证据")

    # 合并所有子 Agent 的 rag_trace
    sub_traces = []
    for result in sub_results:
        trace = result.get("rag_trace")
        if trace:
            normalized_trace = normalize_rag_sub_trace(trace)
            if normalized_trace:
                sub_traces.append(normalized_trace)

    # 6. 判断合并后的 Evidence 是否可用
    original_trace = state.get("rag_trace") or {}
    has_docs = bool(deduped)
    # 去重之后还有evidence就先默认认为 answerable
    retrieval_status = "answerable" if has_docs else "no_knowledge"
    # 只要有一个有效子问题只有 partial，整个复杂问题的检索结果也标记为 partial
    if has_docs and any(result.get("retrieval_status") == "partial" for result in sub_results):
        retrieval_status = "partial"

    # 7. 如果最终没有可用 Evidence，
    # 检查各个子问题是否有需要 HITL 的情况
    hitl_traces = [
        trace for trace in sub_traces
        if trace.get("retrieval_status") in ("needs_clarification", "needs_scope_selection")
    ]
    hitl_route = None
    hitl_prompt = ""
    hitl_options: List[str] = []
    if not has_docs and hitl_traces:
        # 如果多个子agent没找到evidence，但是有 HITL情况
        # 根据trace, 找到第一个需要人为选择的子问题trace
        scope_trace = next(
            (trace for trace in hitl_traces if trace.get("retrieval_status") == "needs_scope_selection"),
            None,
        )
        # 如果没找到人为选择trace，再找人为补充trace
        chosen_trace = scope_trace or hitl_traces[0]
        retrieval_status = chosen_trace.get("retrieval_status") or "needs_clarification"
        hitl_route = "scope_select" if retrieval_status == "needs_scope_selection" else "clarify"

        # 合并多个子问题产生的 HITL 提示，并去重
        prompts = [
            trace.get("hitl_prompt")
            for trace in hitl_traces
            if trace.get("hitl_prompt")
        ]
        hitl_prompt = "；".join(dict.fromkeys(prompts))
        # 合并多个子问题的候选选项，并去重
        for trace in hitl_traces:
            for option in trace.get("hitl_options") or []:
                if option not in hitl_options:
                    hitl_options.append(option)

    # 9. 整理复杂问题整体的 RAG Trace
    rag_trace = {
        **original_trace,
        "tool_used": True,
        "tool_name": "search_knowledge_base",
        "query": state["question"],  # 原始复杂问题
        "retrieved_chunks": deduped, # 合并、去重后的最终 Evidence
        "retrieval_stage": "synthesis",
        "complexity": "complex",  # 复杂问题规划信息
        "complexity_reason": state.get("complexity_reason", ""),
        "sub_questions": state.get("sub_questions", []),
        "sub_agent_count": len(sub_results),  # Sub-Agent 执行情况
        "synthesis_merged_count": len(all_docs),
        "sub_traces": sub_traces,
        "retrieval_status": retrieval_status,   # 最终检索状态
        "evidence_relevance": "strong" if has_docs else "none",
        "evidence_answerability": "partial" if retrieval_status == "partial" else ("sufficient" if has_docs else "none"),
        "evidence_confidence": None,  # synthesis 本身没有重新调用评分模型，因此没有新的 confidence
        "route": "answer" if has_docs else (hitl_route or "no_knowledge"),
        "hitl_prompt": hitl_prompt,
        "hitl_options": hitl_options,
    }

    # 10. 返回本节点产生的 State 更新
    # LangGraph 会把这些字段合并回主 RAGState
    return {
        "docs": deduped,
        "context": context,
        "route": "answer" if has_docs else (hitl_route or "no_knowledge"),
        "retrieval_status": retrieval_status,
        "hitl_prompt": hitl_prompt,
        "hitl_options": hitl_options,
        "rag_trace": rag_trace,
    }
```
#### 2）定义边
##### 2-1）复杂度判断路由
复杂度判断节点判断完之后更新state，然后本函数根据得到的复杂度决定流向哪个节点：
- 简单问题 -> 知识库检索节点
- 复杂问题 -> 准备子问题节点
```python
def _route_after_complexity(state: RAGState):
    """简单问题直接检索，复杂问题并行检索规划出的子问题。"""
    if state.get("complexity") == "complex":
        return "prepare_sub_questions"
    return "retrieve_initial"
```
##### 2-2）子问题并行分发
子问题准备节点执行完之后，需要一个函数**动态创建多个 `Send`**，这里相当于一次性创建多个state，然后一次性路由到四个一样的节点。后续每个子问题分别执行一次 子问题RAG节点。
```python
def _fanout_sub_questions(state: RAGState):
    """将规划出的子问题通过 Send API 并行分发到 rag_sub_agent。"""
    sub_qs = state.get("sub_questions") or []
    ctx = state["request_context"]
    return [
        Send(
            "rag_sub_agent",
            _initial_state(
                sq,
                ctx,
                is_sub_agent=True,
                rag_step_group=f"子问题 {i}",
                rag_step_group_label=sq,
            ),
        )
        for i, sq in enumerate(sub_qs, 1)
    ]
```
##### 2-3）评分结果路由
执行完评分之后，模型会输出answer、rewrite、clarify、scope_select、no_knowledge。需要把他们分别路由：
- rewrite → rewrite_question 节点
- answer / clarify / scope_select / no_knowledge → END （因为这些状态不需要检索了，都是提交上层了）
```python
def _route_after_grade(state: RAGState) -> Literal["rewrite_question", "end"]:
    if state.get("route") == "rewrite":
        return "rewrite_question"
    return "end"
```
### 2.1.2 执行RAG工作流
1. `_initial_state()`：根据用户问题创建初始 RAG State。
2. 执行前面已经定义好的 LangGraph。
3. `_is_hitl_result()`：Graph 结束后判断是否需要用户补充信息。
	- `_build_hitl_resume_state()`：如果需要 HITL，则保存后续恢复所需状态。
```python
def run_rag_graph(question: str, ctx: ChatRequestContext) -> dict:
    result = rag_graph.invoke(_initial_state(question, ctx))
    if _is_hitl_result(result):
        result["hitl_resume_state"] = _build_hitl_resume_state(result)
    return result
```
#### 1）初始化RAG State
- 根据数据结构设计部分，初始胡RAG State
```python
class RAGState(TypedDict):
    question: str
    query: str
    context: str
    docs: List[dict]
    route: Optional[str]
    retrieval_status: Optional[str]
    evidence_relevance: Optional[str]
    evidence_answerability: Optional[str]
    evidence_ambiguity: Optional[str]
    evidence_confidence: Optional[float]
    missing_slots: Optional[List[str]]
    hitl_prompt: Optional[str]
    hitl_options: Optional[List[str]]
    rewrite_count: int
    rewrite_method: Optional[str]
    rewritten_query: Optional[str]
    step_back_question: Optional[str]
    hyde_document: Optional[str]
    rag_trace: Optional[dict]
    # 复杂度路由新增字段
    complexity: Optional[str]
    complexity_reason: Optional[str]
    sub_questions: Optional[List[str]]
    is_sub_agent: bool
    sub_results: Annotated[List[dict], operator.add]
    request_context: ChatRequestContext
    rag_step_group: Optional[str]
    rag_step_group_label: Optional[str]


def _initial_state(
    question: str,
    ctx: ChatRequestContext,
    *,
    is_sub_agent: bool = False,
    rag_step_group: Optional[str] = None,
    rag_step_group_label: Optional[str] = None,
) -> dict:
    return {
        "question": question,
        "query": question,
        "context": "",
        "docs": [],
        "route": None,
        "retrieval_status": None,
        "evidence_relevance": None,
        "evidence_answerability": None,
        "evidence_ambiguity": None,
        "evidence_confidence": None,
        "missing_slots": [],
        "hitl_prompt": "",
        "hitl_options": [],
        "rewrite_count": 0,
        "rewrite_method": None,
        "rewritten_query": None,
        "step_back_question": None,
        "hyde_document": None,
        "rag_trace": None,
        "complexity": None,
        "complexity_reason": None,
        "sub_questions": None,
        "is_sub_agent": is_sub_agent,
        "sub_results": [],
        "request_context": ctx,
        "rag_step_group": rag_step_group,
        "rag_step_group_label": rag_step_group_label,
    }
```

#### 2）判断是否需要HITL
Graph 已经结束后，根据最终的 `route` 和 `retrieval_status` 判断当前结果是否需要用户继续补充信息。
```python
def _is_hitl_result(result: dict | None) -> bool:
    if not isinstance(result, dict):
        return False
    trace = result.get("rag_trace") or {}
    status = result.get("retrieval_status") or trace.get("retrieval_status")
    route = result.get("route") or trace.get("route")
    return status in ("needs_clarification", "needs_scope_selection") or route in ("clarify", "scope_select")

```

#### 3）保存 HITL 恢复状态
HITL之后人为补充信息，然后重新执行检索，重新执行检索时需要第一次检索的信息：
- 问题：question
- 为什么停（用于判断下一步应该执行什么操作：选择/补充）：route 、retrieval_status
- 之前是否重写过（防止恢复后又无限重写）：rewrite_count
- 之前的问题规划结果（复杂度分析就不需要再做了）：complexity、complexity_reason、sub_questions
将这些信息包装到一个类里面，这个类的定义会写道另一个文件
```python
'''
class HitlResumeState(StrictSchema):
    question: str = Field(min_length=1)
    route: Literal["clarify", "scope_select"]
    retrieval_status: Literal["needs_clarification", "needs_scope_selection"]
    rewrite_count: int = Field(default=0, ge=0)
    complexity: Optional[Literal["simple", "complex"]] = None
    complexity_reason: Optional[str] = None
    sub_questions: List[str] = Field(default_factory=list, max_length=4)
'''

def _build_hitl_resume_state(result: dict) -> dict:
    trace = result.get("rag_trace") or {}
    return HitlResumeState(
        question=result.get("question") or trace.get("query") or "",
        route=result.get("route") or trace.get("route"),
        retrieval_status=result.get("retrieval_status") or trace.get("retrieval_status"),
        rewrite_count=int(result.get("rewrite_count") or 0),
        complexity=result.get("complexity") or trace.get("complexity"),
        complexity_reason=result.get("complexity_reason") or trace.get("complexity_reason"),
        sub_questions=result.get("sub_questions") or trace.get("sub_questions") or [],
    ).model_dump()
```

### 2.1.3 RAG 恢复入口
针对HITL情况，人为补充信息之后要恢复之前的RAG状态：
1. 恢复RAG State
2. 手动调用第二次检索（这里是手动调用，跟第一次有所不同，这次也不会再分子问题）
3. 如果仍需要 HITL，再保存新的恢复状态
```python
def resume_rag_from_hitl(
    resume_state: dict,
    user_answer: str,
    ctx: ChatRequestContext,
) -> dict:
    """Resume a paused RAG run from the HITL breakpoint without re-entering the main graph."""
    # 1. 根据上一轮 HITL 状态恢复 State
    state = _state_from_resume(resume_state, user_answer, ctx)
    _emit(state, "▶️", "收到 HITL 补充，继续原 RAG 流程", user_answer)

    # 2. 根据用户补充信息重新检索和评分
    state = _retrieve_resume_query(state)
    # 3. 如果仍需要 HITL，再保存新的恢复状态
    if _is_hitl_result(state):
        state["hitl_resume_state"] = _build_hitl_resume_state(state)
    return state
```

#### 1）恢复RAG State

|字段|来源|是否恢复旧值|
|---|---|---|
|`question`|`refined_question`|否，变成“原问题 + 用户补充”|
|`query`|`refined_question`|否，重新设置|
|`rewrite_count`|`HitlResumeState`|是|
|`complexity`|`HitlResumeState`|是|
|`complexity_reason`|`HitlResumeState`|是|
|`sub_questions`|`HitlResumeState`|是|
|`rag_trace`|重新构造|部分继承 + 新增恢复信息|
|其他字段|`_initial_state()`|否，全部重新初始化|
```python
def _refined_question_for_hitl(resume_state: dict, user_answer: str) -> str:
    # 得到补充后的query
    question = resume_state.get("question") or ""
    answer = user_answer.strip()
    if not question:
        return answer
    if answer and answer in question:
        return question
    return f"{answer}：{question}" if answer else question

def _state_from_resume(
    resume_state: dict,
    user_answer: str,
    ctx: ChatRequestContext,
) -> dict:
    current_resume_state = HitlResumeState.model_validate(resume_state).model_dump()
    refined_question = _refined_question_for_hitl(current_resume_state, user_answer)
    rag_trace = {
        "tool_used": True,
        "tool_name": "search_knowledge_base",
        "query": refined_question,
        "hitl_resumed": True,
        "hitl_answer": user_answer,
        "hitl_resume_from_status": current_resume_state["retrieval_status"],
        "hitl_resume_from_route": current_resume_state["route"],
    }
    if current_resume_state.get("complexity"):
        rag_trace["complexity"] = current_resume_state["complexity"]
    if current_resume_state.get("complexity_reason"):
        rag_trace["complexity_reason"] = current_resume_state["complexity_reason"]
    if current_resume_state.get("sub_questions"):
        rag_trace["sub_questions"] = current_resume_state["sub_questions"]
    state = _initial_state(refined_question, ctx)
    state.update({
        "query": refined_question,
        "rewrite_count": current_resume_state["rewrite_count"],
        "complexity": current_resume_state.get("complexity"),
        "complexity_reason": current_resume_state.get("complexity_reason"),
        "sub_questions": current_resume_state.get("sub_questions") or [],
        "rag_trace": rag_trace,
    })
    return state
```

#### 2）HITL 针对性检索
1. 使用用户补充后的 `refined_question` 重新调用 `检索节点`，得到新的 `docs` 和 `context`。
2. 更新 State。
3. 手动调用 `评分节点` 重新评估证据，得到新的 `retrieval_status` 和 `route` 后返回。
```python
def _retrieve_resume_query(state: dict) -> dict:
    _emit(state, "🔎", "使用 HITL 补充进行针对性检索", "跳过复杂度判断与子问题分解")
    query = state["question"]
    # 重新检索
    retrieved = retrieve_documents(query, top_k=RETRIEVAL_TOP_K)
    results = retrieved.get("docs", [])
    retrieve_meta = retrieved.get("meta", {})
    context = _format_docs(results)
    _emit(
        state,
        "🧱",
        "HITL 三级分块检索",
        (
            f"叶子层 L{retrieve_meta.get('leaf_retrieve_level', 3)} 召回，"
            f"候选 {retrieve_meta.get('candidate_k', 0)}"
        ),
    )
    _emit(
        state,
        "🧩",
        "Auto-merging 合并",
        (
            f"启用: {bool(retrieve_meta.get('auto_merge_enabled'))}，"
            f"应用: {bool(retrieve_meta.get('auto_merge_applied'))}，"
            f"替换片段: {retrieve_meta.get('auto_merge_replaced_chunks', 0)}"
        ),
    )
    _emit(state, "✅", f"HITL 针对性检索完成，找到 {len(results)} 个片段", f"模式: {retrieve_meta.get('retrieval_mode', 'hybrid')}")
    rag_trace = state.get("rag_trace") or {}
    rag_trace.update({
        "tool_used": True,
        "tool_name": "search_knowledge_base",
        "query": query,
        "retrieved_chunks": results,
        "hitl_targeted_retrieved_chunks": results,
        "hitl_resumed": True,
        "hitl_resume_strategy": "targeted_retrieval",
        "retrieval_stage": "hitl_targeted_retrieval",
        **retrieval_trace_fields(retrieve_meta),
    })
    state.update({
        "query": query,
        "docs": results,
        "context": context,
        "rag_trace": rag_trace,
    })
    # 更新 State，并重新做证据评分
    state.update(grade_documents_node(state))
    return state
```

# 3. 模块测试
代码略