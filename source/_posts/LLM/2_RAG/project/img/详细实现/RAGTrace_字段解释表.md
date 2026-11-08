# RAGTrace 字段解释表 · SuperMew-0.1.0

配套流程图：[[图_4RAG工作流模块 1]]。本文依据本地 v0.1.0 的 pipeline.py、utils.py、schemas/chat.py 和 request_context.py；不按最新版或理想设计推测。

## 1. 先分清四种对象

| 对象 | 所在位置 | 应该怎样读 |
| --- | --- | --- |
| S：RAGState | 节点之间的运行状态 | 保存 question、docs、context、改写预算、并行任务、ctx，以及 S.rag_trace。复杂度判断更新这里。 |
| T：原始 rag_trace | S.rag_trace | 是逐节点构造/修改的字典，开始为 None。**复杂度判断的返回值不是 Trace**。 |
| 对外 Trace | ctx/Service 交换或保存前的 normalize_rag_trace(T) | 只保留协议字段，去除 None；不是把所有 59 个键补齐。false、0、空串、空列表会保留。 |
| 恢复快照 | result.hitl_resume_state，随后可存入 pending_hitl.resume_state | 在 Trace 外部，保存下一轮重新检索所需信息，不保存 docs/context。 |

字段范围：**59 个协议字段 + 1 个 raw-only 的 rewrite_count = 60 个外层可能字段**。不是每份 Trace 都有 60 个键。下面另列证据元素的 16 个运行时可能字段、恢复快照的 7 个字段。

读图符号：Q=普通原问题；qᵢ=子问题；q*=合并用户补充的问题；D₀/D₁/Dₕ=初检/改写/恢复检索证据；DΣ=合成后的去重证据。`缺席`表示未写键，`null`表示原始字典显式 None，`[]`是明确空列表。节点框写新增/覆盖项，其余沿用上一步。

**R 组**是下文 G/H/I 三类的 26 个检索 meta 字段；白名单会跳过 None，因此实际键数可少于 26。**GRADE 组**是 route、retrieval_status、五项 evidence_*、missing_slots、hitl_prompt、hitl_options，共 10 项。

## 2. 按节点看 Trace 怎么变化

| 节点 | S 的关键变化 | T 的新增/覆盖及状态 |
| --- | --- | --- |
| N0 `_initial_state` | question=query=Q；rewrite_count=0；证据为空 | rag_trace=None。 |
| N1 `classify_complexity` | 本地规则写 simple/理由；模型写 simple/complex、理由和子问题列表 | **不写 T，仍为 None**。本地分支的 S.sub_questions 沿用 None；模型 simple 分支显式写 []。 |
| C1 `prepare_sub_questions` / Send | 整理父规划；每题新 S，is_sub_agent=true、rewrite_count=0、complexity=None | 父 T 仍 None；每个新子 S 的 T 也为 None，不继承父复杂度。 |
| S1 / C2 `retrieve_initial` | docs/context 为本题检索结果 | 新建 T：工具身份、query、本次/初检证据、stage=initial、complexity/理由、R。尚无评分、route、status。简单题复制 simple；子题复制 null。 |
| S2 / C3 `grade_documents_node` | 更新 route/status；HITL/无知识时清 docs/context | 覆盖 GRADE。stage 不变；HITL 清 retrieved_chunks，保留初检快照；no_knowledge 不主动清 T 中证据。 |
| S3 `rewrite_question_node` | 预算从 0→1，保存新 Query/方法 | 加 method、rewritten_query、raw rewrite_count=1，以及唯一方法的辅助文本；**route/status/评分仍为上次值，stage 仍 initial**。 |
| S4 `retrieve_rewritten` | 替换 docs/context；S.query 仍为 Q | 改写结果覆盖 retrieved_chunks、增加 rewrite_retrieved_chunks，stage=rewritten；覆盖 R。保留初检快照、Q 和旧评分，等待再评分。 |
| S5 第二次评分 | rewrite_count=1，不能再正常进入重写 | GRADE 覆盖为新评分；可能 answer / clarify / scope_select / no_knowledge。stage=rewritten。 |
| C4 `rag_sub_agent` 返回 | sub_results 中收集本题 docs/status/route/Trace | T 是本题“初检→评分”的结果；子题不走重写。子题 partial 可以 route=answer。 |
| C5 `synthesis` | 只接受 answerable/partial 子证据，保序去重 | 建总 T：stage=synthesis、complex、父规划、计数、规范化 sub_traces、派生评价和 route/status。**不汇总 R 或初检/改写快照，不再次 LLM 评分**。 |
| O `run_rag_graph` 结束 | 需要 HITL 时给结果顶层附快照 | 不把 hitl_resume_state 塞入 T；Tool 随后存 ctx 时规范化 T。 |
| H0 `_state_from_resume` | 校验快照、合并补充，恢复预算；is_sub_agent=false | 建新 T：工具身份、query=q*、hitl_resumed/answer/from_*；快照复杂度/理由/子规划有值才带回。此时尚无 stage、证据、评分。 |
| H1 `_retrieve_resume_query` | 用 q* 重新检索，更新 docs/context | 增加当前/针对性证据、R、stage=hitl_targeted_retrieval、strategy=targeted_retrieval，然后立即评分。 |
| H2 恢复评分结束 | 仅检索+评分，没有接回完整图 | 覆盖 GRADE；仍需 HITL 则在顶层附新快照。**route=rewrite 也可能出现，但本函数不会继续执行重写节点**。 |

分类模型必须给复杂问题至少一个有效子问题，否则源码抛 ValueError；提示词建议 2–4 个，校验代码实际允许 1 个。缺少模型或模型输出不合规也会抛异常；没有额外的 trace.error 键。

## 3. 评分后的五种路由

_resolve_route 有优先顺序：无证据/不相关 → 缺条件/多方向 → 已被明确支持的回答 → 子题规则 → 主题改写预算/部分证据规则 → 其余回答或无知识。因此模型建议 route 不一定等于最终 T.route。

| 最终 route | retrieval_status | 证据和后续动作 |
| --- | --- | --- |
| answer | sufficient → answerable；partial → partial | 保留 S.docs/context 和 T.retrieved_chunks。子题只要 partial/sufficient 且未命中更高优先级规则，就可保留给合成。 |
| rewrite | needs_rewrite | 普通主图进入一次改写；保留旧证据和评分，直到重检、再评分覆盖。空召回直接 no_knowledge，不会仅因“空结果”而改写。 |
| clarify | needs_clarification | 结束本次图执行，清 S.docs/context 与最终 retrieved_chunks，生成追问；阶段快照仍留着。 |
| scope_select | needs_scope_selection | 同上，需要用户选方向；不是 select 这个字符串。 |
| no_knowledge | no_knowledge | 清 S.docs/context。若此前召回过片段，T.retrieved_chunks 可以仍非空，不能把它直接当可回答证据。 |

空证据短路的固定值：relevance=none、answerability=none、ambiguity=none、confidence=1.0、reason=no_retrieved_documents、route/status=no_knowledge。模型不被调用。

主图的一次改写预算用完后，partial 通常转 clarify；明确建议 rewrite 但不再有预算时，partial 转 clarify，其他转 no_knowledge。节点内部另有 rewrite_budget_exhausted 防御分支；正常路由会避免再次进入重写，不能把该防御分支画成主图新增的正常终止边。

## 4. 复杂问题合成的四种终态

| 总证据 DΣ | 子状态条件 | 总 route / status | 顶层评价 |
| --- | --- | --- | --- |
| 非空 | 任一子状态为 partial | answer / partial | strong / partial |
| 非空 | 没有 partial | answer / answerable | strong / sufficient；即使其他子问题是 HITL/no_knowledge，也不会要求全部成功才回答。 |
| 空 | 存在子 HITL | clarify / needs_clarification，或 scope_select / needs_scope_selection | none / none；scope_select 优先，合并全部子 HITL 追问与选项。 |
| 空 | 没有子 HITL | no_knowledge / no_knowledge | none / none |

上述 relevance / answerability 由代码派生。总 Trace 的 evidence_confidence 显式为 null（规范化后缺席）；evidence_ambiguity、evidence_reason、missing_slots 在正常总 Trace 中不写。各子问题的评分和检索计数保存在 sub_traces 中。synthesis_merged_count 是去重前条数，最终片段数看 len(retrieved_chunks)。

## 5. 全量外层字段分类表

### A · 工具身份与当前阶段

| 字段 | 类型 | 取值与含义 | 首次/主要写入位置 |
| --- | --- | --- | --- |
| `tool_used` | bool | 流水线写 `true`。未调用知识工具时通常没有整份 Trace，并非固定写 false。 | 初检、合成、恢复 |
| `tool_name` | str | 固定 `search_knowledge_base`。 | 初检、合成、恢复 |
| `query` | str | 普通链为原问题 Q；重写后仍是 Q，实际检索词另见 rewritten_query。子 Trace 为子问题 qᵢ；恢复为合并用户补充后的 q*。 | 初检、合成、恢复 |
| `retrieval_stage` | str | `initial` / `rewritten` / `synthesis` / `hitl_targeted_retrieval`。评分不改变阶段。恢复刚重建时尚未设置。 | 各检索/合成节点 |

### B · 复杂度与子问题

| 字段 | 类型 | 取值与含义 | 首次/主要写入位置 |
| --- | --- | --- | --- |
| `complexity` | str 或 null | `simple` / `complex`。分类先只写 S；初检才复制入 T。子任务新建状态，原始子 Trace 写 null，规范化后该键消失。 | 初检、合成；恢复有值才复制 |
| `complexity_reason` | str 或 null | 分类模型理由，或 `obvious_simple_fast_path:single_fact_marker` / `obvious_simple_fast_path:short_single_intent`。子 Trace 原始值为 null。 | 同 complexity |
| `sub_questions` | list[str] | 父问题规划列表，最多 4 个；分类时只在 S。普通简单初检不写此键；合成写父列表，恢复仅在快照列表非空时写。源码允许 1 个，提示词才要求 2–4 个。 | 合成、恢复 |
| `sub_agent_count` | int | 实际汇总的 `len(sub_results)`，不是推测的规划数量。正常复杂链通常 1–4 个。 | 仅合成顶层 |
| `synthesis_merged_count` | int | 被接受子证据的数量之和，**去重前**计数；不是最终 len(retrieved_chunks)。 | 仅合成顶层 |
| `sub_traces` | list[RagSubTrace] | 按 sub_results 顺序收集并规范化的子 Trace。每份子 Trace 支持 58 个公共协议字段，不递归嵌套 sub_traces。 | 仅合成顶层 |

### C · 不同阶段的证据集合

| 字段 | 类型 | 取值与含义 | 首次/主要写入位置 |
| --- | --- | --- | --- |
| `retrieved_chunks` | list[dict] | 当前 Trace 的证据集合。初检/重检/恢复被替换，合成写去重结果。HITL 会清空；**no_knowledge 不主动清空此键**，可能保留被否决的召回片段。 | 检索、合成、评分 |
| `initial_retrieved_chunks` | list[dict] | 初检快照 D₀，重写后保留。HITL 清最终证据时仍保留它。普通复杂顶层不汇总此字段，子 Trace 各自拥有。 | 仅 retrieve_initial |
| `rewrite_retrieved_chunks` | list[dict] | 重写查询得到的证据快照 D₁；不存在重写则缺席。再次 HITL 时仍可保留此快照。 | 仅 retrieve_rewritten |
| `hitl_targeted_retrieved_chunks` | list[dict] | 恢复后针对性检索快照 Dₕ；HITL 再次清最终证据时仍保留它。 | 仅恢复检索 |

### D · 最终路由与证据评分

| 字段 | 类型 | 取值与含义 | 首次/主要写入位置 |
| --- | --- | --- | --- |
| `route` | str | `answer` / `rewrite` / `clarify` / `scope_select` / `no_knowledge`。是 _resolve_route 校正后的结果；原始模型建议没有单独存一份。 | 评分；合成会重建 |
| `retrieval_status` | str | `answerable` / `partial` / `needs_rewrite` / `needs_clarification` / `needs_scope_selection` / `no_knowledge`。须与 route 联合阅读。 | 评分；合成会重建 |
| `evidence_relevance` | str | `none` / `weak` / `strong`。无证据评分固定 none；合成按是否有可用总证据直接写 strong 或 none。 | 评分、合成 |
| `evidence_answerability` | str | `none` / `partial` / `sufficient`。子问题 partial 可以保留给合成；合成仅按总证据和子状态推导，不再调用评分模型。 | 评分、合成 |
| `evidence_ambiguity` | str | `none` / `missing_slot` / `multiple_candidates`。无证据固定 none。正常合成顶层不写，不代表已经评为 none。 | 评分；在子 Trace 可见 |
| `evidence_confidence` | float 或 null | 评分模型 0–1；无片段短路固定 1.0，表示“无片段判断”的确定性。合成明确写 null，规范化后消失，不能当作 0。 | 评分、合成 |
| `evidence_reason` | str | 模型理由；无片段固定 `no_retrieved_documents`。改写预算防御分支可写 `rewrite_budget_exhausted`。正常合成顶层不写。 | 评分、改写防御分支 |
| `missing_slots` | list[str] | 评分模型指出的缺失条件，可能为空列表。合成不合并各子问题的缺失条件，正常顶层缺席。 | 评分 |

### E · 查询改写

| 字段 | 类型 | 取值与含义 | 首次/主要写入位置 |
| --- | --- | --- | --- |
| `rewrite_method` | str | `step_back` / `hyde`，一次只采用一种；未执行改写时缺席。 | 改写、重检 |
| `rewritten_query` | str | 真正传给第二次检索的 Query，由原问题加退步问题或 HyDE 文档拼成。不要用 query 判断是否已换检索词。 | 改写、重检 |
| `step_back_question` | str | 只在 Step-back 路径写入非空抽象问题；另一方法的键不凭空补空字符串。 | Step-back 改写/重检 |
| `hyde_document` | str | 只在 HyDE 路径写入假设文档；仅辅助检索，不能当真实 Evidence。 | HyDE 改写/重检 |
| `rewrite_count` | int；**仅运行时** | 正常 raw Trace 在改写节点才写 1；初始预算 0 在 S，不自动出现在 T。该键未声明于 RagTrace，normalize_rag_trace 会过滤；跨轮预算另保存在恢复快照。 | rewrite_question_node；规范化后移除 |

### F · HITL 追问与跨轮恢复

| 字段 | 类型 | 取值与含义 | 首次/主要写入位置 |
| --- | --- | --- | --- |
| `hitl_prompt` | str | clarify / scope_select 用模型追问或默认追问；其他评分路由写空串。合成需要 HITL 时将各子追问按首次出现去重，再用“；”连接。 | 评分、合成 |
| `hitl_options` | list[str] | 评分采用模型选项，**源码未强制非 HITL 路由清空它**。合成需要 HITL 时跨子 Trace 去重汇集，否则 []。 | 评分、合成 |
| `hitl_resumed` | bool | 恢复路径写 true；普通链通常不写该键，不是固定 false。 | 恢复状态、恢复检索 |
| `hitl_answer` | str | 本轮用户实际补充文本，保留原输入。 | 恢复状态 |
| `hitl_resume_strategy` | str | 固定 `targeted_retrieval`；表示只重建状态后针对性检索、评分，不重新分解原问题。 | 恢复检索 |
| `hitl_resume_from_status` | str | 快照中的 `needs_clarification` / `needs_scope_selection`；记录从什么状态恢复。 | 恢复状态 |
| `hitl_resume_from_route` | str | 快照中的 `clarify` / `scope_select`；不是本轮评分后的最终 route。 | 恢复状态 |

### G · 召回参数与数量（R 组之一）

| 字段 | 类型 | 取值与含义 | 首次/主要写入位置 |
| --- | --- | --- | --- |
| `retrieval_pipeline` | str | 固定 `recall_merge_rerank`，指检索内部“召回→合并→精排/过滤”，不是 LangGraph 路由。 | 检索 meta → Trace |
| `retrieval_mode` | str | `hybrid`：混合成功；`dense_fallback`：混合召回或后处理异常后 Dense 成功；`failed`：编码失败或两条尝试最终失败。空结果也可以是 hybrid，并非一定 failed。 | 检索 meta → Trace |
| `candidate_k` | int | Milvus 交给后处理的候选池目标数量，至少为最终 Top-K；默认 top_k × 3。不是已经返回的实际数量。 | 同上 |
| `candidate_k_source` | str | `env` 或 `multiplier`；显式候选数解析失败也回到 multiplier。 | 同上 |
| `candidate_k_config_error` | str | 仅配置错误时写 `invalid RETRIEVAL_CANDIDATE_K`；正常不写。 | 同上 |
| `retrieval_candidate_multiplier` | int | 候选池倍数，默认 3；即使采用 env 候选数，也会记录配置倍数。 | 同上 |
| `retrieval_top_k` | int | 精排/排序后最多取多少条，默认 8；随后阈值过滤，最终可能少于 K。 | 同上 |
| `recall_count` | int | 这一次 Hybrid / Dense 实际返回的原始候选数量，合并前。失败时为 0。 | 同上 |
| `post_merge_candidate_count` | int | 两级父块合并后、精排前的候选数量。 | 同上 |
| `candidate_count` | int | 交给精排模块的候选数量，正常等于 post_merge_candidate_count。 | 同上 |
| `leaf_retrieve_level` | int | 召回过滤的 Chunk 层级，默认 3；合并后证据可成为 L2 / L1。 | 同上 |

### H · Auto-Merging（R 组之一）

| 字段 | 类型 | 取值与含义 | 首次/主要写入位置 |
| --- | --- | --- | --- |
| `auto_merge_enabled` | bool | 是否配置启用父块合并，默认 true；不等于本次实际发生替换。 | 检索 meta → Trace |
| `auto_merge_applied` | bool | 本次至少一层处理了可替换子块时为 true；启用但不满足阈值时仍为 false。 | 同上 |
| `auto_merge_threshold` | int | 同父候选触发合并的数量阈值，默认 2。读取代码未做正数钳制。 | 同上 |
| `auto_merge_replaced_chunks` | int | 两层累计处理的被替换子块条目数，**不是候选净减少量**；同一条证据的不同层级可再次计数。 | 同上 |
| `auto_merge_steps` | int | 实际应用替换的层数，0 / 1 / 2，分别对应 L3→L2 和 L2→L1。不是检索次数。 | 同上 |

### I · 精排、阈值与失败信息（R 组之一）

| 字段 | 类型 | 取值与含义 | 首次/主要写入位置 |
| --- | --- | --- | --- |
| `rerank_enabled` | bool | Host、模型名和 API Key 都有效才启用精排；示例占位符视为未配置。 | 检索 meta → Trace |
| `rerank_applied` | bool | 发出精排请求前就置 true，表示“已尝试”，**不保证成功**；须结合 rerank_error。 | 同上 |
| `rerank_model` | str | 精排模型名称；未配置为 None，经白名单过滤后通常不出现在 Trace。 | 同上 |
| `rerank_endpoint` | str | 精排接口，通常 Host + /v1/rerank；未配置可为保留的空字符串。 | 同上 |
| `rerank_error` | str | 可为 HTTP 错误、`empty_rerank_results` 或异常文本；编码/总体检索失败也借此键写 `embedding_failed` / `retrieve_failed`。成功 None 会被过滤。 | 同上 |
| `rerank_timeout_seconds` | float | 精排请求超时秒数，默认 5，至少 0.1。 | 同上 |
| `rerank_min_score` | float | 最终有效分数阈值，默认 0。有效分数优先 rerank_score，否则 score；即使未启用精排也会过滤。 | 同上 |
| `post_rerank_count` | int | 精排或原分排序截取 Top-K 后、阈值过滤前的数量。 | 同上 |
| `post_threshold_count` | int | 阈值过滤后的数量；是该次检索输出量，HITL 后清最终证据不会回写此计数。 | 同上 |
| `retrieval_empty` | bool | 该次检索后处理结果是否为空，不等同最终 route=no_knowledge；初检后被评分清空证据也不会重算此值。 | 同上 |

## 6. 证据列表内部的字段

以下适用于 retrieved_chunks、initial_retrieved_chunks、rewrite_retrieved_chunks、hitl_targeted_retrieved_chunks。在运行时四者都可包含原始业务字典；对外统一经过 RetrievedChunk 过滤。

| 字段 | 含义 | 规范化后 |
| --- | --- | --- |
| filename | 来源文件名，协议要求非空来源才进入规范化列表 | 保留 |
| page_number | 来源页码，str/int | 保留 |
| text | 片段正文 | 保留 |
| score | Hybrid 融合分或 Dense IP 分；父块可继承子块最高分 | 保留 |
| rrf_rank | 精排前按合并后候选位置编号；Dense 路径也有该键，不保证发生过 RRF；合成后按最终顺序重编号 | 保留 |
| rerank_score | Rerank 的 relevance_score | 有值保留 |
| id | Milvus 的自动主键，不等于业务 chunk_id；父块可能没有 | 过滤 |
| file_type | 来源文件类型 | 过滤 |
| file_path | 父块 Store 可能带回的原文件路径 | 过滤 |
| chunk_idx | 分块顺序号 | 过滤 |
| chunk_id | 业务分块 ID | 过滤 |
| parent_chunk_id | 上一级分块 ID | 过滤 |
| root_chunk_id | L1 根块 ID | 过滤 |
| chunk_level | L1/L2/L3 层级 | 过滤 |
| merged_from_children | 父块是否由命中子块替换得到 | 过滤 |
| merged_child_count | 该父块对应的命中子块数量 | 过滤 |

因此，对外 Trace 只能看到六种证据字段，不能仅凭它恢复完整 Chunk 图。raw 中的文件路径、层级、ID 和合并标记不是外层 Trace 协议字段。

## 7. 恢复快照的七个字段（在 Trace 外面）

| 字段 | 值/含义 |
| --- | --- |
| question | 当前任务问题，至少 1 字符；再次 HITL 时可能已经包含此前用户补充 |
| route | clarify / scope_select |
| retrieval_status | needs_clarification / needs_scope_selection |
| rewrite_count | 非负；主图实际正常最多一次，跨轮沿用预算 |
| complexity | simple / complex 或 None |
| complexity_reason | 分类理由或 None |
| sub_questions | 规划列表，最多 4 个 |

快照存在 result.hitl_resume_state；ctx 的暂存包可同时装 `{rag_trace, hitl_resume_state}`；Service 可将快照装入 pending_hitl.resume_state。包里的键不要误算成 rag_trace 字段。

恢复时，即使快照保留 complexity=complex / sub_questions，也只执行针对性检索和评分，不重新 Send、合成。若本次评分是 rewrite，恢复函数直接返回 needs_rewrite；它没有执行 rewrite_question_node，不能在图里接一条自动重写回路。

## 8. 容易混淆、但不属于 Trace 外层的状态

| 名称 | 所属位置 |
| --- | --- |
| question | S 原问题 / 快照问题；T 用 query 表示问题文本 |
| docs、context | S 中真正交给后续使用的证据和格式化上下文；no_knowledge 时可与原始 Trace 快照不同 |
| is_sub_agent、sub_results | S 中的子任务标记和并行归并结果；Trace 对外用 sub_traces |
| request_context、rag_step_group、rag_step_group_label | S 的运行时 ctx 和事件分组，不存入 Trace |
| rag_trace | S 的一个字段，本身不是 Trace 中再套一层 rag_trace |
| hitl_resume_state | 结果/ctx 暂存包的并列快照，不是 T 的子键 |
| elapsed_ms、stage_elapsed_ms、icon、label、detail | rag_step SSE 事件字段，不是 rag_trace 字段 |

## 9. 核对依据与验证范围

- [节点与路由](C:/Users/123/Desktop/file/job/project/SuperMew-0.1.0/backend/rag/pipeline.py:225)。
- [检索 meta 与字段白名单](C:/Users/123/Desktop/file/job/project/SuperMew-0.1.0/backend/rag/utils.py:76)。
- [Trace 协议与规范化](C:/Users/123/Desktop/file/job/project/SuperMew-0.1.0/backend/schemas/chat.py:26)。
- [ctx 规范化及暂存](C:/Users/123/Desktop/file/job/project/SuperMew-0.1.0/backend/chat/request_context.py:97)。

已用原节点函数和固定的模型/检索响应检查 26 份状态快照，覆盖本地/模型分类、五种路由、两种改写、子题 partial、合成去重/无知识/HITL 和恢复返回 rewrite 的情况；未调用真实模型 API、Milvus 或 PostgreSQL，不把演示响应当实际模型效果。
