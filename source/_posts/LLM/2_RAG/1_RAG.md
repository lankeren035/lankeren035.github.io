---
title: 1_RAG
date: 2026-09-11
tags:
  - llm
categories:
  - llm
comment: true
toc: true
published: true
permalink: llm/2_RAG
hexo-path:
---
#
<!--more-->
参考：https://datawhalechina.github.io/all-in-rag/#/chapter1/01_RAG_intro
# 1. RAG 简介
## 1.1 什么是 RAG
### 1.1.1 定义
- RAG（Retrieval-Augmented Generation）将模型内部学到的“**参数化知识**”（模型权重中固化的、模糊的“记忆”），与来自外部知识库的“**非参数化知识**”（精准、可随时更新的外部数据）相结合。在 LLM 生成文本前，先通过检索机制从外部知识库中动态获取相关信息，并将这些“参考资料”融入生成过程，从而提升输出的准确性和时效性。
### 1.1.2 原理
- 它通过两个阶段来实现目的：检索和生成
#### 1）检索
1. **知识向量化**：通过**嵌入模型（Embedding Model）** 将外部知识库编码为向量索引（Index），存入**向量数据库**。
2. **语义召回**：当用户发起查询时，检索模块利用**同样的嵌入模型**将问题向量化，并通过**相似度搜索（Similarity Search）**，从海量数据中精准锁定与问题最相关的文档片段。
#### 2）生成
1. **上下文整合**：得到：
	- 检索片段
	- 用户原始问题
- **指令引导生成**：遵循预设的 **Prompt** 指令，将上下文与问题有效整合，并引导 LLM（如 DeepSeek）进行可控的、有理有据的文本生成。
### 1.1.3 技术演进
- 其演进可分为三个阶段：
![](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/llm/RAG.png)


|          | 初级 RAG                              | 高级 RAG                                                 | 模块化 RAG                                                                     |
| -------- | ----------------------------------- | ------------------------------------------------------ | --------------------------------------------------------------------------- |
| **流程**   | **离线:** `索引`  <br>**在线:** `检索 → 生成` | **离线:** `索引`  <br>**在线:** `...→ 检索前 → ... → 检索后 → ...` | 积木式可编排流程                                                                    |
| **特点**   | 基础线性流程                              | 增加**检索前后**的优化步骤                                        | 模块化、可组合、可动态调整                                                               |
| **关键技术** | 基础向量检索                              | **查询重写（Query Rewrite）**  <br>**结果重排（Rerank）**          | **动态路由（Routing）**  <br>**查询转换（Query Transformation）**  <br>**多路融合（Fusion）** |
| **局限性**  | 效果不稳定，难以优化                          | 流程相对固定，优化点有限                                           | 系统复杂性高                                                                      |
> “离线”指提前完成的数据预处理工作（如索引构建）；“在线”指用户发起请求后的实时处理流程。

## 1.2 为什么用RAG
### 1.2.1 RAG vs. 微调
1. 在选择具体的技术路径时，一个重要的考量是成本与效益的平衡。通常，我们应优先选择对模型改动最小、成本最低的方案，所以技术选型路径往往遵循的顺序是**提示词工程（Prompt Engineering） -> 检索增强生成 -> 微调（Fine-tuning）**。
2. 我们可以从两个维度来理解这些技术的区别。如图所示，**横轴代表“LLM 优化”**，即对模型本身进行多大程度的修改。从左到右，优化的程度越来越深，其中提示工程和 RAG 完全不改变模型权重，而微调则直接修改模型参数。**纵轴代表“上下文优化”**，是对输入给模型的信息进行多大程度的增强。从下到上，增强的程度越来越高，其中提示工程只是优化提问方式，而 RAG 则通过引入外部知识库，极大地丰富了上下文信息。
![](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/llm/RAG_finetuen.png)

3. RAG 的出现填补了通用模型与专业领域之间的鸿沟：

|问题|RAG的解决方案|
|---|---|
|**静态知识局限**|实时检索外部知识库，支持动态更新|
|**幻觉（Hallucination）**|基于检索内容生成，错误率降低|
|**领域专业性不足**|引入领域特定知识库（如医疗/法律）|
|**数据隐私风险**|本地化部署知识库，避免敏感数据泄露|

### 1.2.2 优势
#### 1）准确性与可信度的提升

- 不仅能**补充专业领域的知识盲区**，还能通过提供具体的参考材料，有效**抑制幻觉**。论文研究还表明，RAG 生成的内容在**具体性**和**多样性**上也显著优于纯 LLM。更重要的是，RAG 具备**可溯源性**。

#### 2）时效性

- RAG 允许知识库独立于模型进行**动态更新**。这种能力在论文中被称为“索引热拔插”（Index Hot-swapping）。

#### 3）综合成本效益

- RAG 是一种高性价比的方案：
	- 避免了高频微调的巨额算力成本；
	- 由于有了外部知识，处理特定领域问题时，往往可以使用**参数量更小的基础模型**来达到类似的效果，从而直接降低了推理成本。
#### 4）灵活的模块化可扩展性

- RAG 的架构具备极强的包容性，支持**多源集成**，无论是 PDF、Word 还是网页数据，都能统一构建进知识库中。
- 其**模块化设计**实现了检索与生成的解耦，这意味着我们可以独立优化检索组件（比如更换更好的 Embedding 模型），而不会影响到生成组件的稳定性，便于系统的长期迭代。
### 1.2.3 适用场景风险分级

|风险等级|案例|RAG适用性|
|---|---|---|
|**低风险**|翻译/语法检查|高可靠性|
|**中风险**|合同起草/法律咨询|需结合人工审核|
|**高风险**|证据分析/签证决策|需严格质量控制机制|

## 1.3 如何上手 RAG
### 1.3.1 基础工具链

- 构建 RAG 系统通常涉及几个关键环节的选型。
	1. 在**开发模式**上，我们可以利用 **LangChain** 或 **LlamaIndex** 等成熟框架快速集成，**也可以选择不依赖框架的原生开发**，以获得对系统流程更精细的控制力（在 AI 编程辅助下这并非难事）。
	2. **记忆载体**（向量数据库）方面，既有 **Milvus**、**Pinecone** 等适合大规模数据的方案，也有 **FAISS**、**Chroma** 等轻量级或本地化的选择，需根据具体业务规模灵活决定。后期为了量化效果，还可以引入 **RAGAS** 或 **TruLens** 等自动化**评估工具**。

### 1.3.2 构建最小可行系统

1. **数据准备与清洗**：将 PDF、Word 等多源异构数据标准化，并采用合理的**分块策略**（如按语义段落切分而非固定字符数）。
2. **索引构建**：将切分好的文本通过**嵌入模型**转化为向量，并存入数据库。可以在此阶段关联**元数据**（如来源、页码）。
3. **检索策略优化**：不要依赖单一的向量搜索。可采用**混合检索**（向量+关键词）等方式来提升召回率，并引入**重排序**模型对检索结果进行二次精选。
4. **生成与提示工程**：设计清晰的 **Prompt 模板**，引导 LLM 基于检索结果回答用户问题，并明确要求模型“不知道就说不知道”，防止幻觉。

### 1.3.3 新手友好方案

- 可以尝试 **FastGPT** 或 **Dify** 这样的可视化知识库平台，它们封装了复杂的 RAG 流程，仅需上传文档即可使用。对于开发者，利用 **LangChain4j Easy RAG** 或 GitHub 上的 **TinyRAG** [6](https://datawhalechina.github.io/all-in-rag/#fn-6)等开源模板，也是高效的起手方式。

### 1.3.4 进阶与挑战
#### 1）评估维度与挑战

- **检索相关性**：找到的内容是否包含答案
- **生成质量**：可以细分为**语义准确性**（回答的意思是否正确）和**词汇匹配度**（专业术语是否使用得当）。

这些评估维度也直接对应了 RAG 当前面临的主要挑战。比如，**检索依赖性**问题——如果检索系统召回了错误信息，再强的 LLM 也会“一本正经地胡说八道”。此外，对于需要跨多个文档进行综合分析的**多跳推理**问题，常见的 RAG 架构也普遍感到吃力。

#### 2）优化方向与架构演进

- 针对上述挑战，社区探索出了多种优化路径。
	1. **性能层面**：通过**索引分层**（对高频数据启用缓存）和**多模态扩展**（支持图像/表格检索）来提升效率和能力边界。
	2. **架构层面**，简单的线性流程正在被更复杂的**设计模式**所取代。例如，系统可以通过**分支模式**并行处理多路检索，或通过**循环模式**进行自我修正，这些灵活的架构是通往更智能 RAG 的必由之路。
# 2. 环境配置
参考：https://datawhalechina.github.io/all-in-rag/#/chapter1/02_preparation

如果vscode连接codespace比较卡，可以终端连接：

```powershell
# 1. 安装 GitHub CLI
winget install --id GitHub.cli

# 2. 检查是否安装成功
gh --version

# 3. 登录 GitHub
gh auth login

# 登录时建议选择
# GitHub.com
# HTTPS
# Yes
# Login with a web browser

# 4. 如果 codespace 权限不足
gh auth refresh -h github.com -s codespace

# 5. 查看 Codespace
gh codespace list

# 6. 连接 Codespace
gh codespace ssh

# 7. 指定某个 Codespace
gh codespace ssh -c <codespace-name>

# 例如
gh codespace ssh -c friendly-potato-r5wqpr675qj2xrp6

# 8. 用 VS Code 打开 Codespace
gh codespace code

# 9. 指定某个 Codespace 用 VS Code 打开
gh codespace code -c <codespace-name>

# 10. 查看当前 GitHub 登录状态
gh auth status

# 11. 退出当前 GitHub 账号
gh auth logout
```

# 3. 构建RAG
## 3.1 使用LangChain 框架的 RAG
### 3.1.1 初始化

```python
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
```
### 3.1.2 数据准备
#### 1）加载原始文档
- 选择一个md文档，然后使用`TextLoader`加载
```python
# 1. 数据准备
## 1.1 加载文档
markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"
loader = TextLoader(markdown_path)
docs = loader.load()
```
#### 2）文本分块
- 这里采用**递归字符分割策略**`RecursiveCharacterTextSplitter()` ，其默认行为旨在**最大程度保留文本的语义结构**：
	- **默认分隔符与语义保留**: 按顺序尝试使用一系列预设的分隔符 `["\n\n" (段落), "\n" (行), " " (空格), "" (字符)]` 来递归分割文本。目的是尽可能保持段落、句子和单词的完整性，因为它们通常是语义上最相关的文本单元，直到文本块达到目标大小。
	- **保留分隔符**: 默认情况下 (`keep_separator=True`)，分隔符本身会被保留在分割后的文本块中。
	- **默认块大小与重叠**: 使用其基类 `TextSplitter` 中定义的默认参数 `chunk_size=4000`（块大小）和 `chunk_overlap=200`（块重叠）。这些参数确保文本块符合预定的大小限制，并通过重叠来减少上下文信息的丢失。
```python
## 1.2 文本分块
text_spliter = RecursiveCharacterTextSplitter()
texts = text_spliter.split_documents(docs)
```
### 3.3.3 构建索引
#### 1）初始化embedding模型
- 使用`HuggingFaceEmbeddings`。配置模型在CPU上运行，并启用嵌入归一化 (`normalize_embeddings: True`)。
```python
# 2. 索引构建
## 2.1 初始化embedding模型
embeddings = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-small-zh-v1.5",
    model_kwargs = {'device': 'cpu'},
    encode_kwargs = {'normalize_embeddings': True}
)
```
#### 2）构建向量存储
- 将分割后的文本块通过初始化好的**嵌入模型**转换为**向量表示**，然后使用`InMemoryVectorStore`将这些向量及其对应的原始文本内容添加进去，从而在内存中构建出一个向量索引。
```python
## 2.2 构建向量存储
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(texts)
```
### 3.3.4 查询与检索
#### 1）定义用户查询
```python
# 3. 查询与检索
## 3.1 定义用户查询
question = "文中举了哪些例子？"
```
#### 2）检索数据库
- 使用向量存储的`similarity_search`方法，根据用户问题在索引中查找最相关的 `k` (此处示例中 `k=3`) 个文本块。
```python
retrieved_docs = vectorstore.similarity_search(question, k=3)
```
#### 3）准备上下文
- 将检索到的多个文本块的页面内容合并成一个单一的字符串，并使用双换行符 (`"\n\n"`) 分隔各个块，形成最终的上下文信息 (`docs_content`) 供大语言模型参考。
```python
docs_content= '\n\n'.join(doc.page_content for doc in retrieved_docs)
```
> 使用 `"\n\n"` (双换行符) 而不是 `"\n"` (单换行符) 来连接不同的检索文档块，主要是为了在传递给大型语言模型（LLM）时，能够更清晰地在语义上区分这些独立的文本片段。双换行符通常代表段落的结束和新段落的开始，这种格式有助于LLM将每个块视为一个独立的上下文来源，从而更好地理解和利用这些信息来生成回答。

### 3.3.5 生成继承
```python

# 4. 生成集成
## 4.1 构建提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”
  
上下文:
{context}

问题: {question}

回答:"""
                                          )
## 4.2 配置大语言模型
llm = ChatOpenAI(
    model = "glm-4.7-flash-free",
    temperature=0.7,
    max_tokens=2048,
    api_key=os.getenv('AIHUBMIX_API_KEY'),
    base_url = "https://aihubmix.com/v1"
)
## 4.3 调用llm
answer = llm.invoke(prompt.format(question=question, context=docs_content,))
print(answer)
```
- 输出参数解析：
	- **`content`**: 最核心部分，llm生成的回答。
	- **`additional_kwargs`**: 包含一些额外的参数，在这个例子中是 `{'refusal': None}`，表示模型没有拒绝回答。
	- **`response_metadata`**: 包含了关于LLM响应的元数据。
	    - `token_usage`: 显示了本次调用消耗的token数量，包括完成（completion_tokens）、提示（prompt_tokens）和总量（total_tokens）。
	    - `model_name`: 使用的LLM模型名称，当前是 `deepseek-chat`。
	    - `system_fingerprint`, `id`, `service_tier`, `finish_reason`, `logprobs`: 这些是更详细的API响应信息，例如 `finish_reason: 'stop'` 表示模型正常完成了生成。
	- **`id`**: 本次运行的唯一标识符。
	- **`usage_metadata`**: 与 `response_metadata` 中的 `token_usage` 类似，提供了输入和输出token的统计。

## 3.2 使用Llamaindex构建 RAG
```python
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# 1. 初始化llm
Settings.llm = OpenAILike(
    model="glm-4.7-flash-free",
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    api_base="https://aihubmix.com/v1",
    is_chat_model=True
)

# 2. 初始化embedding模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 3. 加载文档
documents = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

# 4. 切分，向量化，建立索引
index = VectorStoreIndex.from_documents(documents)

# 5. 将索引对象包装成查询引擎
query_engine = index.as_query_engine()

print(query_engine.get_prompts())
print(query_engine.query("文中举了哪些例子?"))
```