---
title: 8_RGA项目实战
date: 2026-10-02
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

# 1. 项目背景
构建一个智能问答系统来解决我的选择困难症？每天面对"今天吃什么"这个世纪难题，如果有个AI助手能根据我的需求推荐菜品、告诉我怎么做，那该多好！

# 2. 环境配置

# 3. 代码设计
```
code
├── config.py                   # 配置管理
├── main.py                     # 主程序入口
├── requirements.txt            # 依赖列表
├── rag_modules/               # 核心模块
│   ├── __init__.py
│   ├── data_preparation.py    # 数据准备模块
│   ├── index_construction.py  # 索引构建模块
│   ├── retrieval_optimization.py # 检索优化模块
│   └── generation_integration.py # 生成集成模块
└── vector_index/              # 向量索引缓存（自动生成）
```

# 4. 数据分析
- 原始数据来自：https://github.com/Anduin2017/HowToCook。 数据特征如下：
	- 全是markdown文档
	- 文档结构比较固定
# 5. 代码实现
## 5.1 数据准备模块
### 5.1.1 类结构设计
- 该模块主要实现一个类：`DataPreparationModule`，然后提供一些数据处理函数。这个类需要几个数据对象：
	- `documents`: 存储完整的菜谱文档（父文档）
	- `chunks`: 存储按标题分割的小块（子文档）
	- `parent_child_map`: 维护父子关系映射作为元数据
创建`rag_modules/data_preparation.py`。初始化的时候用Document对象，这个对象里面预定义了两个属性（page_content和metadata）。
```python
from typing import List, Dict, Any
from langchain_core.documents import Document

class DataPreparationModule:
    '''
    数据准备模块，用于数据加载，清洗和预处理
    '''
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.documents: List[Document] = [] # 完整的一个文档
        self.chunk: List[Document] = [] # 文档中的一个chunk
        self.parent_child_map: Dict[str, str] = {} # chunk 到父文档的映射
```
### 5.1.2 文档加载
- 这部分加载文档之后还要补充元数据，因此提供两个函数
#### 1）批量加载markdown
```python
import logging
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

class DataPreparationModule:
    def load_documents(self) -> List[Document]:
        '''
        加载数据文件夹中的所有文档

        Returns:
            List[Document]: 加载的文档列表
        '''
        logger.info(f'正在从 {self.data_path} 加载文档...')

        # 读取markdown
        documents = []
        data_path_obj = Path(self.data_path)
        for md_file in data_path_obj.rglob('*.md'): # 递归查找
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 每个文档分配id （相对路径）
                try:
                    data_root = Path(self.data_path).resolve() # 绝对路径
                    relative_path = Path(md_file).resolve().relative_to(data_root).as_posix() # 去掉data_root前缀后转成字符串路径
                except Exception as e:
                    relative_path = Path(md_file).as_posix()
                parent_id = hashlib.md5(relative_path.encode('utf-8')).hexdigest() # 文档id, 编码后固定长度，不包含/，

                # 创建Document对象
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': str(md_file),
                        'parent_id': parent_id,
                        'doc_type': 'parent' #标记为父文档
                    }
                )
                documents.append(doc)
            except Exception as e:
                logger.warning(f'加载文档 {md_file} 失败: {e}')

        # 增强文档元数据
        for doc in documents:
            self._enhance_metadata(doc)

        self.documents = documents
        logger.info(f'成功加载 {len(documents)} 个文档.')
        return documents
```
#### 2）元数据增强
- 给文档对象增加元数据，比如：
	- 这个文档做的是什么菜品
	- 这个文档对应菜品的难度
```python
import re

class DataPreparationModule:
    '''
    数据准备模块，用于数据加载，清洗和预处理

    Args:
        data_path: 数据路径（文件夹）
    '''
        # 统一维护的分类与难度配置，供外部复用，避免关键词重复定义
    CATEGORY_MAPPING = {
        'meat_dish': '荤菜',
        'vegetable_dish': '素菜',
        'soup': '汤品',
        'dessert': '甜品',
        'breakfast': '早餐',
        'staple': '主食',
        'aquatic': '水产',
        'condiment': '调料',
        'drink': '饮品'
    }
    def _enhance_metadata(self, doc: Document):
        '''
        增强文档元数据

        Args:
            doc: 需要做增强的文档
        '''
        file_path = Path(doc.metadata.get('source', ''))
        path_parts = file_path.parts # 拆分路径层级得到一个元组

        # 提取菜品部分
        doc.metadata['category'] = '其他'
        for key, value in self.CATEGORY_MAPPING.items():
            if key in path_parts:
                doc.metadata['category'] = value
                break

        # 提取菜品名称
        doc.metadata['dish_name'] = file_path.stem # 文件名作为菜品名称

        # 分析难度等级（使用正则匹配看星号数量
        content = doc.page_content
        star_match = re.search(r'★+', content)
        if star_match:
            star_count = len(star_match.group())
            difficulty_map = {5: '非常困难', 4: '困难', 3: '中等', 2: '简单', 1: '非常简单'}
            doc.metadata['difficulty'] = difficulty_map.get(star_count, '未知')
        else:
            doc.metadata['difficulty'] = '未知'
```
### 5.1.3 文档分块
- 将markdown按照标题进行分块。该部分拆分为两个函数，一个作为分块主流程，一个用于实现分块的策略。
#### 1）分块处理主流程
```python
import uuid
class DataPreparationModule:
    def chunk_documents(self) -> List[Document]:
        '''
        Markdown结构感知分块

        Returns:
            List[Document]: 分块后的文档列表
        '''
        logger.info('正在对Markdown文档进行分块...')

        if not self.documents:
            raise ValueError("请先加载文档后再进行分块.")

        # 使用markdown标题分割器进行分割
        chunks = self._markdown_header_split()

        # 为每个chunk添加基础元数据
        for i, chunk in enumerate(chunks):
            if 'chunk_id' not in chunk.metadata:
                chunk.metadata['chunk_id'] = str(uuid.uuid4()) # 随机生成一个几乎不会重复的唯一 ID
            chunk.metadata['batch_index'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)

        self.chunks = chunks
        logger.info(f'成功分块为 {len(chunks)} 个chunk.')
        return chunks
```
#### 2）具体分块策略
- Markdown标题分割器
	- **三级标题分割**: 按照`#`、`##`、`###`进行层级分割
	- **保留标题**: 设置`strip_headers=False`，保留标题信息便于理解上下文
	- **父子关系**: 每个子块都记录其父文档的`parent_id`
	- **唯一标识**: 每个子块都有独立的`child_id`
```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

class DataPreparationModule:
    def _markdown_header_split(self) -> List[Document]:
        """
        使用Markdown标题分割器进行结构化分割

        Returns:
            按标题结构分割的文档列表
        """
        # 定义分割层级
        headers_to_split_on = [
            ("#", "主标题"),      # 菜品名称
            ("##", "二级标题"),   # 必备原料、计算、操作等
            ("###", "三级标题")   # 简易版本、复杂版本等
        ]

        # 创建makrdown分割器
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers = False,  # 保留标题, 便于理解上下文
        )

        all_chunks = []
        for doc in self.documents:
            try:
                # 对每个文档进行分割
                md_chunks = markdown_splitter.split_text(doc.page_content)

                logger.debug(f"文档 {doc.metadata.get('source', '未知')} 分割为 {len(md_chunks)} 个块")

                # 如果没有分割成功，说明文档可能没有标题
                if len(md_chunks) <=1:
                    logger.warning(f"文档 {doc.metadata.get('source', '未知')} 没有标题结构，无法进行Markdown分割")

                # 为每个分块简历文档联系
                parent_id = doc.metadata['parent_id']

                for i, chunk in enumerate(md_chunks):
                    # 分配chunk_id
                    child_id = str(uuid.uuid4())

                    # 合并文档元数据
                    chunk.metadata.update(doc.metadata)
                    chunk.metadata.update({
                        'chunk_id': child_id,
                        'parent_id': parent_id,
                        'doc_type': 'child',  # 标记为子文档
                        'chunk_index': i, #在父文档中的位置
                    })

                    # 简历父子映射关系
                    self.parent_child_map[child_id] = parent_id

                all_chunks.extend(md_chunks)
            except Exception as e:
                logger.warning(f"文档 {doc.metadata.get('source', '未知')} Markdown分割失败: {e}")
                # 如果Markdown分割失败，将整个文档作为一个chunk
                all_chunks.append(doc)
        logger.info(f"Markdown结构分割完成，生成 {len(all_chunks)} 个结构化块")
        return all_chunks
```

### 5.1.4 检索去重
- 当用户询问"宫保鸡丁怎么做"时，可能会检索到同一道菜的多个子块。比如检索到ck1和ck2都是对应同一个文档，此时这个parent文档我们应该只要load一次。
	1. **统计相关性**: 计算每个父文档被匹配的子块数量
	2. **按相关性排序**: 匹配子块越多的菜谱排名越靠前
	3. **去重输出**: 每个菜谱只输出一次完整文档
- 但是这里为什么设计在这个类里面？这里感觉有点不合理。
```python
class DataPreparationModule:
    def get_parent_document(self, child_chunks: List[Document]) -> List[Document]:
        """
        根据子块获取对应的父文档（智能去重）

        Args:
            child_chunks: 检索到的子块列表

        Returns:
            对应的父文档列表（去重，按相关性排序）
        """
        # 统计每个父文档被匹配的次数（相关性指标）
        parent_relevance = {}
        parent_docs_map = {}

        # 收集所有相关的父文档ID和相关性分数
        for chunk in child_chunks:
            parent_id = chunk.metadata.get('parent_id')
            if parent_id:
                # 增加相关性计数
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1
                
                # 保存父文档对象，避免重复加载
                if parent_id not in parent_docs_map:
                    # 查找对应的父文档
                    for doc in self.documents:
                        if doc.metadata.get('parent_id') == parent_id:
                            parent_docs_map[parent_id] = doc
                            break
        # 按相关性排序（匹配次数多的排在前面）
        sorted_parent_ids = sorted(parent_relevance.keys(),
                                 key=lambda x: parent_relevance[x],
                                 reverse=True)
        # 构建去重后的父文档列表
        parent_docs = []
        for parent_id in sorted_parent_ids:
            if parent_id in parent_docs_map:
                parent_docs.append(parent_docs_map[parent_id])

        # 收集父文档名称和相关性信息用于日志
        parent_info = []
        for doc in parent_docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            parent_id = doc.metadata.get('parent_id')
            relevance_count = parent_relevance.get(parent_id, 0)
            parent_info.append(f"{dish_name}({relevance_count}块)")

        logger.info(f"从 {len(child_chunks)} 个子块中找到 {len(parent_docs)} 个去重父文档: {', '.join(parent_info)}")
        return parent_docs
```
## 5.2 索引构建模块
- 将文本块转换为向量表示，并构建高效的检索索引。
- 向量数据库：FAISS
- embedding模型：BGE-small-zh-v1.5
- 混合索引：**向量检索**基于语义相似度，擅长理解查询意图；**BM25检索**基于关键词匹配，擅长精确匹配。为了综合两种检索方式的优势，我们使用RRF（Reciprocal Rank Fusion）算法来融合检索结果。
创建：`rag_modules/index_construction.py`
### 5.2.1类结构设计
- 需要一个类来构建索引
```python
class IndexConstructionModule:
    """索引构建模块 - 负责向量化和索引构建"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", index_save_path: str = "./vector_index"):
        """
        初始化索引构建模块

        Args:
            model_name: 嵌入模型名称
            index_save_path: 索引保存路径
        """
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.embeddings = None
        self.vectorstore = None
        self.setup_embeddings()
```
- `index_save_path`: 索引保存路径
- `embeddings`: HuggingFace嵌入模型实例
- `vectorstore`: FAISS向量存储实例
### 5.2.2 嵌入模型初始化
```python
import logging
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class IndexConstructionModule:
    def setup_embeddings(self):
        """初始化嵌入模型"""
        logger.info(f"正在初始化嵌入模型: {self.model_name}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        logger.info("嵌入模型初始化完成")
```
### 5.2.3 向量索引构建
```python
from langchain_community.vectorstores import FAISS
from typing import List
from langchain_core.documents import Document

class IndexConstructionModule:
    def build_vector_index(self, chunks: List[Document]) -> FAISS:
        """
        构建向量索引
        
        Args:
            chunks: 文档块列表
            
        Returns:
            FAISS向量存储对象
        """
        logger.info("正在构建FAISS向量索引...")
        
        if not chunks:
            raise ValueError("文档块列表不能为空")
        
        # 构建FAISS向量存储
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        logger.info(f"向量索引构建完成，包含 {len(chunks)} 个向量")
        return self.vectorstore
```
- 使用FAISS作为向量数据库，它的检索速度很快，同时保存了文本内容和元数据信息，支持大规模向量的高效检索。
### 5.2.4 索引缓存机制
```python
from pathlib import Path

class IndexConstructionModule:
    def save_index(self):
        """
        保存向量索引到配置的路径
        """
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")

        # 确保保存目录存在
        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)

        self.vectorstore.save_local(self.index_save_path)
        logger.info(f"向量索引已保存到: {self.index_save_path}")
    
    def load_index(self):
        """
        从配置的路径加载向量索引

        Returns:
            加载的向量存储对象，如果加载失败返回None
        """
        if not self.embeddings:
            self.setup_embeddings()

        if not Path(self.index_save_path).exists():
            logger.info(f"索引路径不存在: {self.index_save_path}，将构建新索引")
            return None

        try:
            self.vectorstore = FAISS.load_local(
                self.index_save_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"向量索引已从 {self.index_save_path} 加载")
            return self.vectorstore
        except Exception as e:
            logger.warning(f"加载向量索引失败: {e}，将构建新索引")
            return None
```
- 索引缓存的效果很明显：首次运行时构建索引需要几分钟，但后续运行时加载索引只需几秒钟。索引文件通常只有几十MB，存储效率很高。
## 5.3 检索优化模块
- 使用混合索引进行优化
创建`rag_modules/retrieval_optimization.py`
### 5.3.1 类结构设计
```python
class RetrievalOptimizationModule:
    """检索优化模块 - 负责混合检索和过滤"""
    
    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        """
        初始化检索优化模块
        
        Args:
            vectorstore: FAISS向量存储
            chunks: 文档块列表
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.setup_retrievers()
```
- `vectorstore`: FAISS向量存储实例
- `chunks`: 文档块列表，用于BM25检索
### 5.3.2 检索器设置
- 通常构建完向量数据库之后要把它实例化成一个retriever，设置好检索依据和k值等等
```python
    def setup_retrievers(self):
        """设置向量检索器和BM25检索器"""
        logger.info("正在设置检索器...")

        # 向量检索器
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # BM25检索器
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=5
        )

        logger.info("检索器设置完成")
```
### 5.3.3 RRF混合检索
- 同时使用向量数据库的向量检索和 chunk向量的元数据过滤检索
```python
    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        混合检索 - 结合向量检索和BM25检索，使用RRF重排

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            检索到的文档列表
        """
        # 分别获取向量检索和BM25检索结果
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        # 使用RRF重排
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)
        return reranked_docs[:top_k]
    

    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = 60) -> List[Document]:
        """
        使用RRF (Reciprocal Rank Fusion) 算法重排文档

        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25检索结果
            k: RRF参数，用于平滑排名

        Returns:
            重排后的文档列表
        """
        doc_scores = {}
        doc_objects = {}

        # 计算向量检索结果的RRF分数
        for rank, doc in enumerate(vector_docs):
            # 使用文档内容的确定性哈希作为唯一标识
            doc_id = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            doc_objects[doc_id] = doc

            # RRF公式: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"向量检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 计算BM25检索结果的RRF分数
        for rank, doc in enumerate(bm25_docs):
            doc_id = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            doc_objects[doc_id] = doc

            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"BM25检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 按最终RRF分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 构建最终结果
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                # 将RRF分数添加到文档元数据中
                doc.metadata['rrf_score'] = final_score
                reranked_docs.append(doc)
                logger.debug(f"最终排序 - 文档: {doc.page_content[:50]}... 最终RRF分数: {final_score:.4f}")

        logger.info(f"RRF重排完成: 向量检索{len(vector_docs)}个文档, BM25检索{len(bm25_docs)}个文档, 合并后{len(reranked_docs)}个文档")

        return reranked_docs
```
在当前系统中，两种检索方式各有优势：

**向量检索的优势**：

- 理解语义相似性，如"简单易做的菜"能匹配到标记为"简单"的菜谱
- 处理同义词和近义词，如"制作方法"和"做法"、"烹饪步骤"
- 理解用户意图，如"适合新手"能找到难度较低的菜谱

**BM25检索的优势**：

- 精确匹配菜名，如"宫保鸡丁"能准确找到对应菜谱
- 匹配具体食材，如"土豆丝"、"西红柿"等关键词
- 处理专业术语，如"爆炒"、"红烧"等烹饪手法

RRF算法能综合两种检索方式的排名信息，既保证了语义理解的准确性，又确保了关键词匹配的精确性。当然还可以用路由的方式，根据查询类型智能选择使用向量检索还是BM25检索。这种方法针对性强，能为不同类型的查询选择最优的检索方式；不足是路由规则的设计和维护比较复杂，边界情况难以处理，而且通常需要调用LLM来判断查询类型，会增加延迟和成本。
### 5.3.4 元数据过滤检索
```python
    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        """
        带元数据过滤的检索
        
        Args:
            query: 查询文本
            filters: 元数据过滤条件
            top_k: 返回结果数量
            
        Returns:
            过滤后的文档列表
        """
        # 先进行混合检索，获取更多候选
        docs = self.hybrid_search(query, top_k * 3)
        
        # 应用元数据过滤
        filtered_docs = []
        for doc in docs:
            match = True
            for key, value in filters.items():
                if key in doc.metadata:
                    if isinstance(value, list):
                        if doc.metadata[key] not in value:
                            match = False
                            break
                    else:
                        if doc.metadata[key] != value:
                            match = False
                            break
                else:
                    match = False
                    break
            
            if match:
                filtered_docs.append(doc)
                if len(filtered_docs) >= top_k:
                    break
        
        return filtered_docs
```
**过滤检索应用场景**：

- 用户询问"推荐几道素菜"时，可以按菜品分类过滤，只检索素菜相关的内容
- 新手用户问"有什么简单的菜谱"时，可以按难度等级过滤，只返回标记为"简单"的菜谱
- 想做汤品时询问"今天喝什么汤"，可以按分类过滤出所有汤品菜谱

## 5.4 生成集成
该部分需要完成：
- **智能查询路由**：根据用户查询自动判断是列表查询、详细查询还是一般查询，选择最适合的生成策略。

- **查询重写优化**：对模糊不清的查询进行智能重写，提升检索效果。比如将"做菜"重写为"简单易做的家常菜谱"。

- **多模式生成**：

	- **列表模式**：适用于推荐类查询，返回简洁的菜品列表
	- **详细模式**：适用于制作类查询，提供分步骤的详细指导
	- **基础模式**：适用于一般性问题，提供常规回答
创建`rag_modules/retrieval_optimization.py`
### 5.4.1 类结构设计
```python
class GenerationIntegrationModule:
    """生成集成模块 - 负责LLM集成和回答生成"""
    
    def __init__(self, model_name: str = "kimi-k2-0711-preview", temperature: float = 0.1, max_tokens: int = 2048):
        """
        初始化生成集成模块
        
        Args:
            model_name: 模型名称
            temperature: 生成温度
            max_tokens: 最大token数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.setup_llm()
```

### 5.4.2 初始化llm
```python
    def setup_llm(self):
        """初始化大语言模型"""
        logger.info(f"正在初始化LLM: {self.model_name}")

        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")

        self.llm = MoonshotChat(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            moonshot_api_key=api_key
        )
        
        logger.info("LLM初始化完成")
```
### 5.4.3 上下文构建
- 根据rag检索出来的文档构建上下文
```python
    def _build_context(self, docs: List[Document], max_length: int = 2000) -> str:
        """
        构建上下文字符串
        
        Args:
            docs: 文档列表
            max_length: 最大长度
            
        Returns:
            格式化的上下文字符串
        """
        if not docs:
            return "暂无相关食谱信息。"
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(docs, 1):
            # 添加元数据信息
            metadata_info = f"【食谱 {i}】"
            if 'dish_name' in doc.metadata:
                metadata_info += f" {doc.metadata['dish_name']}"
            if 'category' in doc.metadata:
                metadata_info += f" | 分类: {doc.metadata['category']}"
            if 'difficulty' in doc.metadata:
                metadata_info += f" | 难度: {doc.metadata['difficulty']}"
            
            # 构建文档文本
            doc_text = f"{metadata_info}\n{doc.page_content}\n"
            
            # 检查长度限制
            if current_length + len(doc_text) > max_length:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        divider = "\n" + "="*50 + "\n"
        return divider + divider.join(context_parts)

```
### 5.4.4 查询路由
```python

    def query_router(self, query: str) -> str:
        """
        查询路由 - 根据查询类型选择不同的处理方式

        Args:
            query: 用户查询

        Returns:
            路由类型 ('list', 'detail', 'general')
        """
        prompt = ChatPromptTemplate.from_template("""
根据用户的问题，将其分类为以下三种类型之一：

1. 'list' - 用户想要获取菜品列表或推荐，只需要菜名
   例如：推荐几个素菜、有什么川菜、给我3个简单的菜

2. 'detail' - 用户想要具体的制作方法或详细信息
   例如：宫保鸡丁怎么做、制作步骤、需要什么食材

3. 'general' - 其他一般性问题
   例如：什么是川菜、制作技巧、营养价值

请只返回分类结果：list、detail 或 general

用户问题: {query}

分类结果:""")

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        result = chain.invoke(query).strip().lower()

        # 确保返回有效的路由类型
        if result in ['list', 'detail', 'general']:
            return result
        else:
            return 'general'  # 默认类型
```
查询路由是整个系统的关键，决定了后续的处理流程。通过LLM自动判断查询意图，比简单的关键词匹配更准确，最终大模型输出关键性判断，从三种生成模式里面选择一个。
### 5.4.5 查询重写优化
```python
    def query_rewrite(self, query: str) -> str:
        """
        智能查询重写 - 让大模型判断是否需要重写查询

        Args:
            query: 原始查询

        Returns:
            重写后的查询或原查询
        """
        prompt = PromptTemplate(
            template="""
你是一个智能查询分析助手。请分析用户的查询，判断是否需要重写以提高食谱搜索效果。

原始查询: {query}

分析规则：
1. **具体明确的查询**（直接返回原查询）：
   - 包含具体菜品名称：如"宫保鸡丁怎么做"、"红烧肉的制作方法"
   - 明确的制作询问：如"蛋炒饭需要什么食材"、"糖醋排骨的步骤"
   - 具体的烹饪技巧：如"如何炒菜不粘锅"、"怎样调制糖醋汁"

2. **模糊不清的查询**（需要重写）：
   - 过于宽泛：如"做菜"、"有什么好吃的"、"推荐个菜"
   - 缺乏具体信息：如"川菜"、"素菜"、"简单的"
   - 口语化表达：如"想吃点什么"、"有饮品推荐吗"

重写原则：
- 保持原意不变
- 增加相关烹饪术语
- 优先推荐简单易做的
- 保持简洁性

示例：
- "做菜" → "简单易做的家常菜谱"
- "有饮品推荐吗" → "简单饮品制作方法"
- "推荐个菜" → "简单家常菜推荐"
- "川菜" → "经典川菜菜谱"
- "宫保鸡丁怎么做" → "宫保鸡丁怎么做"（保持原查询）
- "红烧肉需要什么食材" → "红烧肉需要什么食材"（保持原查询）

请输出最终查询（如果不需要重写就返回原查询）:""",
            input_variables=["query"]
        )

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query).strip()

        # 记录重写结果
        if response != query:
            logger.info(f"查询已重写: '{query}' → '{response}'")
        else:
            logger.info(f"查询无需重写: '{query}'")

        return response
```
将模糊的用户输入转换为更适合检索的查询，显著提升系统的实用性。重写规则包括：保持原意不变、增加相关烹饪术语、优先推荐简单易做的菜品。

### 5.4.6 多模式生成
- 提供三种生成模式
#### 1）列表模式生成
- 仅能用于菜品推荐，根据检索结果，直接输出菜品名称
```python
    def generate_list_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成列表式回答 - 适用于推荐类查询

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            列表式回答
        """
        if not context_docs:
            return "抱歉，没有找到相关的菜品信息。"

        # 提取菜品名称
        dish_names = []
        for doc in context_docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            if dish_name not in dish_names:
                dish_names.append(dish_name)

        # 构建简洁的列表回答
        if len(dish_names) == 1:
            return f"为您推荐：{dish_names[0]}"
        elif len(dish_names) <= 3:
            return f"为您推荐以下菜品：\n" + "\n".join([f"{i+1}. {name}" for i, name in enumerate(dish_names)])
        else:
            return f"为您推荐以下菜品：\n" + "\n".join([f"{i+1}. {name}" for i, name in enumerate(dish_names[:3])]) + f"\n\n还有其他 {len(dish_names)-3} 道菜品可供选择。"
```
#### 2）详细模式生成
- 仅用于菜品制作步骤
```python
    def generate_step_by_step_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成分步骤回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            分步骤的详细回答
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪导师。请根据食谱信息，为用户提供详细的分步骤指导。

用户问题: {question}

相关食谱信息:
{context}

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 🥘 菜品介绍
[简要介绍菜品特点和难度]

## 🛒 所需食材
[列出主要食材和用量]

## 👨‍🍳 制作步骤
[详细的分步骤说明，每步包含具体操作和大概所需时间]

## 💡 制作技巧
[仅在有实用技巧时包含。优先使用原文中的实用技巧，如果原文的"附加内容"与烹饪无关或为空，可以基于制作步骤总结关键要点，或者完全省略此部分]

注意：
- 根据实际内容灵活调整结构
- 不要强行填充无关内容或重复制作步骤中的信息
- 重点突出实用性和可操作性
- 如果没有额外的技巧要分享，可以省略制作技巧部分

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response
        

    def generate_step_by_step_answer_stream(self, query: str, context_docs: List[Document]):
        """
        生成详细步骤回答 - 流式输出

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            详细步骤回答片段
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪导师。请根据食谱信息，为用户提供详细的分步骤指导。

用户问题: {question}

相关食谱信息:
{context}

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 🥘 菜品介绍
[简要介绍菜品特点和难度]

## 🛒 所需食材
[列出主要食材和用量]

## 👨‍🍳 制作步骤
[详细的分步骤说明，每步包含具体操作和大概所需时间]

## 💡 制作技巧
[仅在有实用技巧时包含。优先使用原文中的实用技巧，如果原文的"附加内容"与烹饪无关或为空，可以基于制作步骤总结关键要点，或者完全省略此部分]

注意：
- 根据实际内容灵活调整结构
- 不要强行填充无关内容或重复制作步骤中的信息
- 重点突出实用性和可操作性
- 如果没有额外的技巧要分享，可以省略制作技巧部分

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk
```
#### 3）基础生成模式
- 回答用户其他问题
```python
    def generate_basic_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成基础回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            生成的回答
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

用户问题: {question}

相关食谱信息:
{context}

请提供详细、实用的回答。如果信息不足，请诚实说明。

回答:""")

        # 使用LCEL构建链
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response
        
    def generate_basic_answer_stream(self, query: str, context_docs: List[Document]):
        """
        生成基础回答 - 流式输出

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            生成的回答片段
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

用户问题: {question}

相关食谱信息:
{context}

请提供详细、实用的回答。如果信息不足，请诚实说明。

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk
```
## 5.5 系统整合
- 主程序负责协调各个模块，实现完整的RAG流程：数据准备 → 索引构建 → 检索优化 → 生成集成。同时提供了索引缓存、交互式问答等实用功能。
创建`main.py`
### 5.5.1 主系统类设计
- 协调所有模块
```python
class RecipeRAGSystem:
    """食谱RAG系统主类"""

    def __init__(self, config: RAGConfig = None):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        # 检查数据路径
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        # 检查API密钥
        if not os.getenv("MOONSHOT_API_KEY"):
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")
    
```
### 5.5.2 系统初始化
需要初始化：
- 数据准备模块（类）
- 索引构建模块
- 生成集成模块
```python

    def initialize_system(self):
        """初始化所有模块"""
        print("🚀 正在初始化RAG系统...")

        # 1. 初始化数据准备模块
        print("初始化数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_path)

        # 2. 初始化索引构建模块
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )

        # 3. 初始化生成集成模块
        print("🤖 初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        print("✅ 系统初始化完成！")
```
### 5.5.3 知识库构建
- 先用索引模块尝试加载本地缓存的索引，不存在就要使用数据模块加载文档构建向量数据库。得到向量索引之后还要把源文档切分（这里设计不合理，每次启动都要做切分，切分的chunkid还可能跟之前不一样）。
- 初始化检索优化模块
```python
    def build_knowledge_base(self):
        """构建知识库"""
        print("\n正在构建知识库...")

        # 1. 尝试加载已保存的索引
        vectorstore = self.index_module.load_index()

        if vectorstore is not None:
            print("✅ 成功加载已保存的向量索引！")
            # 仍需要加载文档和分块用于检索模块
            print("加载食谱文档...")
            self.data_module.load_documents()
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()
        else:
            print("未找到已保存的索引，开始构建新索引...")

            # 2. 加载文档
            print("加载食谱文档...")
            self.data_module.load_documents()

            # 3. 文本分块
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()

            # 4. 构建向量索引
            print("构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)

            # 5. 保存索引
            print("保存向量索引...")
            self.index_module.save_index()

        # 6. 初始化检索优化模块
        print("初始化检索优化...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        # 7. 显示统计信息
        stats = self.data_module.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   菜品分类: {list(stats['categories'].keys())}")
        print(f"   难度分布: {stats['difficulties']}")

        print("✅ 知识库构建完成！")
    
```
### 5.5.4 智能问答流程
- 程序执行流程：智能路由 → 查询优化 → 混合检索 → 获取父文档 → 多模式生成。
```python
    def ask_question(self, question: str, stream: bool = False):
        """
        回答用户问题

        Args:
            question: 用户问题
            stream: 是否使用流式输出

        Returns:
            生成的回答或生成器
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")
        
        print(f"\n❓ 用户问题: {question}")

        # 1. 查询路由
        route_type = self.generation_module.query_router(question)
        print(f"🎯 查询类型: {route_type}")

        # 2. 智能查询重写（根据路由类型）
        if route_type == 'list':
            # 列表查询保持原查询
            rewritten_query = question
            print(f"📝 列表查询保持原样: {question}")
        else:
            # 详细查询和一般查询使用智能重写
            print("🤖 智能分析查询...")
            rewritten_query = self.generation_module.query_rewrite(question)
        
        # 3. 检索相关子块（自动应用元数据过滤）
        print("🔍 检索相关文档...")
        filters = self._extract_filters_from_query(question)
        if filters:
            print(f"应用过滤条件: {filters}")
            relevant_chunks = self.retrieval_module.metadata_filtered_search(rewritten_query, filters, top_k=self.config.top_k)
        else:
            relevant_chunks = self.retrieval_module.hybrid_search(rewritten_query, top_k=self.config.top_k)

        # 显示检索到的子块信息
        if relevant_chunks:
            chunk_info = []
            for chunk in relevant_chunks:
                dish_name = chunk.metadata.get('dish_name', '未知菜品')
                # 尝试从内容中提取章节标题
                content_preview = chunk.page_content[:100].strip()
                if content_preview.startswith('#'):
                    # 如果是标题开头，提取标题（仅取第一行）
                    title_end = content_preview.find('\n') if '\n' in content_preview else len(content_preview)
                    section_title = content_preview[:title_end].replace('#', '').strip()
                    chunk_info.append(f"{dish_name}({section_title})")
                else:
                    chunk_info.append(f"{dish_name}(内容片段)")

            print(f"找到 {len(relevant_chunks)} 个相关文档块: {', '.join(chunk_info)}")
        else:
            print(f"找到 {len(relevant_chunks)} 个相关文档块")

        # 4. 检查是否找到相关内容
        if not relevant_chunks:
            return "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"

        # 5. 根据路由类型选择回答方式
        if route_type == 'list':
            # 列表查询：直接返回菜品名称列表
            print("📋 生成菜品列表...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

            # 显示找到的文档名称
            doc_names = []
            for doc in relevant_docs:
                dish_name = doc.metadata.get('dish_name', '未知菜品')
                doc_names.append(dish_name)

            if doc_names:
                print(f"找到文档: {', '.join(doc_names)}")

            return self.generation_module.generate_list_answer(question, relevant_docs)
        else:
            # 详细查询：获取完整文档并生成详细回答
            print("获取完整文档...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

            # 显示找到的文档名称
            doc_names = []
            for doc in relevant_docs:
                dish_name = doc.metadata.get('dish_name', '未知菜品')
                doc_names.append(dish_name)

            if doc_names:
                print(f"找到文档: {', '.join(doc_names)}")
            else:
                print(f"对应 {len(relevant_docs)} 个完整文档")

            print("✍️ 生成详细回答...")

            # 根据路由类型自动选择回答模式
            if route_type == "detail":
                # 详细查询使用分步指导模式
                if stream:
                    return self.generation_module.generate_step_by_step_answer_stream(question, relevant_docs)
                else:
                    return self.generation_module.generate_step_by_step_answer(question, relevant_docs)
            else:
                # 一般查询使用基础回答模式
                if stream:
                    return self.generation_module.generate_basic_answer_stream(question, relevant_docs)
                else:
                    return self.generation_module.generate_basic_answer(question, relevant_docs)
```
### 5.5.5 元数据过滤
- 仅支持对制作难度、菜品类型的过滤：
```python
    def _extract_filters_from_query(self, query: str) -> dict:
        """
        从用户问题中提取元数据过滤条件
        """
        filters = {}
        # 分类关键词
        category_keywords = DataPreparationModule.get_supported_categories()
        for cat in category_keywords:
            if cat in query:
                filters['category'] = cat
                break

        # 难度关键词
        difficulty_keywords = DataPreparationModule.get_supported_difficulties()
        for diff in sorted(difficulty_keywords, key=len, reverse=True):
            if diff in query:
                filters['difficulty'] = diff
                break

        return filters
```
### 5.5.6 交互式问答
```python
    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("🍽️  尝尝咸淡RAG系统 - 交互式问答  🍽️")
        print("=" * 60)
        print("💡 解决您的选择困难症，告别'今天吃什么'的世纪难题！")
        
        # 初始化系统
        self.initialize_system()
        
        # 构建知识库
        self.build_knowledge_base()
        
        print("\n交互式问答 (输入'退出'结束):")
        
        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break
                
                # 询问是否使用流式输出
                stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                use_stream = stream_choice != 'n'

                print("\n回答:")
                if use_stream:
                    # 流式输出
                    for chunk in self.ask_question(user_input, stream=True):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    # 普通输出
                    answer = self.ask_question(user_input, stream=False)
                    print(f"{answer}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")
        
        print("\n感谢使用尝尝咸淡RAG系统！")


def main():
    """主函数"""
    try:
        # 创建RAG系统
        rag_system = RecipeRAGSystem()
        
        # 运行交互式问答
        rag_system.run_interactive()
        
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")

if __name__ == "__main__":
    main()
```

## 5.6 参数配置
创建`config.py`
```python
"""
RAG系统配置文件
"""

from dataclasses import asdict, dataclass
from typing import Dict, Any

@dataclass
class RAGConfig:
    """RAG系统配置类"""

    # 路径配置
    data_path: str = "../../data/C8/cook"
    index_save_path: str = "./vector_index"

    # 模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = "kimi-k2-0711-preview"

    # 检索配置
    top_k: int = 3

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

# 默认配置实例
DEFAULT_CONFIG = RAGConfig()
```
 
 # 6. 优化方向
 - **集成图数据库** 将食谱数据构建为知识图谱，来揭示食材、菜品与烹饪方法间的复杂关联，进而支持复杂关系查询（如“和鸡肉搭配的食材有哪些”）、发掘潜在的食材组合并实现基于图的智能推荐。
 - **融合多模态数据**，结合菜品图片等视觉信息，利用多模态模型进行图文联合检索，不仅能支持“这是什么菜”的视觉搜索，还可以通过图像识别食材来推荐相关菜谱。
 - **增强专业知识**，集成营养成分数据库、烹饪技巧知识图谱以及食材替换规则库等外部知识源，系统将能提供精准的营养分析、专业的烹饪指导，并灵活适应用户的饮食过敏或个人偏好。
# 7. 代码优化
## 7.1 引入图数据库
- 目前缺陷：
	- **关系理解缺失**：无法显式建模食材、菜谱、烹饪方法之间的语义关系
	- **跨文档关联困难**：难以发现不同菜谱之间的相似性、替代关系等隐含联系
	- **推理能力有限**：缺乏基于知识图谱的多跳推理能力，难以回答需要复杂逻辑推理的问题
### 7.1.1 图数据模型设计
- 我们要将markdown的菜谱信息提取成知识图谱数据，需要一个合理的提取方式，每个文档我们输出三类信息：
	- 菜谱信息
	- 食材信息
	- 烹饪步骤
```python
@dataclass
class IngredientInfo:
    """食材信息"""
    name: str
    amount: str = ""
    unit: str = ""
    category: str = ""
    is_main: bool = True  # 是否主要食材
    
@dataclass
class CookingStep:
    """烹饪步骤"""
    step_number: int
    description: str
    methods: List[str]  # 使用的烹饪方法
    tools: List[str]    # 需要的工具
    time_estimate: str = ""  # 时间估计
    
@dataclass
class RecipeInfo:
    """菜谱信息"""
    name: str
    difficulty: int  # 1-5星
    category: str
    cuisine_type: str = ""  # 菜系
    prep_time: str = ""
    cook_time: str = ""
    servings: str = ""
    ingredients: List[IngredientInfo] = None
    steps: List[CookingStep] = None
    tags: List[str] = None
    nutrition_info: Dict = None
    
    def __post_init__(self):
        if self.ingredients is None:
            self.ingredients = []
        if self.steps is None:
            self.steps = []
        if self.tags is None:
            self.tags = []
        if self.nutrition_info is None:
            self.nutrition_info = {}
```
- 最终提一篇文档会被转换成一个菜谱类+多个食材类+多个步骤类。然后输出成一个json。最终我们要构建的知识图谱里面并不仅仅只有这三类节点，会有多种类型的节点：**核心实体类型**：

- **Recipe (菜谱)**：具体的菜品，包含难度、菜系、时间等属性
- **Ingredient (食材)**：制作菜品所需的原料，包含分类、用量、单位等
- **CookingStep (烹饪步骤)**：详细的制作步骤，包含方法、工具、时间估计
- **CookingMethod (烹饪方法)**：如炒、煮、蒸、炸等烹饪技法
- **CookingTool (烹饪工具)**：如炒锅、蒸锅、刀具等
- **DifficultyLevel (难度等级)**：一星到五星的难度分级
- **RecipeCategory (菜谱分类)**：素菜、荤菜、水产、早餐等分类
> 菜谱节点里面已经有了难度属性，为什么还需要单独的难度等级节点？把难度建成独立节点后，就能从“**三星**”这个节点直接找到所有三星菜谱。

最终将数据构建为如下知识图谱，其中一共包含七类节点，烹饪方法和烹饪工具节点目前暂时无用，主要是菜谱节点，他跟四种节点存在关系，每个节点/关系都有自己的id和属性。
![](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/llm/cookrag_graph.png)
**关系类型说明**：

|关系编码|关系类型|说明|属性|
|---|---|---|---|
|**801000001**|REQUIRES|菜谱-食材关系|relationshipId, amount, unit|
|**801000003**|CONTAINS_STEP|菜谱-步骤关系|relationshipId, step_order|
|**801000004**|HAS_DIFFICULTY_LEVEL|菜谱-难度关系|relationshipId|
|**801000005**|BELONGS_TO_CATEGORY|菜谱-分类关系|relationshipId|

**关系特点**：

- **虚线箭头**：表示带有丰富属性的关系（如REQUIRES、CONTAINS_STEP）
- **实线箭头**：表示简单的分类关系
- **层次化结构**：Root节点作为概念层次的顶层节点

### 7.1.2 Neo4j数据导入
创建文件`rag_modules/graph_data_preparation.py`
#### 1）定义节点与关系
```python
@dataclass
class GraphNode:
    """图节点数据结构"""
    node_id: str
    labels: List[str]
    name: str
    properties: Dict[str, Any]

@dataclass
class GraphRelation:
    """图关系数据结构"""
    start_node_id: str
    end_node_id: str
    relation_type: str
    properties: Dict[str, Any]
```
#### 2）数据加载
1. 提供一个类来处理图数据的加载和管理：
```python
class GraphDataPreparationModule:
    """图数据库数据准备模块 - 从Neo4j读取数据并转换为文档"""
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """
        初始化图数据库连接
        
        Args:
            uri: Neo4j连接URI
            user: 用户名
            password: 密码
            database: 数据库名称
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.recipes: List[GraphNode] = []
        self.ingredients: List[GraphNode] = []
        self.cooking_steps: List[GraphNode] = []
        
        self._connect()

    def _connect(self):
        """建立Neo4j连接"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password),
                database=self.database
            )
            logger.info(f"已连接到Neo4j数据库: {self.uri}")
            
            # 测试连接
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_result = result.single()
                if test_result:
                    logger.info("Neo4j连接测试成功")
                    
        except Exception as e:
            logger.error(f"连接Neo4j失败: {e}")
            raise
    
    def close(self):
        """关闭数据库连接"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")
```
2. 加载图数据：加载三类核心节点：菜谱节点、食材节点、烹饪步骤节点。这三类节点会用于后面构建rag。
```python
    def load_graph_data(self) -> Dict[str, Any]:
        """
        从Neo4j加载图数据
        
        Returns:
            包含节点和关系的数据字典
        """
        logger.info("正在从Neo4j加载图数据...")
        
        with self.driver.session() as session:
            # 加载所有菜谱节点，从Category关系中读取分类信息
            recipes_query = """
            MATCH (r:Recipe)
            WHERE r.nodeId >= '200000000'
            OPTIONAL MATCH (r)-[:BELONGS_TO_CATEGORY]->(c:Category)
            WITH r, collect(c.name) as categories
            RETURN r.nodeId as nodeId, labels(r) as labels, r.name as name, 
                   properties(r) as originalProperties,
                   CASE WHEN size(categories) > 0 
                        THEN categories[0] 
                        ELSE COALESCE(r.category, '未知') END as mainCategory,
                   CASE WHEN size(categories) > 0 
                        THEN categories 
                        ELSE [COALESCE(r.category, '未知')] END as allCategories
            ORDER BY r.nodeId
            """
            
            result = session.run(recipes_query)
            self.recipes = []
            for record in result:
                # 合并原始属性和新的分类信息
                properties = dict(record["originalProperties"])
                properties["category"] = record["mainCategory"]
                properties["all_categories"] = record["allCategories"]
                
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=properties
                )
                self.recipes.append(node)
            
            logger.info(f"加载了 {len(self.recipes)} 个菜谱节点")
            
            # 加载所有食材节点
            ingredients_query = """
            MATCH (i:Ingredient)
            WHERE i.nodeId >= '200000000'
            RETURN i.nodeId as nodeId, labels(i) as labels, i.name as name,
                   properties(i) as properties
            ORDER BY i.nodeId
            """
            
            result = session.run(ingredients_query)
            self.ingredients = []
            for record in result:
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=record["properties"]
                )
                self.ingredients.append(node)
            
            logger.info(f"加载了 {len(self.ingredients)} 个食材节点")
            
            # 加载所有烹饪步骤节点
            steps_query = """
            MATCH (s:CookingStep)
            WHERE s.nodeId >= '200000000'
            RETURN s.nodeId as nodeId, labels(s) as labels, s.name as name,
                   properties(s) as properties
            ORDER BY s.nodeId
            """
            
            result = session.run(steps_query)
            self.cooking_steps = []
            for record in result:
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=record["properties"]
                )
                self.cooking_steps.append(node)
            
            logger.info(f"加载了 {len(self.cooking_steps)} 个烹饪步骤节点")
        
        return {
            'recipes': len(self.recipes),
            'ingredients': len(self.ingredients),
            'cooking_steps': len(self.cooking_steps)
        }
```
#### 3）数据格式分析
我们通过将markdown文档构建为图数据，保存为两个csv文件，其中：
1. nodes.csv:
```csv
nodeId,labels,name,preferredTerm,fsn,conceptType,synonyms,category,difficulty,cuisineType,prepTime,cookTime,servings,tags,filePath,amount,unit,isMain,description,stepNumber,methods,tools,timeEstimate
201000184,Recipe,干煎阿根廷红虾,干煎阿根廷红虾,,Recipe,"[{'term': '干pan-fried阿根廷红虾', 'language': 'zh'}]",水产,3.0,,提前1天冷藏解冻+10分钟,约5分钟,1人,"趁热吃,柠檬可增酸提味",dishes\aquatic\干煎阿根廷红虾\干煎阿根廷红虾.md,,,,,,,,
201000185,Ingredient,阿根廷红虾,阿根廷红虾,,Ingredient,,蛋白质,,,,,,,,2-3,只,True,,,,,
201000196,CookingStep,步骤1,步骤1,,CookingStep,,,,,,,,,,,,,阿根廷红虾提前1天从速冻取出放到冷藏里自然解冻,1.0,解冻,冰箱,24小时
```
2. relationships.csv:
```csv
startNodeId,endNodeId,relationshipType,relationshipId,amount,unit,step_order
201000184,201000185,801000001,R_000001,2-3,只,
201000184,201000196,801000003,R_000010,,,1.0
201000184,720000000,801000002,R_000020,,,
```
#### 4）图数据查询
加载数据时用到了一些类似sql的查询语句，这里简单做了解：
##### 基础查询模式
1. 简单说实体查询
```
// 查找所有水产类菜谱
MATCH (r:Recipe)
WHERE r.category = "水产"
RETURN r.name, r.difficulty, r.prepTime, r.cookTime

// 查找包含特定食材的菜谱
MATCH (r:Recipe)-[:REQUIRES]->(i:Ingredient)
WHERE i.name CONTAINS "虾"
RETURN r.name, r.difficulty, i.name, i.amount, i.unit

// 使用全文搜索查找菜谱
CALL db.index.fulltext.queryNodes("recipe_fulltext_index", "川菜 OR 辣椒")
YIELD node, score
RETURN node.name, node.category, score
ORDER BY score DESC
```
2. 多跳关系查询
```
// 查找某个难度等级的所有菜谱（基于属性查询）
MATCH (r:Recipe)
WHERE r.difficulty = 3.0
RETURN r.name, r.category, r.prepTime, r.cookTime, r.difficulty

// 查找菜谱的完整制作流程
MATCH (r:Recipe {name: "干煎阿根廷红虾"})-[:CONTAINS_STEP]->(s:CookingStep)
RETURN r.name, s.stepNumber, s.description, s.methods, s.tools
ORDER BY s.stepNumber
```
##### 复杂推理查询
1. 基于约束的菜谱推荐
```
// 查找适合新手的简单菜谱（低难度、步骤少）
MATCH (r:Recipe)
WHERE r.difficulty <= 2.0
  AND r.stepCount <= 5
RETURN r.name, r.difficulty, r.stepCount, r.category
ORDER BY r.difficulty, r.stepCount

// 查找制作时间短的菜谱
MATCH (r:Recipe)
WHERE r.prepTime IS NOT NULL AND r.cookTime IS NOT NULL
  AND r.prepTime CONTAINS "分钟" AND r.cookTime CONTAINS "分钟"
RETURN r.name, r.prepTime, r.cookTime, r.category
ORDER BY r.name
```
2. 菜谱组合推荐
```
// 查找同一分类下的不同菜谱
MATCH (r1:Recipe), (r2:Recipe)
WHERE r1.category = r2.category
  AND r1.category = "水产"
  AND r1.nodeId <> r2.nodeId
RETURN r1.name, r2.name, r1.category
LIMIT 5

// 查找包含相同食材的不同菜谱
MATCH (r1:Recipe)-[:REQUIRES]->(i:Ingredient)<-[:REQUIRES]-(r2:Recipe)
WHERE r1.nodeId <> r2.nodeId
  AND i.name = "阿根廷红虾"
RETURN r1.name, r2.name, i.name
```
#### 5）图数据转换为结构化文档
从图数据中恢复出一个文档，这个文档类似原始文档，描述一个菜谱。
```python
    def build_recipe_documents(self) -> List[Document]:
        """
        构建菜谱文档，集成相关的食材和步骤信息
        
        Returns:
            结构化的菜谱文档列表
        """
        logger.info("正在构建菜谱文档...")
        
        documents = []
        
        with self.driver.session() as session:
            for recipe in self.recipes:
                try:
                    recipe_id = recipe.node_id
                    recipe_name = recipe.name
                    
                    # 获取菜谱的相关食材
                    ingredients_query = """
                    MATCH (r:Recipe {nodeId: $recipe_id})-[req:REQUIRES]->(i:Ingredient)
                    RETURN i.name as name, i.category as category, 
                           req.amount as amount, req.unit as unit,
                           i.description as description
                    ORDER BY i.name
                    """
                    
                    ingredients_result = session.run(ingredients_query, {"recipe_id": recipe_id})
                    ingredients_info = []
                    for ing_record in ingredients_result:
                        amount = ing_record.get("amount", "")
                        unit = ing_record.get("unit", "")
                        ingredient_text = f"{ing_record['name']}"
                        if amount and unit:
                            ingredient_text += f"({amount}{unit})"
                        if ing_record.get("description"):
                            ingredient_text += f" - {ing_record['description']}"
                        ingredients_info.append(ingredient_text)
                    
                    # 获取菜谱的烹饪步骤
                    steps_query = """
                    MATCH (r:Recipe {nodeId: $recipe_id})-[c:CONTAINS_STEP]->(s:CookingStep)
                    RETURN s.name as name, s.description as description,
                           s.stepNumber as stepNumber, s.methods as methods,
                           s.tools as tools, s.timeEstimate as timeEstimate,
                           c.stepOrder as stepOrder
                    ORDER BY COALESCE(c.stepOrder, s.stepNumber, 999)
                    """
                    
                    steps_result = session.run(steps_query, {"recipe_id": recipe_id})
                    steps_info = []
                    for step_record in steps_result:
                        step_text = f"步骤: {step_record['name']}"
                        if step_record.get("description"):
                            step_text += f"\n描述: {step_record['description']}"
                        if step_record.get("methods"):
                            step_text += f"\n方法: {step_record['methods']}"
                        if step_record.get("tools"):
                            step_text += f"\n工具: {step_record['tools']}"
                        if step_record.get("timeEstimate"):
                            step_text += f"\n时间: {step_record['timeEstimate']}"
                        steps_info.append(step_text)
                    
                    # 构建完整的菜谱文档内容
                    content_parts = [f"# {recipe_name}"]
                    
                    # 添加菜谱基本信息
                    if recipe.properties.get("description"):
                        content_parts.append(f"\n## 菜品描述\n{recipe.properties['description']}")
                    
                    if recipe.properties.get("cuisineType"):
                        content_parts.append(f"\n菜系: {recipe.properties['cuisineType']}")
                    
                    if recipe.properties.get("difficulty"):
                        content_parts.append(f"难度: {recipe.properties['difficulty']}星")
                    
                    if recipe.properties.get("prepTime") or recipe.properties.get("cookTime"):
                        time_info = []
                        if recipe.properties.get("prepTime"):
                            time_info.append(f"准备时间: {recipe.properties['prepTime']}")
                        if recipe.properties.get("cookTime"):
                            time_info.append(f"烹饪时间: {recipe.properties['cookTime']}")
                        content_parts.append(f"\n时间信息: {', '.join(time_info)}")
                    
                    if recipe.properties.get("servings"):
                        content_parts.append(f"份量: {recipe.properties['servings']}")
                    
                    # 添加食材信息
                    if ingredients_info:
                        content_parts.append("\n## 所需食材")
                        for i, ingredient in enumerate(ingredients_info, 1):
                            content_parts.append(f"{i}. {ingredient}")
                    
                    # 添加步骤信息
                    if steps_info:
                        content_parts.append("\n## 制作步骤")
                        for i, step in enumerate(steps_info, 1):
                            content_parts.append(f"\n### 第{i}步\n{step}")
                    
                    # 添加标签信息
                    if recipe.properties.get("tags"):
                        content_parts.append(f"\n## 标签\n{recipe.properties['tags']}")
                    
                    # 组合成最终内容
                    full_content = "\n".join(content_parts)
                    
                    # 创建文档对象
                    doc = Document(
                        page_content=full_content,
                        metadata={
                            "node_id": recipe_id,
                            "recipe_name": recipe_name,
                            "node_type": "Recipe",
                            "category": recipe.properties.get("category", "未知"),
                            "cuisine_type": recipe.properties.get("cuisineType", "未知"),
                            "difficulty": recipe.properties.get("difficulty", 0),
                            "prep_time": recipe.properties.get("prepTime", ""),
                            "cook_time": recipe.properties.get("cookTime", ""),
                            "servings": recipe.properties.get("servings", ""),
                            "ingredients_count": len(ingredients_info),
                            "steps_count": len(steps_info),
                            "doc_type": "recipe",
                            "content_length": len(full_content)
                        }
                    )
                    
                    documents.append(doc)
                    
                except Exception as e:
                    logger.warning(f"构建菜谱文档失败 {recipe_name} (ID: {recipe_id}): {e}")
                    continue
        
        self.documents = documents
        logger.info(f"成功构建 {len(documents)} 个菜谱文档")
        return documents
```
> 实际上这里完全可以直接用原始markdown文档，本项目其实就是在向量数据库的基础上增加了一个图数据库，然后两个数据库通过菜谱id联系起来。这里的方式直接从图数据库获取菜谱id然后作为元数据放到向量数据库，实现的时候其实也完全可以直接用原始markdown就行，只不过要保证两个数据库的菜谱id对应关系。另外，如果图数据库中拼接出来的文档包含更多多跳推理信息，比如某个食材与某个食材相近等额外信息，这里的处理方式才有用。
#### 6）图RAG分块

- 本项目所谓的图RAG其实就是增加了一个图数据库而已。分块方式也做了一点点改进。

| 特性        | 传统RAG          | 图RAG       |
| --------- | -------------- | ---------- |
| **数据来源**  | 直接读取Markdown文件 | 从图数据库构建文档  |
| **上下文获取** | 父子文档映射         | 图关系遍历      |
| **关系信息**  | 有限（仅父子关系）      | 丰富（多种图关系）  |
| **分块策略**  | 按Markdown标题分块  | 按语义+长度智能分块 |
| **元数据来源** | 文件路径+内容推断      | 图节点结构化数据   |

**图RAG分块的特点**：

1. **保持图关联**：每个chunk通过`parent_id`与图节点关联
2. **语义优先分块**：优先按章节分块，保持语义完整性
3. **丰富的元数据**：直接从图节点获取结构化信息
4. **双重上下文**：既有文本块关系，又有图关系信息

```python
    def chunk_documents(self, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
        """
        对文档进行分块处理
        
        Args:
            chunk_size: 分块大小
            chunk_overlap: 重叠大小
            
        Returns:
            分块后的文档列表
        """
        logger.info(f"正在进行文档分块，块大小: {chunk_size}, 重叠: {chunk_overlap}")
        
        if not self.documents:
            raise ValueError("请先构建文档")
        
        chunks = []
        chunk_id = 0
        
        for doc in self.documents:
            content = doc.page_content
            
            # 简单的按长度分块
            if len(content) <= chunk_size:
                # 内容较短，不需要分块
                chunk = Document(
                    page_content=content,
                    metadata={
                        **doc.metadata,
                        "chunk_id": f"{doc.metadata['node_id']}_chunk_{chunk_id}",
                        "parent_id": doc.metadata["node_id"],
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "chunk_size": len(content),
                        "doc_type": "chunk"
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
            else:
                # 按章节分块（基于标题）
                sections = content.split('\n## ')
                if len(sections) <= 1:
                    # 没有二级标题，按长度强制分块
                    total_chunks = (len(content) - 1) // (chunk_size - chunk_overlap) + 1
                    
                    for i in range(total_chunks):
                        start = i * (chunk_size - chunk_overlap)
                        end = min(start + chunk_size, len(content))
                        
                        chunk_content = content[start:end]
                        
                        chunk = Document(
                            page_content=chunk_content,
                            metadata={
                                **doc.metadata,
                                "chunk_id": f"{doc.metadata['node_id']}_chunk_{chunk_id}",
                                "parent_id": doc.metadata["node_id"],
                                "chunk_index": i,
                                "total_chunks": total_chunks,
                                "chunk_size": len(chunk_content),
                                "doc_type": "chunk"
                            }
                        )
                        chunks.append(chunk)
                        chunk_id += 1
                else:
                    # 按章节分块
                    total_chunks = len(sections)
                    for i, section in enumerate(sections):
                        if i == 0:
                            # 第一个部分包含标题
                            chunk_content = section
                        else:
                            # 其他部分添加章节标题
                            chunk_content = f"## {section}"
                        
                        chunk = Document(
                            page_content=chunk_content,
                            metadata={
                                **doc.metadata,
                                "chunk_id": f"{doc.metadata['node_id']}_chunk_{chunk_id}",
                                "parent_id": doc.metadata["node_id"],
                                "chunk_index": i,
                                "total_chunks": total_chunks,
                                "chunk_size": len(chunk_content),
                                "doc_type": "chunk",
                                "section_title": section.split('\n')[0] if i > 0 else "主标题"
                            }
                        )
                        chunks.append(chunk)
                        chunk_id += 1
        
        self.chunks = chunks
        logger.info(f"文档分块完成，共生成 {len(chunks)} 个块")
        return chunks
```

从图数据库构建标准化文档，每个chunk通过`parent_id`与原始Recipe节点保持关联，**既继承了传统的父子文档映射关系，又能通过图关系遍历获取更丰富的上下文信息**（其实就是因为你多存了一个图数据库）。在具体实现上，采用智能分块策略：短文档保持完整避免破坏语义，长文档优先按`##`标题进行章节分块，必要时才进行长度分块，同时为每个chunk提供丰富的元数据（如chunk_id、chunk_index、total_chunks等），确保后续处理的灵活性和可追溯性。

## 7.2 向量数据库升级
- 这一节其实就是将向量数据库换成Milvus。
**FAISS的局限性**：

- **纯库模式**：FAISS是一个向量搜索库，缺乏数据库的完整功能
- **无持久化**：需要手动管理数据持久化和备份
- **单机限制**：难以实现分布式部署和水平扩展
- **元数据支持有限**：无法高效存储和查询复杂的结构化元数据
- **并发性能**：在高并发场景下性能受限

**Milvus的优势**：

- **完整数据库功能**：提供CRUD操作、事务支持、数据一致性保证
- **云原生架构**：支持分布式部署、自动扩缩容、高可用性
- **丰富的元数据支持**：支持复杂Schema设计，适合图RAG的多维度数据
- **生产级特性**：监控、日志、备份恢复等企业级功能
创建`rag_modules/milvus_index_construction.py`
### 7.2.1 定义索引构建器
- 初始化 Milvus 配置、embedding 配置和状态变量
```python
class MilvusIndexConstructionModule:
    """Milvus索引构建模块 - 负责向量化和Milvus索引构建"""

    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 19530,
                 collection_name: str = "cooking_knowledge",
                 dimension: int = 512,
                 model_name: str = "BAAI/bge-small-zh-v1.5"):
        """
        初始化Milvus索引构建模块

        Args:
            host: Milvus服务器地址
            port: Milvus服务器端口
            collection_name: 集合名称
            dimension: 向量维度
            model_name: 嵌入模型名称
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.model_name = model_name
        
        self.client = None
        self.embeddings = None
        self.collection_created = False
        
        self._setup_client()
        self._setup_embeddings()
        
    def _setup_client(self):
        """初始化Milvus客户端"""
        try:
            self.client = MilvusClient(
                uri=f"http://{self.host}:{self.port}"
            )
            logger.info(f"已连接到Milvus服务器: {self.host}:{self.port}")
            
            # 测试连接
            collections = self.client.list_collections()
            logger.info(f"连接成功，当前集合: {collections}")
            
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise
    
    def _setup_embeddings(self):
        """初始化嵌入模型"""
        logger.info(f"正在初始化嵌入模型: {self.model_name}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cuda:0'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        logger.info("嵌入模型初始化完成")
```
### 7.2.2 建立向量数据库
#### 1）定义建立流程
- 总体建立流程为：创建 Collection→ 文本向量化→ 数据写入 Milvus→ 建立 HNSW 索引→ 加载 Collection
```python
    def build_vector_index(self, chunks: List[Document]) -> bool:
        """
        构建向量索引
        
        Args:
            chunks: 文档块列表
            
        Returns:
            是否构建成功
        """
        logger.info(f"正在构建Milvus向量索引，文档数量: {len(chunks)}...")
        
        if not chunks:
            raise ValueError("文档块列表不能为空")
        
        try:
            # 1. 创建集合（如果schema不兼容则强制重新创建）
            if not self.create_collection(force_recreate=True):
                return False
            
            # 2. 准备数据
            logger.info("正在生成向量embeddings...")
            texts = [chunk.page_content for chunk in chunks]
            vectors = self.embeddings.embed_documents(texts)
            
            # 3. 准备插入数据
            entities = []
            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                entity = {
                    "id": self._safe_truncate(chunk.metadata.get("chunk_id", f"chunk_{i}"), 150),
                    "vector": vector,
                    "text": self._safe_truncate(chunk.page_content, 15000),
                    "node_id": self._safe_truncate(chunk.metadata.get("node_id", ""), 100),
                    "recipe_name": self._safe_truncate(chunk.metadata.get("recipe_name", ""), 300),
                    "node_type": self._safe_truncate(chunk.metadata.get("node_type", ""), 100),
                    "category": self._safe_truncate(chunk.metadata.get("category", ""), 100),
                    "cuisine_type": self._safe_truncate(chunk.metadata.get("cuisine_type", ""), 200),
                    "difficulty": int(chunk.metadata.get("difficulty", 0)),
                    "doc_type": self._safe_truncate(chunk.metadata.get("doc_type", ""), 50),
                    "chunk_id": self._safe_truncate(chunk.metadata.get("chunk_id", f"chunk_{i}"), 150),
                    "parent_id": self._safe_truncate(chunk.metadata.get("parent_id", ""), 100)
                }
                entities.append(entity)
            
            # 4. 批量插入数据
            logger.info("正在插入向量数据...")
            batch_size = 100
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i + batch_size]
                self.client.insert(
                    collection_name=self.collection_name,
                    data=batch
                )
                logger.info(f"已插入 {min(i + batch_size, len(entities))}/{len(entities)} 条数据")
            
            # 5. 创建索引
            if not self.create_index():
                return False
            
            # 6. 加载集合到内存
            self.client.load_collection(self.collection_name)
            logger.info("集合已加载到内存")
            
            # 7. 等待索引构建完成
            logger.info("等待索引构建完成...")
            time.sleep(2)
            
            logger.info(f"向量索引构建完成，包含 {len(chunks)} 个向量")
            return True
            
        except Exception as e:
            logger.error(f"构建向量索引失败: {e}")
            return False
```
#### 2）创建Collection
- 建表
```python
    def create_collection(self, force_recreate: bool = False) -> bool:
        """
        创建Milvus集合
        
        Args:
            force_recreate: 是否强制重新创建集合
        
        Returns:
            是否创建成功
        """
        try:
            # 检查集合是否存在
            if self.client.has_collection(self.collection_name):
                if force_recreate:
                    logger.info(f"删除已存在的集合: {self.collection_name}")
                    self.client.drop_collection(self.collection_name)
                else:
                    logger.info(f"集合 {self.collection_name} 已存在")
                    self.collection_created = True
                    return True
            
            # 创建集合
            schema = self._create_collection_schema()
            
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                metric_type="COSINE",  # 使用余弦相似度
                consistency_level="Strong"
            )
            
            logger.info(f"成功创建集合: {self.collection_name}")
            self.collection_created = True
            
            return True
            
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False

    def _create_collection_schema(self) -> CollectionSchema:
        """
        创建集合模式
        
        Returns:
            集合模式对象
        """
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=150, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=15000),
            FieldSchema(name="node_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="recipe_name", dtype=DataType.VARCHAR, max_length=300),
            FieldSchema(name="node_type", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="cuisine_type", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="difficulty", dtype=DataType.INT64),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=150),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=100)
        ]
        
        # 创建集合模式
        schema = CollectionSchema(
            fields=fields,
            description="中式烹饪知识图谱向量集合"
        )
        
        return schema
```
#### 3）建立 HNSW 索引
- 在向量列建立索引，索引方式：HNSW 会提前把相似向量连成一个“图”。询的时候，不需要把所有向量都比一遍，而是：先找到一个大概接近的区域然后沿着图中的邻居不断往更相似的节点走最后找到 Top-K。所以它属于：近似最近邻搜索
```python

    def create_index(self) -> bool:
        """
        创建向量索引
        
        Returns:
            是否创建成功
        """
        try:
            if not self.collection_created:
                raise ValueError("请先创建集合")
            
            # 使用prepare_index_params创建正确的IndexParams对象
            index_params = self.client.prepare_index_params()
            
            # 添加向量字段索引
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={
                    "M": 16,
                    "efConstruction": 200
                }
            )
            
            self.client.create_index(
                collection_name=self.collection_name,
                index_params=index_params
            )
            
            logger.info("向量索引创建成功")
            return True
            
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            return False
```
#### 5）加载 Collection
```python

    def load_collection(self) -> bool:
        """
        加载集合到内存
        
        Returns:
            是否加载成功
        """
        try:
            if not self.client.has_collection(self.collection_name):
                logger.error(f"集合 {self.collection_name} 不存在")
                return False
            
            self.client.load_collection(self.collection_name)
            self.collection_created = True
            logger.info(f"集合 {self.collection_name} 已加载到内存")
            return True
            
        except Exception as e:
            logger.error(f"加载集合失败: {e}")
            return False
```
#### 6）数据库查找
```python

    def similarity_search(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filters: 过滤条件
            
        Returns:
            搜索结果列表
        """
        if not self.collection_created:
            raise ValueError("请先构建或加载向量索引")
        
        try:
            # 生成查询向量
            query_vector = self.embeddings.embed_query(query)
            
            # 构建过滤表达式
            filter_expr = ""
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_conditions.append(f'{key} == "{value}"')
                    elif isinstance(value, (int, float)):
                        filter_conditions.append(f'{key} == {value}')
                    elif isinstance(value, list):
                        # 支持IN操作
                        if all(isinstance(v, str) for v in value):
                            value_str = '", "'.join(value)
                            filter_conditions.append(f'{key} in ["{value_str}"]')
                        else:
                            value_str = ', '.join(map(str, value))
                            filter_conditions.append(f'{key} in [{value_str}]')
                
                if filter_conditions:
                    filter_expr = " and ".join(filter_conditions)
            
            # 执行搜索 - 修复参数传递
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}
            }
            
            # 构建搜索参数，避免重复传递
            search_kwargs = {
                "collection_name": self.collection_name,
                "data": [query_vector],
                "anns_field": "vector",
                "limit": k,
                "output_fields": ["text", "node_id", "recipe_name", "node_type", 
                                "category", "cuisine_type", "difficulty", "doc_type",
                                "chunk_id", "parent_id"],
                "search_params": search_params
            }
            
            # 只在有过滤条件时添加filter参数
            if filter_expr:
                search_kwargs["filter"] = filter_expr
                
            results = self.client.search(**search_kwargs)
            
            # 处理结果
            formatted_results = []
            if results and len(results) > 0:
                for hit in results[0]:  # results[0]因为我们只发送了一个查询向量
                    result = {
                        "id": hit["id"],
                        "score": hit["distance"],  # 注意：在COSINE距离中，值越大相似度越高
                        "text": hit["entity"]["text"],
                        "metadata": {
                            "node_id": hit["entity"]["node_id"],
                            "recipe_name": hit["entity"]["recipe_name"],
                            "node_type": hit["entity"]["node_type"],
                            "category": hit["entity"]["category"],
                            "cuisine_type": hit["entity"]["cuisine_type"],
                            "difficulty": hit["entity"]["difficulty"],
                            "doc_type": hit["entity"]["doc_type"],
                            "chunk_id": hit["entity"]["chunk_id"],
                            "parent_id": hit["entity"]["parent_id"]
                        }
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []
```
## 7.3 智能查询路由
创建：`rag_modules/intelligent_query_router.py`
### 7.3.1 查询设计
- 优化后的系统支持多种类型的查询，这是优化后引入图数据库的关键。
**简单查询**：

- "川菜有哪些？"
- "宫保鸡丁怎么做？"
- "减肥菜推荐"

**复杂推理查询**：

- "适合糖尿病人吃的低糖川菜有哪些，并且制作时间不超过30分钟？"
- "如果我只有鸡肉和蔬菜，能做什么菜，最好是不同菜系的？"
- "哪些菜可以用豆腐替代肉类，并且保持相似的口感？"

**中等复杂查询**：

- "家常菜中哪些适合新手制作？"
- "有什么菜可以用剩余的土豆和胡萝卜？"

不同复杂度的查询需要不同的检索策略来获得最佳效果。这里设计了三种查询方式：

#### 1）传统混合检索
- 使用三路召回做RRF:
	- Dual-Level Retrieval：这是上一个版本没有的，他有两个分叉，分为实体级检索+主题级检索。例如“推荐几个减肥鸡胸肉菜”，LLM可能拆成：实体关键词：`鸡胸肉、西兰花……`，主题关键词：`减肥、低脂、高蛋白`。然后对于实体关键词：鸡胸肉，去图数据库里面找跟他相邻的节点，对于主题关键词减肥，去找匹配的边，比如鸡胸肉节点和辣子鸡节点之间的关系为：必要资源，那么对于这条边会保存一个字典{’必要资源‘：鸡胸肉节点与辣子鸡节点这条边的id}，然后匹配 主题关键词’减肥’,能否在字典里面找到。（实际上这取决于你事先是否把减肥硬编码成关系key）。
	- Milvus 向量检索：检索出来一个chunk之后根据这个chunk对应的菜谱id找到图数据库中的节点，把这个节点的所有邻居节点也拿出来。（但是如果只是一跳的话，感觉得不到新的信息，还不如直接用父文档）
	- BM25关键词匹配
#### 2）图检索
- 将用户查询通过llm翻译成图查询，遍历图之后得到的结果转换成文档，支持的查询类型要预先定义。
#### 3）组合查询
- 融合上面两种查询，然后去重，交替插入

### 7.3.2 查询路由器
#### 1）llm路由
- 对于输入的查询，通过四个维度进行分析，最终把他路由到三种查询方式：
1. 查询复杂度 (0-1)： 
	- 0.0-0.3: 简单信息查找（如：红烧肉怎么做？） 
	- 0.4-0.7: 中等复杂度（如：川菜有哪些特色菜？） 
	- 0.8-1.0: 高复杂度推理（如：为什么川菜用花椒而不是胡椒？） 
2. 关系密集度 (0-1)： 
	- 0.0-0.3: 单一实体信息（如：西红柿的营养价值） 
	- 0.4-0.7: 实体间关系（如：鸡肉配什么蔬菜？） 
	- 0.8-1.0: 复杂关系网络（如：川菜的形成与地理、历史的关系） 
3. 推理需求：是否需要多跳推理、因果分析、对比分析？ 
4. 实体识别：查询中包含多少个明确实体？
```python
class SearchStrategy(Enum):
    """搜索策略枚举"""
    HYBRID_TRADITIONAL = "hybrid_traditional"  # 传统混合检索
    GRAPH_RAG = "graph_rag"  # 图RAG检索
    COMBINED = "combined"  # 组合策略
    
@dataclass
class QueryAnalysis:
    """查询分析结果"""
    query_complexity: float  # 查询复杂度 (0-1)
    relationship_intensity: float  # 关系密集度 (0-1)
    reasoning_required: bool  # 是否需要推理
    entity_count: int  # 实体数量
    recommended_strategy: SearchStrategy
    confidence: float  # 推荐置信度
    reasoning: str  # 推荐理由

class IntelligentQueryRouter:
    """
    智能查询路由器
    
    核心能力：
    1. 查询复杂度分析：识别简单查找 vs 复杂推理
    2. 关系密集度评估：判断是否需要图结构优势
    3. 策略自动选择：路由到最适合的检索引擎
    4. 结果质量监控：基于反馈优化路由决策
    """
    
    def __init__(self, 
                 traditional_retrieval,  # 传统混合检索模块
                 graph_rag_retrieval,    # 图RAG检索模块
                 llm_client,
                 config):
        self.traditional_retrieval = traditional_retrieval
        self.graph_rag_retrieval = graph_rag_retrieval
        self.llm_client = llm_client
        self.config = config
        
        # 路由统计
        self.route_stats = {
            "traditional_count": 0,
            "graph_rag_count": 0,
            "combined_count": 0,
            "total_queries": 0
        }
        
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        深度分析查询特征，决定最佳检索策略
        """
        logger.info(f"分析查询特征: {query}")
        
        # 使用LLM进行智能分析
        analysis_prompt = f"""
        作为RAG系统的查询分析专家，请深度分析以下查询的特征：
        
        查询：{query}
        
        请从以下维度分析：
        
        1. 查询复杂度 (0-1)：
           - 0.0-0.3: 简单信息查找（如：红烧肉怎么做？）
           - 0.4-0.7: 中等复杂度（如：川菜有哪些特色菜？）
           - 0.8-1.0: 高复杂度推理（如：为什么川菜用花椒而不是胡椒？）
        
        2. 关系密集度 (0-1)：
           - 0.0-0.3: 单一实体信息（如：西红柿的营养价值）
           - 0.4-0.7: 实体间关系（如：鸡肉配什么蔬菜？）
           - 0.8-1.0: 复杂关系网络（如：川菜的形成与地理、历史的关系）
        
        3. 推理需求：
           - 是否需要多跳推理？
           - 是否需要因果分析？
           - 是否需要对比分析？
        
        4. 实体识别：
           - 查询中包含多少个明确实体？
           - 实体类型是什么？
        
        基于分析推荐检索策略：
        - hybrid_traditional: 适合简单直接的信息查找
        - graph_rag: 适合复杂关系推理和知识发现
        - combined: 需要两种策略结合
        
        返回JSON格式：
        {{
            "query_complexity": 0.6,
            "relationship_intensity": 0.8,
            "reasoning_required": true,
            "entity_count": 3,
            "recommended_strategy": "graph_rag",
            "confidence": 0.85,
            "reasoning": "该查询涉及多个实体间的复杂关系，需要图结构推理"
        }}
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=800
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            
            analysis = QueryAnalysis(
                query_complexity=result.get("query_complexity", 0.5),
                relationship_intensity=result.get("relationship_intensity", 0.5),
                reasoning_required=result.get("reasoning_required", False),
                entity_count=result.get("entity_count", 1),
                recommended_strategy=SearchStrategy(result.get("recommended_strategy", "hybrid_traditional")),
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", "默认分析")
            )
            
            logger.info(f"查询分析完成: {analysis.recommended_strategy.value} (置信度: {analysis.confidence:.2f})")
            return analysis
            
        except Exception as e:
            logger.error(f"查询分析失败: {e}")
            # 降级方案：基于规则的简单分析
            return self._rule_based_analysis(query)
```
#### 2）规则路由
- 当llm的路由分析失败后，需要一个基于规则的分析兜底：通过统计查询中“复杂度关键词”和“关系关键词”的命中比例，超过阈值就走 GraphRAG，否则走传统混合检索。
```python
    def _rule_based_analysis(self, query: str) -> QueryAnalysis:
        """基于规则的降级分析"""
        # 简单的规则判断
        complexity_keywords = ["为什么", "如何", "关系", "影响", "原因", "比较", "区别"]
        relation_keywords = ["配", "搭配", "组合", "相关", "联系", "连接"]
        
        complexity = sum(1 for kw in complexity_keywords if kw in query) / len(complexity_keywords)
        relation_intensity = sum(1 for kw in relation_keywords if kw in query) / len(relation_keywords)
        
        if complexity > 0.3 or relation_intensity > 0.3:
            strategy = SearchStrategy.GRAPH_RAG
        else:
            strategy = SearchStrategy.HYBRID_TRADITIONAL
            
        return QueryAnalysis(
            query_complexity=complexity,
            relationship_intensity=relation_intensity,
            reasoning_required=complexity > 0.3,
            entity_count=len(query.split()),
            recommended_strategy=strategy,
            confidence=0.6,
            reasoning="基于规则的简单分析"
        )
```
#### 3）智能路由执行
- 根据路由结果执行检索
```python

    def route_query(self, query: str, top_k: int = 5) -> Tuple[List[Document], QueryAnalysis]:
        """
        智能路由查询到最适合的检索引擎
        """
        logger.info(f"开始智能路由: {query}")
        
        # 1. 分析查询特征
        analysis = self.analyze_query(query)
        
        # 2. 更新统计
        self._update_route_stats(analysis.recommended_strategy)
        
        # 3. 根据策略执行检索
        documents = []
        
        try:
            if analysis.recommended_strategy == SearchStrategy.HYBRID_TRADITIONAL:
                logger.info("使用传统混合检索")
                documents = self.traditional_retrieval.hybrid_search(query, top_k)
                
            elif analysis.recommended_strategy == SearchStrategy.GRAPH_RAG:
                logger.info("🕸️ 使用图RAG检索")
                documents = self.graph_rag_retrieval.graph_rag_search(query, top_k)
                
            elif analysis.recommended_strategy == SearchStrategy.COMBINED:
                logger.info("🔄 使用组合检索策略")
                documents = self._combined_search(query, top_k)
            
            # 4. 结果后处理
            documents = self._post_process_results(documents, analysis)
            
            logger.info(f"路由完成，返回 {len(documents)} 个结果")
            return documents, analysis
            
        except Exception as e:
            logger.error(f"查询路由失败: {e}")
            # 降级到传统检索
            documents = self.traditional_retrieval.hybrid_search(query, top_k)
            return documents, analysis
    
    def _combined_search(self, query: str, top_k: int) -> List[Document]:
        """
        组合搜索策略：结合传统检索和图RAG的优势
        """
        # 分配结果数量
        traditional_k = max(1, top_k // 2)
        graph_k = top_k - traditional_k
        
        # 执行两种检索
        traditional_docs = self.traditional_retrieval.hybrid_search(query, traditional_k)
        graph_docs = self.graph_rag_retrieval.graph_rag_search(query, graph_k)
        
        # 合并和去重
        combined_docs = []
        seen_contents = set()
        
        # 交替添加结果（Round-robin）
        max_len = max(len(traditional_docs), len(graph_docs))
        for i in range(max_len):
            # 先添加图RAG结果（通常质量更高）
            if i < len(graph_docs):
                doc = graph_docs[i]
                content_hash = hashlib.md5(doc.page_content[:100].encode('utf-8')).hexdigest()
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    doc.metadata["search_source"] = "graph_rag"
                    combined_docs.append(doc)
            
            # 再添加传统检索结果
            if i < len(traditional_docs):
                doc = traditional_docs[i]
                content_hash = hashlib.md5(doc.page_content[:100].encode('utf-8')).hexdigest()
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    doc.metadata["search_source"] = "traditional"
                    combined_docs.append(doc)
        
        return combined_docs[:top_k]
```
### 7.3.3 检索策略
#### 1）传统混合检索
- 混合检索这一块的逻辑感觉并不实用，持怀疑态度，这里仅说明他的流程：
**同一个用户查询，同时走三条完全不同的召回通路——图键值双层检索、Milvus 向量检索、BM25 关键词检索，然后再把三路结果用 RRF 做统一排序，最后可选地把命中的 chunk 替换成完整父菜谱文档。** 在系统初始化阶段，它先连接 Neo4j，然后用传入的 `chunks` 建 BM25 索引；同时调用 `_build_graph_index()`，把已经解析好的 Recipe、Ingredient、CookingStep 封装成实体 KV，把 Neo4j 中的关系封装成关系 KV，并建立诸如“实体名 → entity_id”“relation key → relation_id”这样的内存索引；最后再建立 `node_id → 完整菜谱 Document` 的父文档映射。也就是说，真正开始搜索之前，它已经同时准备好了三种检索基础设施：BM25 的文本索引、Milvus 中已有的向量索引，以及 Python 内存中的图 KV 索引和 Neo4j 图数据库。

真正调用 `hybrid_search(query)` 时，代码先把每一路的候选数放大，比如最终要 `top_k=5`，它会让每一路先取至少 10 个候选，避免三路结果太少而导致融合没有意义。第一路是 `dual_level_retrieval()`，它先让 LLM 把原始 query 拆成两组关键词：一组是具体实体词，例如“鸡胸肉、宫保鸡丁”，另一组是主题词，例如“减肥、川菜、低脂”。实体词进入 `entity_level_retrieval()`，优先通过前面建立的 `entity_name → entity_id → EntityKeyValue` 查节点，命中以后再去 Neo4j 查这个节点的一跳邻居，把邻居名称附加到结果中；如果结果数量不够，再调用 Neo4j 全文索引补充 Recipe。主题词则进入 `topic_level_retrieval()`，先通过 `relation key → relation_id` 去找与主题词匹配的关系，再通过关系中的 `source_entity` 和 `target_entity` 取出具体实体内容，拼成一个“主题 + 关系 + 菜品 + 相关信息”的结果；接着它还尝试用主题词直接查 Recipe 实体，最后如果结果还是不足，就直接在 Neo4j 中用 `category / cuisineType / tags` 做属性匹配补充。最后实体级结果和主题级结果合并，按照代码中写死的 `relevance_score` 排序，并按照 `node_id` 去重，得到第一路 `dual_docs`。

第二路是 `vector_search_enhanced()`：它直接拿原始 query 到 Milvus 做 embedding 相似度搜索，找到语义最相近的 chunk；每个 Milvus 结果里又带有对应菜谱的 `node_id`，所以代码会拿这个 `node_id` 再去 Neo4j 查最多几个一跳邻居，把诸如食材、步骤之类的邻居名称追加到原 chunk 后面，因此作者把它称为“增强向量检索”。第三路是 `bm25_search()`：先对 query 用 jieba 分词并去停用词，然后用 `BM25Okapi.get_scores()` 给所有 chunk 打关键词匹配分，按分数从高到低返回结果。这两路和第一路的区别很明显：dual-level 主要依赖图 KV 和 Neo4j，vector 主要依赖语义相似度，BM25 主要依赖字面关键词匹配，所以三路的目标就是互补召回。 粘贴的 Markdown (1)

第二路是 `vector_search_enhanced()`：它直接拿原始 query 到 Milvus 做 embedding 相似度搜索，找到语义最相近的 chunk；每个 Milvus 结果里又带有对应菜谱的 `node_id`，所以代码会拿这个 `node_id` 再去 Neo4j 查最多几个一跳邻居，把诸如食材、步骤之类的邻居名称追加到原 chunk 后面，因此作者把它称为“增强向量检索”。第三路是 `bm25_search()`：先对 query 用 jieba 分词并去停用词，然后用 `BM25Okapi.get_scores()` 给所有 chunk 打关键词匹配分，按分数从高到低返回结果。这两路和第一路的区别很明显：dual-level 主要依赖图 KV 和 Neo4j，vector 主要依赖语义相似度，BM25 主要依赖字面关键词匹配，所以三路的目标就是互补召回。 粘贴的 Markdown (1)
#### 2）图检索
- 用于复杂推理查询，这也是引入图数据库带来的。主要定义如下查询类型：
	- 实体关系查询：A和B有什么关系？
	- 多跳查询：A通过什么连接到C？
	- 子图查询：A相关的所有信息
	- 路径查找：从A到B的最佳路径
	- 聚类查询：和A相似的都有什么？
图检索主要支持如下功能：
- 第一块是**图查询意图理解**。用户输入自然语言问题后，`understand_graph_query()` 会调用 LLM，把问题转换成结构化的 `GraphQuery`，主要提取 `query_type、source_entities、target_entities、relation_types、max_depth`。比如“鸡肉配什么蔬菜？”会被转换成“以鸡肉为起点、蔬菜为目标、最多走 3 跳的 multi-hop 查询”。也就是说，这一步负责把“自然语言问题”翻译成“图数据库应该怎么查”。 

- 第二块是**路径型图检索**，主要由 `multi_hop_traversal()` 完成。它拿上一阶段生成的 `GraphQuery`，从 `source_entities` 对应的 Neo4j 节点出发，在 `1~max_depth` 跳范围内寻找路径；如果有 `target_entities`，还会限制路径终点的 `name/category`。找到路径以后，再按照“路径长度、路径节点的度数、是否包含指定关系类型”计算一个人工相关性分数，排序取前 20 条，最后包装成 `GraphPath`。例如“鸡肉配什么蔬菜”就可能找到 `鸡肉 ← REQUIRES — 宫保鸡丁 — REQUIRES → 胡萝卜` 这样的两跳路径。这个部分是这份文件最核心、也是真正执行图结构检索的部分。 

- 第三块是**局部知识子图提取**。`extract_knowledge_subgraph()` 不再寻找某一条具体路径，而是以某个实体为中心，把 `max_depth` 跳以内的一批节点和关系整体取出来。例如查询“川菜有什么特色”，就可能以“川菜”为中心，把相关菜谱、食材、步骤等两跳范围内的节点一起取出来，并统计 `node_count、relationship_count、density` 等简单图指标，最后封装成 `KnowledgeSubgraph`。因此前面的 `multi_hop_traversal()` 是“找一条条路径”，这里则是“捞出一片局部图”。 
#### 3）组合检索
- 使用两种检索策略，然后交叉合并
```python
    def _combined_search(self, query: str, top_k: int) -> List[Document]:
        """
        组合搜索策略：结合传统检索和图RAG的优势
        """
        # 分配结果数量
        traditional_k = max(1, top_k // 2)
        graph_k = top_k - traditional_k
        
        # 执行两种检索
        traditional_docs = self.traditional_retrieval.hybrid_search(query, traditional_k)
        graph_docs = self.graph_rag_retrieval.graph_rag_search(query, graph_k)
        
        # 合并和去重
        combined_docs = []
        seen_contents = set()
        
        # 交替添加结果（Round-robin）
        max_len = max(len(traditional_docs), len(graph_docs))
        for i in range(max_len):
            # 先添加图RAG结果（通常质量更高）
            if i < len(graph_docs):
                doc = graph_docs[i]
                content_hash = hashlib.md5(doc.page_content[:100].encode('utf-8')).hexdigest()
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    doc.metadata["search_source"] = "graph_rag"
                    combined_docs.append(doc)
            
            # 再添加传统检索结果
            if i < len(traditional_docs):
                doc = traditional_docs[i]
                content_hash = hashlib.md5(doc.page_content[:100].encode('utf-8')).hexdigest()
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    doc.metadata["search_source"] = "traditional"
                    combined_docs.append(doc)
        
        return combined_docs[:top_k]
```