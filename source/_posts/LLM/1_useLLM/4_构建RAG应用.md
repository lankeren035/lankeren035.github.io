---
title: 4_构建RAG应用
date: 2026-09-01
tags:
  - llm
categories:
  - llm
comment: true
toc: true
published: true
permalink: llm/1_use_LLM
hexo-path:
---

#
<!--more-->

# 1. 将LLM接入LangChain

## 1.1 基于LangChain调用ChatGPT
1. 实例化一个 ChatOpenAI 类，可以在实例化时传入超参数来控制回答，例如 `temperature` 参数。
```python
from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(
	model='glm-5',
	temperature=0.8,
	api_key = os.environ['OPENAI_API_KEY'],
	base_url = "https://api.lkeap.cloud.tencent.com/coding/v3"
)
print(llm)
```
几种常用的超参数设置包括：
- model_name：所要使用的模型，默认为 ‘gpt-3.5-turbo’，参数设置与 OpenAI 原生接口参数设置一致。 
- temperature：温度系数，取值同原生接口。 
- openai_api_key：OpenAI API key，如果不使用环境变量设置 API Key，也可以在实例化时设置。 
- openai_proxy：设置代理，如果不使用环境变量设置代理，也可以在实例化时设置。 
- streaming：是否使用流式传输，即逐字输出模型回答，默认为 False，此处不赘述。 
- max_tokens：模型输出的最大 token 数，意义及取值同上。

2. 使用llm：
```python
output = llm.invoke("请你自我介绍一下自己！")
print(output)
```

3. 提示词模板。大多数情况下不会直接将用户的输入直接传递给 LLM。
```python
prompt = """请你将由三个反引号分割的文本翻译成英文！\
text: ```{text}```
"""

text = "我带着比身体重的行李，游入尼罗河底，经过几道闪电 看到一堆光圈，不确定是不是这里。"
prompt.format(text=text)
print(prompt)
```

4. 提示词模板样例
```python
from langchain_core.prompts import ChatPromptTemplate

template = "你是一个翻译助手，可以帮助我将 {input_language} 翻译成 {output_language}."
human_template = "{text}"

chat_prompt = ChatPromptTemplate([
	('system', template),
	('human', human_template),
])

text = "我带着比身体重的行李，\
游入尼罗河底，\
经过几道闪电 看到一堆光圈，\
不确定是不是这里。\
"

messages = chat_prompt.invoke({
	'input_language': '中文',
	'output_language':'英文',
	'text':text,
})
print(messages)

# 使用定义好的模板和用户提示词
output = llm.invoke(messages)
print(output)
```

5. 格式化模型输出。OutputParsers 将语言模型的原始输出转换为可以在下游使用的格式：
	- 将 LLM 文本转换为结构化信息（例如 JSON）
	- 将 ChatMessage 转换为字符串
	- 将除消息之外的调用返回的额外信息（如 OpenAI 函数调用）转换为字符串
```python
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
output_parser.invoke(output)
```

6. 将上述的三个过程：1）定义提示词模板 2）提示词输入llm， 3）解析llm输出打包。 LangChain 里用 `|` 把几个步骤串成一条处理链。
```python
chain = chat_prompt | llm | output_parser
chain.invoke({
	'input_language': '中文',
	'output_language':'英文',
	'text':text,
})
```
LCEL 是 LangChain 的“管道式链构建语法”；`|` 负责搭链，`invoke/batch/stream` 等负责执行这条链。链接之后前一个的输出作为后一个的输入。invoke函数就类似网络的forward函数。

# 2. 构建检索问答链

## 2.1 加载向量数据库

```python
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    openai_api_key=os.environ["SILICONFLOW_API_KEY"],
    openai_api_base="https://api.siliconflow.cn/v1",
    check_embedding_ctx_length=False
)
persist_directory = './data_base/my_vector_db/chroma'

from langchain_community.vectorstores import Chroma
# 加载数据库
vectordb = Chroma(
	#documents = split_docs, 
	embedding_function = embedding,
	persist_directory=persist_directory
)
print(f"向量库中存储的数量：{vectordb._collection.count()}")
```
1. 把向量数据库构造成检索器。然后进行检索
```python
question = "什么是prompt engineering?"
retriever = vectordb.as_retriever(search_kwargs={"k":3})
docs = retriever.invoke(question)
print(f"检索到的内容数：{len(docs)}")
for i, doc in enumerate(docs): 
	print(f"检索到的第{i}个内容: \n {doc.page_content}", end="\n-----------------------------------------------------\n")
```
## 2.2 创建检索链
```python
from langchain_core.runnables import RunnableLambda
def combine_docs(docs):
	return '\n\n'.join(doc.page_content for doc in docs)
	
combiner = RunnableLambda(combine_docs)
retrieval_chain = retriever | combiner

retrieval_chain.invoke('南瓜书是什么')
```
## 2.3 创建LLM
```python
from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(
	model='glm-5',
	temperature=0,
	api_key = os.environ['OPENAI_API_KEY'],
	base_url = "https://api.lkeap.cloud.tencent.com/coding/v3"
)
llm.invoke("请你自我介绍一下自己！").content
```
## 2.4 构建检索问答链
```python
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# 1. 提示词模板
template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。最多使用三句话。尽量使答案简明扼要。请你在回答的最后说“谢谢你的提问！”。 {context} 问题: {input} """
prompt = PromptTemplate(template=template)

# 2. 链接
qa_chain = (
	RunnableParallel({'context':retrieval_chain, 'input':RunnablePassthrough()})
	| prompt
	| llm
	| StrOutputParser()
)

# 3. 测试
question_1 = "什么是南瓜书？"
question_2 = "Prompt Engineering for Developer是谁写的？"
result = qa_chain.invoke(question_1)
print("大模型+知识库后回答 question_1 的结果：")
print(result)
result = qa_chain.invoke(question_2)
print("大模型+知识库后回答 question_2 的结果：")
print(result)
print('原始llm回答1：',llm.invoke(question_1).content)
print('原始llm回答2：',llm.invoke(question_2).content)

```
- template有两个变量：{context}  {input} ，所以后面传给 `prompt` 的东西必须长这样：
```json
{
    "context": "...检索到的知识...",
    "input": "南瓜书是什么？"
}
```
1. 关键在代码RunnableParallel怎么理解，它的意思是：对同一个输入，同时走两条路。
	1. 假设：question = "南瓜书是什么？" ，他作为输入，输入到两条路中。
	2. 第一条：'context': retrieval_chain相当于：context = retrieval_chain.invoke("南瓜书是什么？") 得到检索结果
	3. 第二条：'input': RunnablePassthrough() 什么都不做，输入什么就原样输出什么。input = "南瓜书是什么？"
	4. 最终 `RunnableParallel` 输出：
```json
{
    "context": "检索到的知识内容……",
    "input": "南瓜书是什么？"
}
```

## 2.5 向检索链添加聊天记录
使用 LangChain 中的`ChatPromptTemplate`，将先前的对话嵌入到语言模型中，使其具有连续对话的能力。 这些历史记录添加到上下文中。
```python
from langchain_core.prompts import ChatPromptTemplate

# 问答链的系统prompt
system_prompt = ( 
	"你是一个问答任务的助手。 "
	"请使用检索到的上下文片段回答这个问题。 "
	"如果你不知道答案就说不知道。 "
	"请使用简洁的话语回答用户。"
	"\n\n"
	"{context}"
) 

# 制定prompt template
qa_prompt = ChatPromptTemplate(
	[
		("system", system_prompt),
		("placeholder", "{chat_history}"),
		("human", "{input}"),
	]
)

# 1. 无历史记录
messages = qa_prompt.invoke(
	{
		"input": "南瓜书是什么？",
		"chat_history": [],
		"context": ""
	}
)
for message in messages.messages:
	print(message.content)

# 2. 有历史记录
messages = qa_prompt.invoke(
	{
		"input": "你可以介绍一下他吗？",
		"chat_history": [
			("human", "西瓜书是什么？"),
			("ai", "西瓜书是指周志华老师的《机器学习》一书，是机器学习领域的经典入门教材之一。"),
		],
		"context": ""
	}
)
for message in messages.messages:
	print(message.content)
```
## 2.6 带有信息压缩的检索链
多轮对话中，可能需要用到模型输出的结果。例如：
```text
用户：西瓜书是什么？
ai：西瓜书是指周志华老师的《机器学习》一书，是机器学习领域的经典入门教材之一。
用户：你可以介绍一下他吗？
```
此时用户想问的其实是“你可以介绍下周志华老师吗？”的意思。为了解决这个问题我们将采取信息压缩的方式，让llm根据历史记录完善用户的问题。
```python
from langchain_core.runnables import RunnableBranch

## 1. prompt模板
# 压缩问题的系统prompt
condense_question_system_template = (
	"请根据聊天记录完善用户最新的问题，"
	"如果用户最新的问题不需要完善则返回用户的问题。"
)
# 构造 压缩问题的 prompt template
condense_question_prompt = ChatPromptTemplate([
	("system", condense_question_system_template),
	("placeholder", "{chat_history}"),
	("human", "{input}"),
])

## 2. 构造检索链
#RunnableBranch(
#    (条件1, 执行链1),
#    (条件2, 执行链2),
#    默认执行链
#)
retrieve_docs = RunnableBranch( #根据条件选择要运行的分支
	# 分支 1: 若聊天记录中没有 chat_history 则直接使用用户问题查询向量数据库
	(lambda x: not x.get('chat_history', False), (lambda x: x['input']) | retriever,),
	# 分支 2 : 若聊天记录中有 chat_history 则先让 llm 根据聊天记录完善问题再查询向量数据库
	condense_question_prompt | llm | StrOutputParser() | retriever,
)
```

**支持聊天记录的检索问答链**： 之前创建的问答链，输入是一个字符串，然后RunnableParallel得到字典，字典输入模板，模板结果再输入llm。 这次要加入上下文，因此原始输入变成了一个字典：
```text
{
"input": "西瓜书是什么？",
"chat_history": []
}
```
先把这个字典输入给检索链，得到检索的context，然后放回字典
```text
{
"input": "西瓜书是什么？",
"chat_history": [],
"context": [片段1， 片段2， 片段3]
}
```
使用assing输入一个字典，处理后得到的结果作为新的字段放回输入字典
```python
from langchain_core.runnables import RunnablePassthrough
# 这里原始输入就是字典了，因此需要修改函数
def combine_docs(docs):
	return "\n\n".join(doc.page_content for doc in docs["context"]) # 将 docs 改为 docs["context"]
	
# 定义问答链
qa_chain = (
	RunnablePassthrough.assign(context=combine_docs) # 使用 combine_docs 函数整合 qa_prompt 中的 context项
	| qa_prompt
	| llm
	| StrOutputParser()
)
# 加入检索链和历史记录
qa_history_chain = RunnablePassthrough.assign(
	context = (lambda x: x) | retrieve_docs # 第一段输入x输出x，输出的x作为第二段的输入，第二段查询到结果。查询结果赋值为context字段放回输入字典. 由此构造qachain的输入
).assign(answer=qa_chain) # 输入内容经过qa_chain处理之后的结果赋值为answer字段

# 1. 第一次测试不带聊天记录
qa_history_chain.invoke({
	"input": "西瓜书是什么？",
	"chat_history": []
})

# 2. 第二次测试，将上一条测试的结果作为历史
qa_history_chain.invoke({
	"input": "南瓜书跟它有什么关系？",
	"chat_history": [
		("human", "西瓜书是什么？"),
		("ai", "西瓜书是指周志华老师的《机器学习》一书，是机器学习领域的经典入门教材之一。"),
	]
})
```
可以看到，LLM 准确地判断了“它”是什么，代表着我们成功地传递给了它历史信息。另外召回的内容也有着问题的答案，证明我们的信息压缩策略也起到了作用。这种关联前后问题及压缩信息并检索的能力，可大大增强问答系统的连续性和智能水平。

# 3. 部署知识库助手
## 3.1 Streamlit
一个用于快速创建数据应用程序的开源 Python 库。和常规 Web 框架，如 Flask/Django 的不同之处在于，它不需要你去编写任何客户端代码（HTML/CSS/JS），只需要编写普通的 Python 模块，就可以在很短的时间内创建美观并具备高度交互性的界面，从而快速生成数据分析或者机器学习的结果；另一方面，和那些只能通过拖拽生成的工具也不同的是，你仍然具有对代码的完整控制权。
- st.write()：这是最基本的模块之一，用于在应用程序中呈现文本、图像、表格等内容。
    
- st.title()、st.header()、st.subheader()：这些模块用于添加标题、子标题和分组标题，以组织应用程序的布局。
    
- st.text()、st.markdown()：用于添加文本内容，支持 Markdown 语法。
    
- st.image()：用于添加图像到应用程序中。
    
- st.dataframe()：用于呈现 Pandas 数据框。
    
- st.table()：用于呈现简单的数据表格。
    
- st.pyplot()、st.altair_chart()、st.plotly_chart()：用于呈现 Matplotlib、Altair 或 Plotly 绘制的图表。
    
- st.selectbox()、st.multiselect()、st.slider()、st.text_input()：用于添加交互式小部件，允许用户在应用程序中进行选择、输入或滑动操作。
    
- st.button()、st.checkbox()、st.radio()：用于添加按钮、复选框和单选按钮，以触发特定的操作。
## 3.2 构建应用程序
首先，创建一个新的 Python 文件并将其保存为 streamlit_app.py再根目录
```python
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_community.vectorstores import Chroma
import os
import sys

# 1. 定义检索器
def get_retriever():
	embedding = OpenAIEmbeddings(
	    model="BAAI/bge-m3",
	    openai_api_key=os.environ["SILICONFLOW_API_KEY"],
	    openai_api_base="https://api.siliconflow.cn/v1",
	    check_embedding_ctx_length=False
	)
	persist_directory = './data_base/my_vector_db/chroma'
	# 加载数据库
	vectordb = Chroma(
		#documents = split_docs, 
		embedding_function = embedding,
		persist_directory=persist_directory
	)
	print(f"向量库中存储的数量：{vectordb._collection.count()}")
	return vectordb.as_retriever()
	
# 2. 检索结果收集函数
def combine_docs(docs):
	return '\n\n'.join( doc.page_content for doc in docs['context'])
	
# 3. 定义检索问答链
def get_qa_history_chain():
	retriever = get_retriever()
	llm = ChatOpenAI(
		model='glm-5',
		temperature=0,
		api_key = os.environ['OPENAI_API_KEY'],
		base_url = "https://api.lkeap.cloud.tencent.com/coding/v3"
	)
	condense_question_system_template = (
		"请根据聊天记录总结用户最近的问题，"
		"如果没有多余的聊天记录则返回用户的问题。"
	)
	condense_question_prompt = ChatPromptTemplate([
		('system', condense_question_system_template),
		('placeholder','{chat_history}'),
		('human', '{input}'),
	])
	retrieve_docs = RunnableBranch(
		(lambda x: not x.get('chat_history',False), (lambda x: x['input']) | retriever),
		condense_question_prompt | llm | StrOutputParser() | retriever,
	)
	
	system_prompt = (
		"你是一个问答任务的助手。 "
		"请使用检索到的上下文片段回答这个问题。 "
		"如果你不知道答案就说不知道。 "
		"请使用简洁的话语回答用户。"
		"\n\n"
		"{context}"
	)
	qa_prompt = ChatPromptTemplate.from_messages([
		('system', system_prompt),
		('placeholder','{chat_history}'),
		('human', '{input}'),
	])
	qa_chain = (
		RunnablePassthrough().assign(context=combine_docs)
		| qa_prompt
		| llm
		| StrOutputParser()
	)
	
	qa_history_chain = RunnablePassthrough().assign(
		context = retrieve_docs,
	).assign(answer=qa_chain)
	return qa_history_chain
	
# 4. 接受检索问答链、用户输入及聊天历史，并以流式返回该链输出
def get_response(chain, input, chat_history):
 response = chain.stream({ #invoke是一次性返回所有结果， stream是边生成边返回
	 'input': input,
	 'chat_history': chat_history
 })
 for res in response:
	 if 'answer' in res.keys():
		 yield res['answer']
		 
# 5. 制定显示效果与逻辑
def main():
	st.markdown('### 🦜🔗 动手学大模型应用开发')
	# 1. 存储对话历史
	if 'messages' not in st.session_state:
		st.session_state.messages = []
	
	# 2. 存储检索问答链
	if 'qa_history_chain' not in st.session_state:
		st.session_state.qa_history_chain = get_qa_history_chain()
		
	# 3. 建立容器，高度500px
	chat_container = st.container(height=550)
	
	# 4. 显示整个历史对话
	for message in st.session_state.messages: #遍历对话历史
		# 假设当前记录：
		#    ('human', '你好'),
		#    ('ai', '你好，有什么可以帮助你的？')
		# 那么遍历到第一行的时候有两个元素[0]是身份， [1]是内容
		# with关键字在chat_container容器里面创建了一个用户聊天气泡，然后把[1]的内容显示进去
		with chat_container.chat_message(message[0]):
			st.write(message[1]) #打印内容
	# := 海象运算符 ， 同时完成赋值与判断
	# chat_input会在页面下面生成输入框
	if prompt := st.chat_input('Say something'):
		chat_history = st.session_state.messages.copy()
		# 将用户输入添加到对话历史
		st.session_state.messages.append(('human', prompt))
		# 显示当前用户输入
		with chat_container.chat_message('human'):
			st.write(prompt)
		
		# 生成回复
		answer = get_response(
			chain = st.session_state.qa_history_chain,
			input = prompt,
			chat_history=chat_history
		)
		# 流式输出
		with chat_container.chat_message('ai'):
			output = st.write_stream(answer)
		# 将输出存入st.session_state.messages
		st.session_state.messages.append(('ai', output))
		
if __name__ == "__main__":
    main()
```
- 用户每进行一次交互，Streamlit 基本都会把整个 Python 脚本重新执行一遍。比如第一次打开网页：运行main()，用户输入’你好‘，Streamlit 又会：重新运行 main()。st.session_state则在这多次运行之间共享。

优化方向：
- 界面中添加上传本地文档，建立向量数据库的功能
- 添加多种LLM 与 embedding方法选择的按钮
- 添加修改参数的按钮
- 更多......