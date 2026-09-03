---
title: 2_使用LLM API开发应用
date: 2026-08-31
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

# 1. 基本概念
## 1.1 Prompt
- 我们每一次访问大模型的输入为一个 Prompt，而大模型给我们的返回结果则被称为 Completion。
## 1.2 Temperature
- Temperature 一般取值在 0~1 之间，当取值较低接近 0 时，预测的随机性会较低，产生更保守、可预测的文本，不太可能生成意想不到或不寻常的词。当取值较高接近 1 时，预测的随机性会较高，所有词被选择的可能性更大，会产生更有创意、多样化的文本，更有可能生成不寻常或意想不到的词。
- 对于不同的问题与应用场景，我们可能需要设置不同的 temperature。例如，在本教程搭建的个人知识库助手项目中，我们一般将 temperature 设置为 0，从而保证助手对知识库内容的稳定使用，规避错误内容、模型幻觉；在产品智能客服、科研论文写作等场景中，我们同样更需要稳定性而不是创造性；但在个性化 AI、创意营销文案生成等场景中，我们就更需要创意性，从而更倾向于将 temperature 设置为较高的值。
## 1.3 System Prompt
- 你可以设置两种 Prompt：一种是 System Prompt，该种 Prompt 内容会在整个会话过程中持久地影响模型的回复，且相比于普通 Prompt 具有更高的重要性；另一种是 User Prompt，这更偏向于我们平时提到的 Prompt，即需要模型做出回复的输入。
- 我们一般设置 System Prompt 来对模型进行一些初始化设定，例如，我们可以在 System Prompt 中给模型设定我们希望它具备的人设如一个个人知识库助手等。System Prompt 一般在一个会话中仅有一个。在通过 System Prompt 设定好模型的人设或是初始设置后，我们可以通过 User Prompt 给出模型需要遵循的指令。例如，当我们需要一个幽默风趣的个人知识库助手，并向这个助手提问我今天有什么事时，可以构造如下的 Prompt：
```json
{
    "system prompt": "你是一个幽默风趣的个人知识库助手，可以根据给定的知识库内容回答用户的提问，注意，你的回答风格应是幽默风趣的",
    "user prompt": "我今天有什么事务？"
}
```
# 2. 使用LLM API

1. 调用openai api

```python
# 1. 调用api（确保你的api key可以被读取到）
# 可以将apiKey 放到 .env文件然后读取或者在codebase中可以手动添加环境变量

from openai import OpenAI
import os

client = OpenAI(
	api_key=os.environ.get("OPENAI_API_KEY"),
	# 如果用的别的api可以在这里指定baseurl
	base_url="https://api.lkeap.cloud.tencent.com/coding/v3"
)

# 导入所需库
# 注意，此处我们假设你已根据上文配置了 OpenAI API Key，如没有将访问失败
completion = client.chat.completions.create(
    # 调用模型：ChatGPT-4o
    model="glm-5",
    # messages 是对话列表
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

print(completion)
print(completion.choices[0].message.content)
```

调用该 API 会返回一个 ChatCompletion 对象，其中包括了回答文本、创建时间、id 等属性。我们一般需要的是回答文本，也就是回答对象中的 content 信息。

调用 API 常会用到的几个参数：
- model，即调用的模型，一般取值包括“gpt-3.5-turbo”（ChatGPT-3.5）、“gpt-3.5-turbo-16k-0613”（ChatGPT-3.5 16K 版本）、“gpt-4”（ChatGPT-4）、“gpt-4o”（ChatGPT-4o）、。注意，不同模型的成本是不一样的。
- messages，即我们的 prompt。ChatCompletion 的 messages 需要传入一个列表，列表中包括多个不同角色的 prompt。我们可以选择的角色一般包括 system：即前文中提到的 system prompt；user：用户输入的 prompt；assistant：助手，一般是模型历史回复，作为提供给模型的参考内容。
- temperature，温度。即前文中提到的 Temperature 系数。
- max_tokens，最大 token 数，即模型输出的最大 token 数。OpenAI 计算 token 数是合并计算 Prompt 和 Completion 的总 token 数，要求总 token 数不能超过模型上限（如默认模型 token 上限为 4096）。因此，如果输入的 prompt 较长，需要设置较大的 max_token 值，否则会报错超出限制长度。

如下是一个简单的封装 OpenAI 接口的函数，支持我们直接传入 prompt 并获得模型的输出：
```python
from openai import OpenAI
import os

client = OpenAI(
	api_key = os.environ.get('OPENAI_API_KEY') ,
	    base_url="https://api.lkeap.cloud.tencent.com/coding/v3"
)

def gen_gpt_messages(prompt):
	messages = [
		{'role': 'user', 'content':prompt}
	]
	return messages
	
def get_completion(prompt, model='glm-5', temperature=0):
	response = client.chat.completions.create(
		model = model,
		messages=gen_gpt_messages(prompt),
		temperature = temperature
	)
	if len(response.choices) > 0:
		return response.choices[0].message.content
	return 'error'
	
get_completion('你好')
```

# 3. Prompt Engineering
设计高效 Prompt 的两个关键原则：**编写清晰、具体的指令**和**给予模型充足思考时间**。
## 3.1 使用分隔符
- 你可以选择用 ```，"""，< >， ，: 等做分隔符，只要能明确起到隔断作用即可。
- 寻求结构化的输出
- 要求模型检查是否满足条件
- 提供少量示例
## 3.2 给模型时间去思考
可以要求其先列出对问题的各种看法，说明推理依据，然后再得出最终结论。在 Prompt 中添加逐步推理的要求，能让语言模型投入更多时间逻辑思维，输出结果也将更可靠准确。
- 指导模型在下结论之前找出一个自己的解法