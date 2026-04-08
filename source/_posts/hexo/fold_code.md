---
title: hexo代码块折叠
date: 2026-04-05
tags:
  - hexo
categories:
  - hexo
comment: true
toc: true
published: true
hexo-path:
permalink: hexo
---

#
<!--more-->

1. 先安装插件：npm install hexo-fold --save

2. 如果 Hexo 没有自动识别，再在站点根目录 `_config.yml` 里加：
	```json
	plugins:  
	   hexo-fold 
	```

3. 你文章里就直接这样写：
	```markdown
	
{% fold "展开查看代码" %}  
code
{% endfold %}
	```
