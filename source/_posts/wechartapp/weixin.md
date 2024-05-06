---
title: 微信小程序基础
date: 2021-03-01 20:00:00
toc: true
tags: [微信小程序]
categories: [微信小程序]

---
#

<!--more-->

## 1.总体结构

### （1）概述
#### 1)项目结构
- page：存放所有页面
- utils: 存放工具模块（如格式化时间的自定义模块）
- app.js: 项目入口
- app.json: 项目全局配置
- app.wxss: 项目全局样式
- project.config.json: 项目配置文件
- sitemap.json: 配置小程序及其页面是否允许被微信索引

#### 2）页面组成
- .js文件：页面脚本，存放页面数据、事件处理函数等
- .json文件：配置文件，配置窗口外观、表现等
- .wxml文件：模板结构文件
- .wxss文件: 样式表文件

### （2）详述
#### 1）json文件
 json是一种数据格式，在开发中json总是以配置文件的形式出现。
|app.json|afadsf|
|-------------------|------|
|project.config.json|
|sitemap.json|
|每个页面的json|