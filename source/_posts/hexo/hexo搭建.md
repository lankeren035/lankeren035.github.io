---
title: hexo博客搭建
toc: true
categories: [博客]
tags: [博客,hexo]
---
## 1.安装git

 <!-- more --> 

- 输入网址https://npm.taobao.org/mirrors/git-for-windows/（你也可以去官网下载）
- 点击你想下载的版本
![](../../../themes/yilia/source/img/hexo/6.png)
![](./img/hexo/6.png)
- 下载exe文件
![](../../../themes/yilia/source/img/hexo/7.png)
![](./img/hexo/7.png)
- 一路next即可
![](../../../themes/yilia/source/img/hexo/8.png)
![](./img/hexo/8.png)
## 2.安装nodejs
- 输入网址：nodejs.org
- 按装左边长期支持版本
![](../../../themes/yilia/source/img/hexo/1.png)
![](./img/hexo/1.png)
- 安装点击next即可
## 3.通过cnpm安装hexo
- 进入控制台使用管理员身份
- 输入：npm install -g cnpm --registry=https://registry.npm.taobao.org<回车>
![](../../../themes/yilia/source/img/hexo/2.png)
![](./img/hexo/2.png)
- 输入：cnpm install -g hexo-cli<回车>
## 4.搭建博客
- 创建一个空文件夹(d:/blog/test)
- 进入该文件夹：cd /d d:/blog/test<回车>
- 初始化：hexo init<回车>
- 打开：hexo s<回车>
- 浏览器访问：http://localhost:4000/
- 命令行点击ctr+c输入y<回车>即可关闭
## 5.操作博客
- 新建博客：hexo n "test1.md"(保存在了blog\test\source\_posts\test1.md)（也可直接在该目录下新建.md文件）
- 编辑博客：使用vscode/typora等
## 6.将博客部署到GitHub
- 登录GitHub
- 新建仓库
![](../../../themes/yilia/source/img/hexo/3.png)
![](./img/hexo/3.png)
- 注意前面部分要跟自己的名字一样（以后通过
![](../../../themes/yilia/source/img/hexo/4.png)
![](./img/hexo/4.png)
- 安装插件：命令行输入：cnpm install --save hexo-deployer-git
- 找到_config.yml文件,将最后面加上：
```
type: git
  repo: git@github.com:lankeren035/lankeren035.github.io.git
  branch: main
```
- 输入hexo d即可部署到远端（在test文件夹下）
- 浏览器输入仓库名：lankeren035.github.io即可访问

## 7.换主题
- 找到目标主题：github.com/litten/hexo-theme-yilia
- 命令行输入：git clone https://github.com/litten/hexo-theme-yilia.git themes/yilia则会在theme下创建yilia文件夹
- 在_config.yml中将theme后的改为yilia
![](../../../themes/yilia/source/img/hexo/5.png)
![](./img/hexo/5.png)
- hexo g再hexo s通过本地看看
- 最后hexo d再把远端的也更新了