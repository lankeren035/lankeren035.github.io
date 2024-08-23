\---

title: 1- git基础

date: 2024-8-21 13:00:00

tags: [Git]

categories: [Git]

comment: true

toc: true

\---



**####** 



<!--more-->





# 1- git基础

## 1.1 版本控制

- 多人开发时需要保存不同的版本
- 常见版本控制器：
  - Git
  - SVN (subversion)
  - CVS (Concurrent Versions System)
  - VSS (Micorosoft visual SourceSafe)
  - TFS (Team Foundation Server)
  - Visual Studio Online

### 1.1.1 本地版本控制

- 记录文件每次的更新，对每个版本做一个快照：`版本1， 版本2， 版本3`

### 1.1.2 集中版本控制 （SVN）

- 所有的版本数据都保存在服务器上，协同开发者从服务器同步更新或上传自己的修改
- 所有的版本数据都存在服务器上，用户只有自己一千所同步的版本，如果不联网，用户就看不到历史版本。所有数据保存在单一服务器上，有风险。`svn, cvs, vss`

### 1.1.3 分布式版本控制 （Git)

- 所有版本信息仓库全部同步到本地的每个用户。



## 1.2 Git历史

- Linux内核开源项目有众多参与者。绝大多数的Linux内核维护工作都花在了提交补丁和保存归档的繁琐事务上（1911-2002）由一个人负责。2002年，整个项目组开始启用一个专有的分布式版本控制系统BitKeeper来管理和维护代码。由于Linux社区中有很多大佬研究破解BitKeeper，2005年，开发BitKeeper的商业公司同Linux内核开源社区的合作关系结束，收回了Linux内核社区免费使用BitKeeper的权力。迫使Linux开源社区（特别是Linux的缔造者Linux Torvalds)基于使用BitKeeper时的经验教训，开发出了自己的版本系统，也就是后来的Git。



