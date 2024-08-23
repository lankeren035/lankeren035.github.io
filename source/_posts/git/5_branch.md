---
title: 5- Git分支

date: 2024-8-23 13:00:00

tags: [Git]

categories: [Git]

comment: true

toc: true


---

#### 

<!--more-->



# 5- Git分支

- 分支在Git中相对较难，如果两个分支互不干扰，对你没啥影像，如果两个分支有交集就需要处理一些问题。



## 5.1git分支中常用指令

```bash
# 列出所有本地分支
git branch

# 列出所有远程分支
git branch -r

# 新建一个分支，但依然停留在当前分支
git branch [branch-name]

# 新建一个分支并切换到该分支
git checkout -b [branch]

# 合并指定分支到当前分支
git merge [branch]

# 删除分支
git branch -d [branch-name]

# 删除远程分支
git push origin --delete [branch-name]
git branch -dr [romte/branch]
```



- 如果多个分支并行执行，就会导致我们代码冲突，也就是同时存在多个版本
- 如果同一个文件在合并分支时都被修改了，则会引起冲突
  - 解决办法时我们可以修改冲突文件后重新提交
- master主分支应该非常稳定，用来发布系版本，一般情况下不允许在上面工作，工作一般情况下在新建的dev分支上工作，工作完后比如要发布，或者说dev分支代码稳定后可以合并到主分支master上来