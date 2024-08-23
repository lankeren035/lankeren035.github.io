---
title: git使用

date: 2024-8-12 14:00:00

tags: [Git]

categories: [Git]

comment: true

toc: true
---



####



<!--more-->





# git使用



## 1. 删除远端文件夹

```bash
git rm -r --cached source/_posts/deeplearning/code/pytorch/data
git commit -m "Remove data folder from remote repository"
git push
```



## 2. 使用.gitignore

- 将不想上传的路径写到.gitignore文件里面