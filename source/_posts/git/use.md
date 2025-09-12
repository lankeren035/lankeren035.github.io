---
title: "git使用"

date: 2024-8-12 14:00:00

tags: [Git]

categories: [Git]

comment: true

toc: true
---



####



<!--more-->





# git使用



## 1. 删除远端文件夹（本地保留）

```bash
git rm -r --cached source/_posts/deeplearning/code/pytorch/data
git commit -m "Remove data folder from remote repository"
git push
```



## 2. 使用.gitignore

- 将不想上传的路径写到.gitignore文件里面



## 3. 在本地删除了一个文件夹后，如何将远端也删除

```bash
git add -u
git commit -m "Remove folder"
git push
```

