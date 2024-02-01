---
title: git部署博客源码

date: 2024-1-30

tags: [博客]

categories: [博客,git]

comment: true

toc: true


---

#

 <!--more-->

- 在blog/config文件里面确定了hexo s将博客部署到仓库的main分支，但是部署的是渲染后的代码，如何保存源码进行备份。



在blog下打开gitbash

```
git init 
```

去掉blog/.gitignore文件中这三项：

```
db.json,
Thumbs.db
node-modules/,
```

运行

```
git add *
git commit -m "first commit"
```

创建一个source分支，上传源码（注意此处仓库地址，参考blog/config文件里面的部署部分，是带@符号的那个名字，不是简单的博客地址）

```
git checkout -b source
git remote add github 远程仓库地址
git push -u github source
```

建议将source分支设置成默认分支，防止误删

![](D:\blog\themes\yilia\source\img\hexo\config\1.png)

![](img/hexo/config/1.png)

![](D:\blog\themes\yilia\source\img\hexo\config\2.png)

![](img/hexo/config/2.png)