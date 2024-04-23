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

# git部署博客源码

- 在blog/config文件里面确定了hexo s将博客部署到仓库的main分支，但是部署的是渲染后的代码，如何保存源码进行备份。



## 1 创建分支

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



## 2 设置默认分支

建议将source分支设置成默认分支，防止误删

![](D:\blog\themes\yilia\source\img\hexo\config\1.png)

![](img/hexo/config/1.png)

![](D:\blog\themes\yilia\source\img\hexo\config\2.png)

![](img/hexo/config/2.png)



- 以后只需要执行：

  ```bash
  git add .
  git commit -m "Your commit message"
  git push -u github source
  
  ```

## 3 在新设备上编写博客
1. 在设备上生成ssh密钥

    ```bash
    ssh-keygen
    ```

- 然后一路回车，会在:user/.ssh下面有id_rsa和id_rsa.pub。

2. 将id_rsa.pub内容复制到github中：
- 在github点击右上角自己的图像
- settings
- 左侧SSH and GPG keys
- 右侧：New SSH key

3. 在设备上提供github用户名和邮箱
- 用户名可以在右上角头像获得
- 又像可以通过setting -> email获得
  ```bash
  git config --global user.name "Your Name"
  git config --global user.email "you@example.com"
  ```
