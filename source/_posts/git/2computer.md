---
title: "Git在两台电脑上上传一个项目"

date: 2025-1-23 13:00:00

tags: [Git]

categories: [Git]

comment: true

toc: true

---

####

<!--more-->



# 1. 初始拉取项目

1. 生成本地ssh密钥

   ```shell
   ssh-keygen #生成本地ssh密钥，一路回车
   ```

2. 一路回车， 会在:user/.ssh下面有id_rsa和id_rsa.pub 

3. 将id_rsa.pub内容复制到github中： 

   - 在github点击右上角自己的图像
   - settings
   - 左侧SSH and GPG keys
   - 右侧：New SSH key

4. 在本地提供 github用户名和邮箱 

   - 用户名可以在右上角头像获得
   -  邮箱可以通过setting -> email获得 

   ```shell
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"
   ```

5.  拉取项目

   ```shell
   git init 
   git checkout -b source
   git remote add github 远程仓库地址 #注意替换成你的仓库地址，以.git结尾
   # 如果add github 仓库地址这里地址填错了：
   # git remote remove github
   # git remote add github 远程仓库地址
   git pull github source
   ```

6.  上传项目

   ```shell
   git add .
   git commit -m "提示"
   git push --set-upstream github source
   ```

   

# 2. 后续上传项目

- 当别的设备上传他的文件之后，需要先拉取最新版本，然后上传自己的。

- 先拉取

  ```shell
  git fetch github
  git pull github source
  ```

  - 如果发现出错：

    ```shell
    error: Your local changes to the following files would be overwritten by merge:
            1111test.md
            db.json
            source/_posts/programs/stable_diffusion/1_setup.md
    Please commit your changes or stash them before you merge.
    ```

  - 说明本地这几个文件跟远端不一致。

    - 如果你需要保存本地更改：

      ```shell
      git add .
      git commit -m "保存本地更改"
      git pull github source
      ```

    - 如果你不需要本地更改，强制拉取远程更新：

      ```shell
      git reset --hard
      git pull github source
      ```

- 

  ```shell
  # 第一次上传用：git push -u github source
  ```

  