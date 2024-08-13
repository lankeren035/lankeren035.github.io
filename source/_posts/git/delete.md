\---

title: git中断后删除文件夹重传

date: 2024-8-12 13:00:00

tags: [Git]

categories: [Git]

comment: true

toc: true

\---



**####** 



<!--more-->



# git中断后删除文件夹重传



- 使用git上传代码, 上传到一半发现传的东西有点多, 有个数据集文件夹有300M, 于是git push执行到一半ctrl + c中断了, 然后手动将项目中的data文件夹删除了, 发现重新上传的时候data文件夹还在缓存里, 并且会出现:

  ```bash
  $ git push
  Enumerating objects: 108, done.
  Counting objects: 100% (108/108), done.
  Delta compression using up to 6 threads
  Compressing objects: 100% (86/86), done.
  Writing objects: 100% (96/96), 327.79 MiB | 1.82 MiB/s, done.
  Total 96 (delta 36), reused 0 (delta 0), pack-reused 0 (from 0)
  remote: Resolving deltas: 100% (36/36), completed with 7 local objects.
  remote: error: Trace: 57708200081f8cc52dc33877e84bd73e6731a99f1c6fa4096b45c55a1b24b2d7
  remote: error: See https://gh.io/lfs for more information.
  remote: error: File source/_posts/deeplearning/code/pytorch/12_computer_vision/data/cifar-10-python.tar.gz is 162.60 MB; this exceeds GitHub's file size limit of 100.00 MB
  remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
  To github.com:lankeren035/lankeren035.github.io.git
   ! [remote rejected] source -> source (pre-receive hook declined)
  error: failed to push some refs to 'github.com:lankeren035/lankeren035.github.io.git'
  ```

- 解决:

  ```bash
  git rm -r --cached source/_posts/deeplearning/code/pytorch/12_computer_vision/data
  rm -r source/_posts/deeplearning/code/pytorch/12_computer_vision/data
  git commit -m "Removed data folder from repository"
  git add .
  git checkout -- .
  git filter-branch --force --index-filter \
    'git rm -r --cached --ignore-unmatch source/_posts/deeplearning/code/pytorch/12_computer_vision/data' \
    --prune-empty --tag-name-filter cat -- --all
  git push
  ```

  