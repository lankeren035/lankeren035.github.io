---

title: 4- Git项目

date: 2024-8-23 12:00:00

tags: [Git]

categories: [Git]

comment: true

toc: true

---

####

<!--more-->



# 4- Git项目

## 4.1 初始化项目

- 创建

```bash
git init
```

- 克隆
```bash
git clone 连接
```



## 4.2 文件操作

### 4.2.1 文件四种状态

- Untracked：未跟踪，此文件在文件夹中，但并没有加入到git仓库，不参与版本控制，通过`git add`状态变为`staged`

- Unmodify：文件已经入库，未修改，即版本库中的文件快照内容与文件夹中完全一致，这种类型的文件有两种去处：

  - 如果他被修改，则变为`modified`
  - 如果使用`git rm`一处版本库，则成为`Untracked`文件

- Modified：文件已修改，仅仅是修改，并没有进行其他操作，这个文件也有两个去处：

  - 通过`git add`可以进入`staged`状态
  - 使用`git checkout`则丢弃修改，返回到`unmodify`状态，这个`git checkout`即从库中取出文件，覆盖当前修改。

- Staged：暂存状态，执行`git commit`则将修改同步到库中，这是库中的文件和本地文件又变为一致，文件为`Unmodify`状态。执行`git reset HEAD filename`取消暂存，文件状态为`Modified`

  

### 4.2.2 操作命令

- 查看文件状态

  ```bash
  git status
  ```

- 忽略文件：在`.gitignore`配置（支持Linux通配符）

  ```bash
  #为注释
  *.txt #忽略所有txt文件
  !lib.txt #但lib.txt除外
  /temp #仅忽略根目录下的TODO文件，不包括其他目录temp
  build/ #忽略build文件夹
  doc/*.txt #忽略doc/notes.txt但不包括doc/server/arch.txt
  ```

  