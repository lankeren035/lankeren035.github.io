---

title: 2- Git环境配置

date: 2024-8-21 14:00:00

tags: [Git]

categories: [Git]

comment: true

toc: true

---

#### 

<!--more-->

# 2- Git环境配置

## 2.1 安装基础

- Git bash: linux命令
- Git cmd: windows命令
- Git gui: 图形界面

## 2.2 基本Linux命令

| 命令    | 解释                |
| ------- | ------------------- |
| cd      | 改变目录            |
| pwd     | 显示当前目录        |
| ls(ll)  | 列出所有文件        |
| touch   | 新建文件            |
| rm      | 删除文件            |
| mkdir   | 新建目录            |
| mv      | 移动文件            |
| reset   | 重新初始化终端/清屏 |
| clear   | 清屏                |
| history | 查看命令历史        |
| help    | 帮助                |
| exit    | 推出                |
| #       | 注释                |



## 2.3 Git配置

- 查看Git配置：

  ```bash
  git config -l #所有配置
  git config --system --list #系统配置的：Git/etc/gitconfig
  git config --global --list #用户配置的: User/username/.gitconfig
  ```

- 自定义配置 (下载Git后需要配置用户名邮箱）：

  ```bash
  git config --global user.name "xxx"
  git config --global user.email 'xxx@xxx.com'
  ```

  

