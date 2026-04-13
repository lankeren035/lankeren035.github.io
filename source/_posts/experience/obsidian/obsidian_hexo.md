---
title: obsidian笔记转为hexo博客
date: 2026-04-04
tags:
  - 经验
  - Obsidian
categories:
  - 经验
comment: true
toc: true
---
#
<!--more-->

# Obsidian 笔记部署为 Hexo 博客

## 一、目标

把 Obsidian 笔记发布到 Hexo 博客，并解决以下问题：

- 在 Obsidian 中把笔记转换为 Hexo 文章
- 在 Obsidian 中直接执行发布、生成、部署
- 用 Syncthing 同步多设备，但避免同步设备相关配置
- 在 Obsidian 中直接对 `blog` 仓库做 Git pull / push

---

## 二、使用到的插件

本文实际使用了这几个插件：

1. **Hexo Toolkit（不是必须）** ： 用来把 Obsidian 特有语法尽量转换成 Hexo 兼容格式
2. **Hexo Integration**
   - 用来把当前笔记转换成 Hexo 文章、发布到 `_posts`、执行 `hexo g / hexo d`
3. **Obsidian Git**
   - 用来在 Obsidian 中直接对 Git 仓库执行 pull / push / commit-and-sync
4. **BRAT**
   - 仅用于安装 `Hexo Integration`
5. **hexo插件：**
   - hexo-backlink 用于转换obsidian文档中的wiki链接
   
---

## 三、插件安装


## 1. 安装 Hexo Toolkit

在 Obsidian 中打开：

- `设置 -> 第三方插件 -> 浏览`

搜索：

- `Hexo Toolkit`

安装并启用。
## 2. 安装 BRAT

在 Obsidian 中打开：

- `设置 -> 第三方插件 -> 浏览`

搜索：

- `BRAT`

安装并启用。

---

## 3. 用 BRAT 安装 Hexo Integration

安装好 BRAT 后：

1. 打开命令面板
2. 执行：

```text
BRAT: Add a beta plugin for testing
````

3. 填入仓库：
    

```text
nanjo712/obsidian-hexo-integration
```

4. 安装完成后，到第三方插件列表中启用：
    

```text
Hexo Integration
```

### 说明

`Hexo Integration` 当时在插件市场里搜不到，所以需要通过 BRAT 安装。

如果 BRAT 提示 GitHub API rate limit exceeded，可以：

- 等待限流恢复后再装
    
- 或使用 GitHub Token
    
- 或手动安装 release 文件
    

---

## 4. 安装 Obsidian Git

在 Obsidian 中打开：

- `设置 -> 第三方插件 -> 浏览`
    

搜索：

- `Git`
安装并启用。


---
## 5. 安装hexo-backlink

```
npm install hexo-backlink
```
---
然后在hexo博客配置文件`_config.json`中设置：backlink: true ，该插件可以将obsidian中的链接转换成hexo形式的链接
## 四、Hexo Integration 配置

## 1. 设置博客根目录

打开：

- `设置 -> 第三方插件 -> Hexo Integration`
    

设置：

- `Root Directory` = 你的 Hexo 博客根目录绝对路径

例如：

```text
D:/blog
```

**注意**：如果你在多台设备同步一个目录，博客路径可能不同，此时需要忽略同步文件：[[Syncthing#14. `.obsidian` 文件夹要不要同步]]



## 2. 解决 publish 时 Baidu Translate 报错

实际使用时，`Hexo Integration` 有一个问题：

即使当前 permalink 方式不是 `Baidu Translate`，有时也会在 publish 时提示：

```text
Baidu app ID or API key not configured
```

### 实际可用的处理方法

先在 `Hexo Integration` 设置中：

1. 把 permalink 生成方式切到 `Baidu Translate`
    
2. 在 `Baidu App ID` 和 `API Key` 里随便填一点内容
    
3. 再把 permalink 生成方式切回你真正想用的方式，例如：
    
    - `Short hash`
        
    - `Pinyin`
        
    - `Note title`
        

这样之后就可以正常 publish。

---

## 3. 图片设置

如果希望发布后的图片使用普通 Markdown 语法，而不是 Hexo tag，则在 `Hexo Integration` 中设置：

- `Image syntax` = `Markdown`
    

同时在 Hexo 的 `_config.yml` 中建议设置：

```yaml
post_asset_folder: true
marked:
  prependRoot: true
  postAsset: true
```

这样文章资源目录中的图片可以用相对路径引用。
### 3.1 远程图片

- 你可以将图片放到远程github仓库，然后在md文档中引用：
	- 如`lankeren035` 用户的 `blog_assets` 远程仓库下的`main` 分支下的 `img/head.png` 图片：`https://github.com/lankeren035/blog_assets/blob/main/img/head.png`
	- 引用路径：`https://cdn.jsdelivr.net/gh/lankeren035/blog_assets@main/img/head.png`

---

## 五、Hexo Toolkit 的使用

如果笔记中使用了 Obsidian 特有语法，例如：

- `[[双链]]`
    
- `![[图片]]`
    
- 部分嵌入语法
    

那么在发布前先执行：

```text
ctrl + p
Hexo Toolkit: Convert
```

### 注意

如果是文件链接，例如：

```md
[[Syncthing]]
```

要想被正确转换成 Hexo 链接，被链接的目标笔记本身需要有hexo-path属性，具体路径随便写

```yaml
hexo-path: /posts/syncthing
```

否则不会正确转换。
转换后在右下角会显示hexo: success，你需要自己复制过来

---

## 六、Hexo Integration 的使用流程

## 1. 首次处理当前笔记

- 该步骤相当于保证每个blog有data, title等数据，其实只要你写了就不用该操作

先执行：

```text
Hexo Integration: Convert current file to Hexo format
```

这一步用于：

- 给当前笔记补充 Hexo 所需 front matter
    
- 让该笔记进入 Hexo Integration 的工作流
    

---

## 2. 发布当前文章

执行：

```text
Hexo Integration: Publish current post
```

### 说明

这一步只是把文章发布到 Hexo 源目录中，一般是：

```text
source/_posts
```

但是你可能不希望所有文件都直接放在_posts目录下，对此你可以将整个blog文件夹放到vault里面。这样就不需要发布当前文章了，直接hexo g就行


---

## 3. 生成 Hexo 页面

执行：

```text
Hexo Integration: Generate hexo pages
```

等价于：

```bash
hexo g
```

---

## 4. 部署博客

执行：

```text
Hexo Integration: Deploy hexo pages
```

等价于：

```bash
hexo d
```

---



## 八、用 Syncthing 同步时的 ignore 设置

## 1. 目标

使用 Syncthing 同步 Obsidian / Hexo 相关目录时，需要避免同步以下内容：

- `Hexo Integration` 的本地路径配置
    
- Hexo 生成产物
    
- 缓存文件
    
- deploy 目录
    
- 大体积本地依赖
    
- Git 元数据
    
- Obsidian 的设备级工作区配置
    
```
# Ignore Hexo Integration local settings
/obsidian笔记/.obsidian/plugins/hexo-integration/data.json
.obsidian/plugins/hexo-integration/data.json

# Hexo generated / cache / deploy artifacts
/blog/.deploy_git
/blog/node_modules
/blog/public
/blog/db.json
/blog/themes/yilia/source/img

# Git metadata - recommended to keep local per device
/blog/.git

# Obsidian device-specific files
/.obsidian/workspace.json
/.obsidian/workspaces.json
```

---

## 九、Git 插件配置

## 1. 目录结构

假设当前目录结构是：

```text
Vault/
  docs/
  blog/
  code/
```

其中：

- `blog/` 是 Git 仓库
    
- Obsidian 打开的是整个 `Vault`
    

---

## 2. Obsidian Git 设置

打开：

- `设置 -> 第三方插件 -> Git`
    

把：

```text
Custom base path
```

设置为：

```text
blog
```

### 含义

这表示 Git 插件只对：

```text
Vault/blog
```

这个子目录中的 Git 仓库执行操作。

---

## 3. 常用命令

在命令面板中可用：

```text
Git: Pull
Git: Push
Git: Commit-and-sync
```

其中：

```text
Git: Commit-and-sync
```

实际执行流程相当于：

```text
stage everything -> commit -> pull -> push
```

也就是：

```bash
git add .
git commit -m "..."
git pull
git push
```




- obsidian使用：[[Obsidian]]