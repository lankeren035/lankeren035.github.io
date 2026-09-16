---
title: excalidraw使用
date: 2026-04-04 11:00:00
tags:
  - 经验
  - Obsidian
published: true
hexo-path: /2026/04/04/experience/obsidian/excalidraw
---
#
<!--more-->

# Obsidian Excalidraw 插件使用教程：常用功能、典型工作流与常用设置

## 1. 这是什么，适合拿来做什么

Excalidraw 是 Obsidian 里的一个画图插件。  
它不是拿来替代 markdown 笔记的，而是拿来补充下面这些场景：

- 画局部流程图
- 画函数调用细图
- 画模块结构草图
- 画参数组织图
- 在图里放图片、PDF、已有笔记
- 给图中的元素加链接，连回你的 Obsidian 笔记

对源码学习来说，建议这样分工：

- **Canvas**：项目总图、总导航、大结构
- **Excalidraw**：局部流程、函数内部逻辑、数据流草图
- **Markdown**：详细解释正文

所以如果你在学 3DGS：

- 用 Canvas 画 `train.py -> Scene -> GaussianModel -> render() -> rasterizer` 这种总图
- 用 Excalidraw 画 `training()` 或 `render()` 的详细流程
- 用 markdown 写每个模块和函数的说明

---

## 2. 最常用的创建方式

## 2.1 新建一张 Excalidraw 图

最稳的方法不是右键菜单，而是命令面板。

操作：

1. 按 `Ctrl + P`
2. 输入 `Excalidraw`
3. 选择下面任意一个命令：

- `Excalidraw: Create new drawing - IN A NEW TAB`
- `Excalidraw: Create new drawing - IN THE CURRENT ACTIVE WINDOW`
- `Excalidraw: Create new drawing - IN AN ADJACENT WINDOW`

建议新手直接用：

```text
Excalidraw: Create new drawing - IN A NEW TAB
```

执行后，会新建并打开一张 Excalidraw 图。

---

## 2.2 文件会保存成什么

Excalidraw 图会保存在你的 vault 里。  
你会看到类似下面这种文件名：

```text
Drawing 2026-04-03 11.11.31.excalidraw
```

或者某些配置下会是带 markdown 存储格式的文件。

建议你手动改成有意义的名字，例如：

```text
training_flow.excalidraw
render_pipeline.excalidraw
gaussian_model_structure.excalidraw
```

不要一直保留自动生成的时间戳名字，不然后面很难找。

---

## 3. 打开之后的界面怎么理解

打开 Excalidraw 之后，你看到的是一个白板界面。最常用的部分有三块：

### 3.1 画布中间
这是你真正画图的区域。

### 3.2 顶部工具栏
通常会有这些最常用工具：

- 选择工具
- 矩形
- 菱形
- 圆形
- 箭头
- 直线
- 自由绘制
- 文本
- 橡皮擦
- 图片

### 3.3 右侧或浮动操作区
会有一些附加操作，比如：

- 放大缩小
- 适应画布
- 插入文件
- 更多菜单

新手阶段最常用的，其实只有下面几个：

- 选择
- 矩形
- 箭头
- 文本
- 图片
- 橡皮擦

---

## 4. 最基础的使用方法

## 4.1 画流程框

最常见的就是矩形框。

操作：

1. 选择矩形工具
2. 在画布上拖出一个框
3. 双击框，输入文字

例如画 `training()` 流程时，可以先画几个框：

```text
取相机
render()
计算 loss
backward
optimizer.step()
densify / prune
保存与日志
```

---

## 4.2 连接箭头

流程图最常用的是箭头。

操作：

1. 选择箭头工具
2. 从一个框拖到另一个框
3. 如果需要，双击箭头中间输入说明

例如：

```text
取相机 -> render()
render() -> 计算 loss
计算 loss -> backward
```

箭头适合表达：

- 执行顺序
- 数据流向
- 调用关系
- 依赖关系

---

## 4.3 输入文本

如果你不想先画框，只想直接写字：

1. 选择文本工具
2. 在空白处点击
3. 输入内容

适合写：

- 简短备注
- 标题
- 解释说明
- 分组标签

例如：

```text
Python 端
CUDA 端
渲染主链
densify 分支
```

---

## 4.4 移动、缩放、删除

这些是最基本的操作：

- 点击元素：选中
- 拖动：移动
- 拖角点：缩放
- `Delete`：删除
- `Ctrl + Z`：撤销
- `Ctrl + Shift + Z` 或 `Ctrl + Y`：重做

---

## 5. 你最常用的 3 种用法

## 5.1 用法一：画函数内部流程图

这是你最常用的场景。

例如画 `training()`：

```text
training()
-> 取 viewpoint stack
-> render()
-> 计算 loss
-> backward
-> step
-> densify / prune
-> 保存
```

适合的问题：

- 这个函数到底分几步？
- 顺序是什么？
- 哪几步会修改状态？
- 哪几步只是计算中间量？

---

## 5.2 用法二：画模块结构图

例如画 `GaussianModel`：

```text
GaussianModel
├ xyz
├ scaling
├ rotation
├ opacity
├ features_dc
└ features_rest
```

适合的问题：

- 这个类里维护了哪些核心状态？
- 哪些参数参与优化？
- 哪些是渲染输入？
- 哪些是 densify / prune 会改的？

---

## 5.3 用法三：画边界图

例如画 Python 到 CUDA 的边界：

```text
render()
-> GaussianRasterizer
-> diff-gaussian-rasterization
```

适合的问题：

- 哪一步还是 Python
- 哪一步已经进扩展
- 哪一步开始调用底层 rasterizer
- 哪些量是从 Python 传给 C++/CUDA 的

---

## 6. 适合你当前源码学习的推荐工作流

## 6.1 总体原则

不要让 Excalidraw 取代 markdown。  
它最适合做“图”，不适合承载全部解释。

推荐分工：

- **Canvas**：画整个项目的大图
- **Excalidraw**：画某个模块或函数的细图
- **Markdown**：写这张图对应的详细说明

---

## 6.2 一个实际例子

### 第一步：Canvas 里放总图
你先在 Canvas 里画：

```text
train.py -> Scene -> GaussianModel -> render() -> diff-gaussian-rasterization
```

### 第二步：Excalidraw 里画局部细图
再单独建一张 `training_flow.excalidraw`，画：

```text
training()
-> 取相机
-> render()
-> loss
-> backward
-> step
-> densify/prune
-> 保存
```

### 第三步：Markdown 写详细说明
在 `03_Functions/training().md` 里写：

- 输入是什么
- 输出是什么
- 每一步干什么
- 调了谁
- 被谁调

这样你就形成了：

- 总图
- 局部流程图
- 文字说明

这三层结构。

---

## 7. 你最可能用到的常用功能

## 7.1 新建图并嵌入当前笔记

命令面板里有一类很实用的命令：

```text
Excalidraw: Create new drawing - ... - and embed into active document
```

这个命令会：

1. 新建一张 Excalidraw 图
2. 直接把它嵌入当前打开的 markdown 笔记

适合场景：

- 你在写 `training().md`
- 想顺手插入一张 `training_flow.excalidraw`
- 不想后面再手工链接

如果你经常是“边写笔记边插图”，这个命令很好用。

---

## 7.2 插入图片

你可以把图片放进 Excalidraw。

常见方式：

- 拖图片到画布
- 使用工具栏里的图片按钮
- 命令面板插入图片或文件

适合场景：

- 插入代码截图
- 插入网络结构图
- 插入论文里的图
- 插入训练日志截图

例如你可以把：

- `render()` 的代码截图
- `gaussian_model.py` 里的参数定义截图
- 训练过程中的 loss 曲线截图

都拖进来，再在旁边做解释。

---

## 7.3 插入 vault 中已有文件

命令面板里有这些很实用的功能：

- `Excalidraw: Insert image or Excalidraw drawing from your vault`
- `Excalidraw: Insert markdown file from vault`
- `Excalidraw: Insert ANY file`

这意味着你可以把：

- 一张已有图片
- 一张已有 Excalidraw 图
- 一个 markdown 文件
- 其他文件

放进当前图中。

适合场景：

- 在 `render_pipeline.excalidraw` 里插入 `render().md`
- 在流程图里放一张已有的代码截图
- 在大图里嵌入另一个局部 Excalidraw 图

---

## 7.4 在图里嵌入 markdown 笔记

这是一个很有用但很多人一开始没意识到的功能。

你可以把某篇 markdown note 以嵌入的形式放进 Excalidraw 图中。  
这样图里就不只是框和箭头，还能直接出现笔记内容的可视化块。

适合场景：

- 在 `training_flow.excalidraw` 里直接放 `training().md`
- 在 `gaussian_model_structure.excalidraw` 里直接放 `GaussianModel.md`

这样你的一张图就可以同时包含：

- 流程框
- 箭头
- 局部说明文字
- 已有 markdown 内容

---

## 7.5 给元素加链接

Excalidraw 里可以把元素和 Obsidian 笔记链接起来。

适合场景：

- 一个框代表 `render()`
- 点击它就回到 `render().md`
- 一个框代表 `GaussianModel`
- 点击它就跳到 `GaussianModel.md`

对源码学习非常有用，因为它能把“图”和“文档”连起来。

建议用法：

- 总图中的每个核心框，都尽量对应一个 note
- 局部图中的关键节点，也尽量连到对应函数说明页

---

## 7.6 导出图片

命令面板里有：

```text
Excalidraw: Export Image
```

适合场景：

- 你想把当前图导出成 PNG/SVG
- 想放到论文笔记里
- 想发给别人
- 想插入普通 markdown 文档

这个功能很常用。

---

## 7.7 图片裁剪和标注

命令面板里常见还有：

- `Excalidraw: Annotate image in Excalidraw`
- `Excalidraw: Crop and mask image`

适合场景：

- 对代码截图做标注
- 截一张论文图只保留局部
- 给某一块图加箭头解释

如果你经常想“截图 + 箭头 + 注释”，这两个功能很实用。

---

## 7.8 插入 LaTeX 公式

如果你要画算法或论文相关内容，LaTeX 很有用。

命令面板里可以插入公式。

适合场景：

- 画 loss 结构
- 画符号关系
- 在 3DGS 流程图里写数学表达式

例如可以放：

```text
L = L_render + λ L_reg
```

这种公式做局部说明。

---

## 7.9 插入 PDF 页

如果你在读论文，会很有用。

常见相关命令有：

- `Insert active PDF page as image`
- `Insert last active PDF page as image`

适合场景：

- 正在读 3DGS 论文
- 想把某一页关键图直接拖进 Excalidraw
- 在图上做自己的注释

---

## 8. 最常用的命令面板命令整理

下面这些命令你最可能会用到：

```text
Excalidraw: Create new drawing - IN A NEW TAB
Excalidraw: Create new drawing - IN THE CURRENT ACTIVE WINDOW
Excalidraw: Create new drawing - IN AN ADJACENT WINDOW
Excalidraw: Create new drawing - ... - and embed into active document
Excalidraw: Insert image or Excalidraw drawing from your vault
Excalidraw: Insert markdown file from vault
Excalidraw: Insert ANY file
Excalidraw: Export Image
Excalidraw: Annotate image in Excalidraw
Excalidraw: Crop and mask image
Excalidraw: Insert LaTeX formula
Excalidraw: Embed a drawing
```

新手阶段你先记住前 5 个就够了。

---

## 9. 常用设置：先调哪些

Excalidraw 设置很多，但你现在不需要全看。  
先看下面这些。

---

## 9.1 本地字体设置

如果你不喜欢默认字体，或者想让中文显示风格更统一，可以设置本地字体。

设置路径里要关注：

- `Fonts`
- `Enable local font option`
- `Local font file`

### 操作思路

1. 先把你要用的字体文件放进 vault（项目文件夹内）
2. 字体文件格式可以是：
   - `.otf`
   - `.ttf`
   - `.woff`
   - `.woff2`
3. 在 Excalidraw 设置里启用本地字体选项
4. 选择这个字体文件
5. 之后在文本元素的字体选项里使用它，选中文本，下面的编辑选项有字体，可以选择

### 建议

如果你是为了中文更稳定，建议选你自己常用的一款中文字体。  
如果只是为了导出体积更小，优先考虑 `.woff2`。

---

## 9.2 嵌入 markdown 时的默认字体

注意，这和“图中普通文本元素的字体”不是一回事。

当你把 markdown 文件嵌入 Excalidraw 时，插件有一套专门的嵌入字体设置。  
可以设置：

- 默认字体
- 字体颜色
- 边框颜色
- CSS 文件

这部分主要影响“嵌入的 markdown 外观”。

如果你当前主要只是画流程图，先不用深调。  
如果你后面频繁把 markdown note 嵌进图里，再回来调这一组设置。

---

## 9.3 自定义笔

设置里有 `Custom pens`。

作用：

- 在画布侧边或菜单附近放自己常用的笔
- 不用每次重新挑线宽、颜色、透明度

适合场景：

- 你习惯一种固定高亮笔
- 习惯一种固定箭头颜色
- 经常做截图标注

如果你大量做代码截图批注，这个会很省事。

---

## 9.4 自动保存

一般建议保持自动保存开启。  
除非你非常清楚自己在做什么，否则不要随便关掉 autosave。

因为你做的是知识库和流程图，不是一次性草稿，自动保存更安全。

---

## 9.5 导出相关设置

如果你经常导出图片，后面可以关注这些设置：

- 是否透明背景
- 深色/浅色导出
- 导出 padding
- PNG scale

如果只是日常在 Obsidian 内部用，暂时不用细调。  
只有在你准备把图导出给别人、或者插到文档里时，再来调。

---

## 10. 怎么设置字体：最直接的做法

## 10.1 图中普通文本元素字体

如果你要改“画布里文本框”的字体：

1. 打开 Excalidraw 设置
2. 找到 `Fonts`
3. 打开 `Enable local font option`
4. 在 `Local font file` 中选择你放进 vault 的字体文件
5. 回到画布
6. 选中文本元素
7. 在文本属性里切换到这个本地字体

建议先准备一个字体文件，例如：

```text
04_Images/fonts/你的字体.ttf
```

或者：

```text
04_Images/fonts/你的字体.woff2
```

然后再在插件设置里选它。

---

## 10.2 嵌入 markdown 的字体

如果你说的是“嵌入 note 后显示出来的字”，那要改的是嵌入 markdown 的字体设置，不是普通文本工具本身。

这部分是两套不同设置，不要混。

---

## 11. 你现在最值得先掌握的 5 个功能

如果你不想一下子看太多，就先只用下面 5 个：

### 功能 1：新建图
用命令面板创建一张新图。

### 功能 2：矩形 + 箭头 + 文本
先把最基础的流程图画起来。

### 功能 3：插入图片
把代码截图或论文图拖进来标注。

### 功能 4：给框加链接
把图和 markdown 笔记连起来。

### 功能 5：导出图片
把完成的图导出成 PNG 或 SVG。

只掌握这 5 个，就已经足够服务你的 3DGS 学习记录了。

---

## 12. 一个适合 3DGS 的起步练习

建议你现在就做下面两张图。

## 12.1 第一张：`training_flow.excalidraw`

内容：

```text
training()
-> 取 viewpoint camera
-> render()
-> 计算 loss
-> backward
-> optimizer.step()
-> densify / prune
-> 保存与日志
```

目的：

- 练习矩形、箭头、文本
- 形成第一个局部流程图

---

## 12.2 第二张：`render_pipeline.excalidraw`

内容：

```text
输入 viewpoint_camera + GaussianModel
-> 组装 rasterizer settings
-> 调用 GaussianRasterizer
-> 输出 image / radii / visibility
```

目的：

- 练习画边界图
- 练习后续给元素加笔记链接

---

## 13. 对源码学习最实用的建议

### 13.1 不要一开始就追求画得很好看
先把结构画出来，比样式更重要。

### 13.2 不要让一张图塞太多东西
一张图只回答一个问题。

例如：

- `training_flow.excalidraw`：只回答 training 怎么跑
- `render_pipeline.excalidraw`：只回答 render 怎么跑
- `gaussian_model_structure.excalidraw`：只回答 GaussianModel 里有哪些核心参数

### 13.3 图一定要和 markdown note 配合
不要只留图，不写文字说明。  
否则过几天你自己都忘了当时为什么这么画。

---

## 14. 我建议你的最终工作流

### 总图
用 Canvas。

### 局部图
用 Excalidraw。

### 详细正文
用 markdown。

### 推荐顺序
1. 先在 Canvas 里搭总图
2. 再在 Excalidraw 里补局部图
3. 最后在 markdown note 里写说明

---

## 15. 结论

Excalidraw 不是拿来替代你的笔记，而是拿来让你的源码学习“可视化”。

对你现在的 3DGS 学习最实用的用法是：

- 画 `training()` 的流程图
- 画 `render()` 的处理链
- 画 `GaussianModel` 的参数结构
- 插入代码截图和论文图做标注
- 把图中的关键框链接回 markdown 笔记

先不要试图把所有高级功能都学完。  
你先把下面这条最小闭环跑通就够了：

```text
新建 Excalidraw 图
-> 画矩形和箭头
-> 写文本
-> 插入一张截图
-> 给一个节点加链接
-> 导出图片
```

这一步跑通后，这个插件就已经真正能帮到你了。