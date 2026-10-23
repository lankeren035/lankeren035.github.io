---
title: title
date: 2023-12-12
tags: [其他]
categories: [其他]
comment: true
toc: true
published: true
permalink: experience/test
hexo-path: 
---
#
<!--more-->
# Obsidian 每日学习日志完整流程

## 一、整体思路

用 Obsidian 自带的「每日笔记」插件一天生成一篇日志，模板按小时划分时段，每个时段包含日常记录和学习记录两部分，学习记录用内联字段标注主题、知识点，用可勾选任务标注掌握程度。再用 Dataview 插件从所有日志中提取数据，生成待复习清单、知识点汇总、按日期浏览等查询页面，实现记录与复习的闭环。

## 二、插件安装与设置

需要两个插件。「每日笔记」是 Obsidian 核心插件，在设置→核心插件中开启即可，它负责一键创建当天日志、自动命名、套用模板。进入其选项页，日期格式设为 `YYYY-MM-DD 时间日志`，新文件位置选 `record/日志`，模板文件指向你存放模板的路径。如果希望文件自动按年月归档到子文件夹（如 `record/日志/2026/10/`），核心插件做不到，需改用第三方插件 Periodic Notes，在其 Daily Notes 设置里把 Folder 填为 `record/日志/{{date:YYYY}}/{{date:MM}}` 即可，其余配置相同。

Dataview 是第三方插件，在社区插件市场搜索安装并启用。进入设置，开启 Enable Queries 和 Enable JavaScript Queries 两项，前者支持普通表格 / 列表查询，后者支持用 JS 代码做更复杂的按块提取。Dataview 的 `FROM "record/日志"` 会递归包含该文件夹下所有子文件夹，所以无论日志是否按年月分目录，查询都不用改路径。

## 三、模板设计

在库中新建 `Templates` 文件夹，创建 `每日时段日志.md` 作为模板。模板顶部是 YAML 元数据，用 `---` 包裹，包含 date、weekday 和 tags 三个字段，日期用 Daily Notes 支持的变量 `{{date:YYYY-MM-DD}}`，注意不要用 Templater 的 `<% tp.date.now() %>` 语法，每日笔记插件不会执行它。正文从 07:00 到 23:00，每个小时一个二级标题，下面分两个三级标题：「🌱 日常」写当时的生活关键词或一句话，「📚 学习」用项目符号列出 `主题::`、`知识点::` 两个内联字段，以及掌握程度的三个可勾选任务（了解、熟悉、掌握），任务下方留空行写大段学习内容。每个小时块结尾用 `---` 分割。模板底部是今日总复盘区，包含今日一句话、学到了什么、踩坑、待复习任务列表、明天计划、心情状态。内联字段 `主题::` `知识点::` 是 Dataview 能识别的关键，写在项目符号里即可，不需要表格。掌握程度用 `- [ ]` 原生任务框，鼠标直接点击勾选，三选一只勾一个。

## 四、日常使用

每天打开 Obsidian，按 `Ctrl+P` 调出命令面板，执行「每日笔记：打开今日笔记」（Periodic Notes 则是 Open daily note），插件会自动在指定文件夹创建当天文件、套用模板、自动命名。在对应时段的「日常」里写生活关键词，在「学习」里填主题和知识点，勾选掌握程度，下方写几百字的学习总结、踩坑记录、理解推导。一天结束后在底部复盘区用自己的话复述当天学到的内容，把需要回顾的点写进「待复习」的任务框里。整个写作过程就是普通 Markdown，编辑器区域不受任何卡片限制，可随意粘贴图片、代码块、长段落。

## 五、查询汇总

新建一篇 `汇总.md` 放在 `record/日志` 下，把以下查询分别放入代码块，代码块语言标记必须写 `dataviewjs`（不能只写 dataview，否则 JS 代码会报解析错误），切换到阅读视图自动渲染。

所有日记按年月日排列的总览，读取每篇顶部的「今日一句话」作为摘要，日期列可点击跳转：

```
const rows = [];
for (const page of dv.pages('"record/日志"')) {
  const content = await dv.io.load(page.file.path);
  if (!content) continue;
  let summary = "";
  const m1 = content.match(/今日一句话：[ \t]*([^\n]+)/);
  if (m1 && m1[1].trim()) summary = m1[1].trim();
  else {
    const m2 = content.match(/今日一句话：\s*\n\s*([^\n>]+)/);
    if (m2) summary = m2[1].trim();
  }
  const dm = page.file.name.match(/(\d{4})-(\d{2})-(\d{2})/);
  if (!dm) continue;
  rows.push({ year: dm[1], month: dm[2], date: dm[0], link: page.file.link, summary: summary || "（未写）" });
}
rows.sort((a, b) => b.date.localeCompare(a.date));
dv.table(["年", "月", "日期", "今日一句话"], rows.map(r => [r.year, r.month, r.link, r.summary]));
```

所有待复习知识点，提取每篇底部「🔁 待复习」下未勾选的任务：

```
const rows = [];
for (const page of dv.pages('"record/日志"')) {
  const content = await dv.io.load(page.file.path);
  if (!content) continue;
  const sec = content.match(/##\s*🔁\s*待复习([\s\S]*?)(?=\n##\s|\n#\s|$)/);
  if (!sec) continue;
  for (const m of sec[1].matchAll(/-\s*\[\s\]\s*(.+)/g)) {
    const text = m[1].trim();
    if (text) rows.push([page.file.link, text]);
  }
}
rows.sort((a, b) => b[0].path.localeCompare(a[0].path));
if (rows.length === 0) dv.paragraph("✅ 没有待复习项");
else dv.table(["日期", "待复习内容"], rows);
```

所有「了解」级别的知识点，按小时块分割文本，检测学习区是否勾选了了解，输出日期、时段（可跳转到文件内对应标题）、知识点三列：

```
const rows = [];
for (const page of dv.pages('"record/日志"')) {
  const content = await dv.io.load(page.file.path);
  if (!content) continue;
  const sections = content.split(/^## (\d{2}:\d{2})\s*$/m);
  for (let i = 1; i < sections.length; i += 2) {
    const hour = sections[i];
    const body = sections[i + 1] || "";
    if (!body.includes("### 📚 学习")) continue;
    let kp = body.match(/知识点::\s*([^\n]+)/);
    if (!kp) kp = body.match(/主题::\s*([^\n]+)/);
    const knowledge = kp ? kp[1].trim() : "（未填知识点）";
    if (/-\s*\[[xX]\]\s*了解/.test(body)) {
      rows.push({ date: page.file.name, dateLink: page.file.link, hour, hourLink: dv.fileLink(page.file.path + "#" + hour, false, hour), knowledge });
    }
  }
}
rows.sort((a, b) => a.date !== b.date ? b.date.localeCompare(a.date) : a.hour.localeCompare(b.hour));
dv.table(["日期", "时段", "知识点"], rows.map(r => [r.dateLink, r.hourLink, r.knowledge]));
```

昨天的知识点，分待复习和了解两个查询，核心是用 `new Date()` 计算昨天日期字符串，再用 `page.file.name.includes()` 筛选文件，其余提取逻辑与上面相同。最近五天则是生成最近五天的日期数组循环匹配，最后把待复习和了解分两个表格输出。这两个查询的完整代码较长，核心模式不变，需要时把日期筛选条件替换进去即可。

## 六、注意事项

模板顶部的 YAML 必须从文件第一行开始，前面不能有空行或脚本，否则 Obsidian 不识别为元数据。标签在 YAML 里用列表写法且不带 `#` 号。Daily Notes 只认 `{{date}}` 变量，不要混入 Templater 语法。DataviewJS 读取文件内容用 `await dv.io.load(page.file.path)`，不要用 `page.file.content`，后者在多数版本里为 undefined 会报错。时段超链接通过 `dv.fileLink(path + "#" + hour)` 构造，可直接跳转到文件内对应小时标题。掌握程度的勾选状态通过正则 `/-\s*\[[xX]\]\s*了解/` 检测，换成熟悉或掌握只需改关键词。