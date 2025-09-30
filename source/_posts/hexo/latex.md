---
title: hexo博客关于latex公式

date: 2024-1-30

tags: [博客]

categories: [博客,hexo]

comment: true

toc: true



---

#

<!--more-->

- 在yilia主题文件夹下的config文件中设置

  `mathjax: true`

- 一些网页无法渲染的公式

  在\前_后^后都放个空格

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 网页上无法渲染：$$ softmax(\mathbf{X})_{ij} = \frac{ \exp(\mathbf{X}_{ij}) } { \sum_k \exp(\mathbf{X}_{ik}) } $$ | 在后面的下标ij的下划线前加上空格$$ softmax(\mathbf{X})_{ij} = \frac{ \exp(\mathbf{X} _{ij}) } { \sum_k \exp(\mathbf{X} _{ik}) } $$ |
| 用左括号并列的公式无法渲染：$h ^\prime =\left\{ \begin{matrix}0 &amp;p \\\frac{h}{1-p} &amp;1-p \\\end{matrix}\right.$ | 使用cases，并用html换行：<span style="display:block"> $h ^\prime =\begin{cases}0 &p \\\\ \frac{h}{1-p} &1-p \\\\ \end{cases}.$   </span> |

- 公式无法换行：

  - ```bash
    cd blog
    npm uninstall hexo-renderer-marked
    npm install hexo-renderer-marked@1.0.0
    ```

  - 编辑 blog/node_modules/marked/lib/marked.js 

    - 第539行

    - ```bash
      escape: /^\\([!"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~])/,
      改成
      escape: /^\\([!"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~])/,
      ```

    - 第564行

    - ```bash
      inline._escapes = /\\([!"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~])/g;
      改成
      inline._escapes = /\\([!"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~])/g;
      ```

  - 

