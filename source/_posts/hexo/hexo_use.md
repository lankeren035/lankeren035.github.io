---
title: hexo博客使用

date: 2022-12-12

tags: [博客]

categories: [博客]

comment: true

toc: true



---

#  

 <!--more-->



# hexo博客使用

## 1. 站内文章跳转

- 推荐使用标签：假设你想跳转`Hello World`博客（是博客的title，不是文件名）

  ```
  {%post_link 'Hello World'%}
  ```

  {%post_link 'Hello World'%}

- 使用相对路径（保证你跳转的博客date跟当前博客一致）

  ```
  [hexo搭建](../hexo搭建)
  ```

  [hexo搭建](../hexo搭建)

- 绝对路径

  ```
  [Hello world](https://lankeren035.github.io/2014/12/24/hexo/hello-world/)
  ```

  [Hello world](https://lankeren035.github.io/2014/12/24/hexo/hello-world/)


