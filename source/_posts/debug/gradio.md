---
title: gradio无法访问temp

date: 2024-7-15

tags: [conda,debug]

categories: [debug]

comment: false

toc: true


---

#

<!--more-->

# gradio无法访问temp

- 问题：使用stable diffusion webforge时，将项目从用户1迁移到用户2，运行之后发现无法使用control net线稿。每次运行后预览窗口没图像，后端显示：`Permission denied: '/tmp/gradio/tmpzrcdfehy.png'`

- 原因：权限问题

- 解决：在webui.sh最前面加上环境变量，修改tmp默认路径

  ```bash
  export GRADIO_TEMP_DIR="home/用户名/tmp"
  ```

  