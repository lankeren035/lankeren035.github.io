(需要修改的地方)
显示分类时修改source/category里的index:

title: categories
date: 2023-07-20 10:18:50
type: "categories"
layout: "categories"
comments: false


- 换头像
theme/yilia/config文件
看到：avatar: img/head.png
将head.png放到：blog/themes/yilia/source/img
将blog/config文件中post_asset_folder: true





- 代码块自动换行
```
pre {
    overflow: auto;
    white-space: pre;
    /*white-space: pre-wrap;*/
    word-wrap: break-word
}
```
- 代码块滚动条颜色：

```
::-webkit-scrollbar-thumb { /*滚动条的滑块*/

  border-radius: 8px;

  background-color: rgb(189, 189, 189)

}```
```