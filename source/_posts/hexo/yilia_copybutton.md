---
title: yilia主题代码块添加复制按钮

date: 2024-8-23 14:00:00

tags: [博客]

categories: [博客]

comment: true

toc: true



---

#### 

<!--more-->





# yilia主题代码块添加复制按钮



1. 在`themes/yilia/source/js`中新建`clipboard_use.js`：

   ```js
   $(".highlight").wrap("<div class='code-wrapper' style='position:relative'></div>");
   /*页面载入完成后，创建复制按钮*/
   !function (e, t, a) {
       /* code */
       var initCopyCode = function () {
           var copyHtml = '';
           copyHtml += '<button class="btn-copy" data-clipboard-snippet="">';
           copyHtml += '  <i class="fa fa-clipboard"></i><span>复制</span>';
           copyHtml += '</button>';
           $(".highlight .code").before(copyHtml);
           var clipboard = new ClipboardJS('.btn-copy', {
               target: function (trigger) {
                   return trigger.nextElementSibling;
               }
           });
           clipboard.on('success', function (e) {
               e.trigger.innerHTML = "<i class='fa fa-clipboard'></i><span>复制成功</span>"
               setTimeout(function () {
                   e.trigger.innerHTML = "<i class='fa fa-clipboard'></i><span>复制</span>"
               }, 1000)
              
               e.clearSelection();
           });
           clipboard.on('error', function (e) {
               e.trigger.innerHTML = "<i class='fa fa-clipboard'></i><span>复制失败</span>"
               setTimeout(function () {
                   e.trigger.innerHTML = "<i class='fa fa-clipboard'></i><span>复制</span>"
               }, 1000)
               e.clearSelection();
           });
       }
       initCopyCode();
   }(window, document);
   
   ```

   

2. 在`themes/yilia/layout/layout.ejs`中的`</body>`前引入：

   ```ejs
   <!-- 代码块复制功能 -->
   <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/clipboard@2.0.4/dist/clipboard.js"></script>
   <script type="text/javascript" src="https://apps.bdimg.com/libs/jquery/2.1.4/jquery.min.js"></script>
   <script type="text/javascript" src="/js/clipboard_use.js"></script>
   
   ```

   

3. 在`themes\yilia\source\main.0cf68a.css `中搜索

   ```css
   pre .css~* .id,
pre .id {
     color: #fd971f
}
   ```

4. 在其下方添加：

   - 样式1：
   
       ```css
       .btn-copy {
           display: inline-block;
           cursor: pointer;
           background-color: #eee;
           background-image: linear-gradient(#fcfcfc, #eee);
           border: 1px solid #d5d5d5;
           border-radius: 3px;
           -webkit-user-select: none;
           -moz-user-select: none;
           -ms-user-select: none;
           user-select: none;
           -webkit-appearance: none;
           font-size: 13px;
           font-weight: 700;
           line-height: 20px;
           color: #333;
           -webkit-transition: opacity .3s ease-in-out;
           -o-transition: opacity .3s ease-in-out;
           transition: opacity .3s ease-in-out;
           padding: 2px 6px;
           position: absolute;
           right: 5px;
           top: 5px;
           opacity: 0;
       }

       .btn-copy span {
           margin-left: 5px
       }

       .code-wrapper:hover .btn-copy {
           opacity: 1;
       }

       ```
   
   - 样式2：
   
     ```css
     .btn-copy {
         display: inline-block;
       position: absolute;
         right: 1px;
         top: -25px;
         cursor: pointer;
         background-color: #515151;
         border: none;
         -webkit-user-select: none;
         -moz-user-select: none;
         -ms-user-select: none;
         user-select: none;
         -webkit-appearance: none;
         font-size: 13px;
         font-weight: 700;
         line-height: 20px;
         color: #d5d5d5;
         -webkit-transition: opacity .3s ease-in-out;
         -o-transition: opacity .3s ease-in-out;
         transition: opacity .3s ease-in-out;
         padding: 2px 6px;
     }
     
     .btn-copy span {
         margin-left: 5px
     }
     
     .code-wrapper .btn-copy:hover {
         color: #fff;
     }
     
     ```
     
     

