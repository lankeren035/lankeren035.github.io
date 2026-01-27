## 1. 进程信息

- 查看父进程与完整命令

```
ps -ef | grep 387411
```

- 查看启动时间

	```
	ps -p 1234 -o lstart
	```

	



## 2. cuda相关

- 查看可用cuda

	```
	ls -l /usr/local | grep cuda
	```

	



## 3. 目录

- 递归查找

	```
	find . -type d -name "aaa"
	```


- 重命名

	```
	mv old_dir new_dir
	```

	

## 4. vim

- 启用鼠标

	```
	set mouse=a
	```

	

## 5. vsode

- vscode无法启动

	```
	code --disable-extensions
	```

	





## 视频处理

- 视频转gif

  ```
  ffmpeg -i input.mp4 -vf "fps=10" -loop 0 output.gif
  ```

- 视频大小

	```
	ffprobe 你的视频文件.mp4
	```

	

- 视频resize （宽高）

  ```
  ffmpeg -i 输入视频.mp4 -vf scale=640:480 -c:v libx265 -crf 28 输出视频.mp4
  
  
  ffmpeg -i input.mp4 \
  -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2" \
  -c:v libx264 -crf 18 -preset veryslow \
  -c:a copy \
  output.mp4
  
  ```

  