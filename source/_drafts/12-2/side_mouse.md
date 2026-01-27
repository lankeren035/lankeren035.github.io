



## ubuntu中使用鼠标侧键

- 先查看侧键名字：

	```shell
	xev | grep -A2 --line-buffered 'ButtonPress' | grep 'button'
	```

	- 出现图形界面后点击界面，然后点击侧键可以看到侧键的名字（一般是8、9两个键）

- 设置：

	- 9：短按ctrl+c；长按ctrl+v
	- 8：短按Enter；长按Super+D

```shell
sudo apt install xbindkeys xdotool
vim ~/.mouse_side_key_handler.sh
```

- ```
	#!/bin/bash
	
	KEY_NUM=$1
	if [ -z "$KEY_NUM" ]; then echo "请指定侧键编号（8或9）"; exit 1; fi
	
	# 核心函数：识别当前活跃应用类型（终端/VS Code/浏览器/其他）
	
	get_app_type() {
	    window_id=$(xdotool getactivewindow 2>/dev/null)
	    if [ -z "$window_id" ]; then
	        echo "other"
	        return
	    fi
	
	    # 用xprop获取窗口类名（已适配你的终端）
	    wm_class=$(xprop -id "$window_id" WM_CLASS 2>/dev/null | awk -F '", "' '{print $2}' | sed 's/"//')
	
	    # 终端识别：明确添加你的终端类名 Gnome-terminal
	    if [[ $wm_class == "Gnome-terminal" ||  # 你的终端类名，必须添加
	          $wm_class =~ "Konsole" || 
	          $wm_class =~ "Alacritty" || 
	          $wm_class =~ "Terminal" ]]; then
	        echo "terminal"
	        return
	    fi
	
	    # VS Code识别（不变）
	    if [[ $wm_class == "Code" ]]; then
	        echo "vscode"
	        return
	    fi
	
	    # 浏览器识别（不变）
	    if [[ $wm_class =~ "Chrome" || $wm_class =~ "Firefox" || $wm_class =~ "Chromium" || $wm_class =~ "Edge" ]]; then
	        echo "browser"
	        return
	    fi
	
	    echo "other"
	}
	
	# 配置参数（按侧键编号和应用类型动态设置）
	case $KEY_NUM in
	    8)
	        THRESHOLD=500  # 长按阈值（ms）
	        SHORT_ACTION="xdotool key Return"       # 短按：回车（不变）
	        LONG_ACTION="xdotool key Super+d"       # 长按：显示桌面（不变）
	        ;;
	    9)
	        THRESHOLD=300  # 长按阈值（ms）
	        app_type=$(get_app_type)
	        # 根据应用类型设置复制（短按）和粘贴（长按）动作
	        case $app_type in
	            "terminal")
	                # 终端：用Ctrl+Shift+C/V
	                SHORT_ACTION="xdotool key ctrl+shift+c"
	                LONG_ACTION="xdotool key ctrl+shift+v"
	                ;;
	            "vscode")
	                # VS Code：编辑器和终端统一用Ctrl+C/V（后续需修改VS Code快捷键配合）
	                SHORT_ACTION="xdotool key ctrl+c"
	                LONG_ACTION="xdotool key ctrl+v"
	                ;;
	            "browser"|"other")
	                # 浏览器/其他应用：用标准Ctrl+C/V
	                SHORT_ACTION="xdotool key ctrl+c"
	                LONG_ACTION="xdotool key ctrl+v"
	                ;;
	        esac
	        ;;
	    *)
	        echo "不支持的侧键编号：$KEY_NUM"
	        exit 1
	        ;;
	esac
	
	# 长按即时触发逻辑（复用之前的优化版，减少延迟）
	TIMER_PID_FILE="/tmp/mouse_timer_${KEY_NUM}.pid"
	LONG_EXECUTED_FILE="/tmp/mouse_long_executed_${KEY_NUM}"
	
	case "$2" in
	    press)
	        rm -f "$LONG_EXECUTED_FILE" "$TIMER_PID_FILE"
	        # 启动后台定时器，达到阈值立即执行长按
	        (
	            start_time=$(date +%s%3N)
	            while true; do
	                current_time=$(date +%s%3N)
	                duration=$((current_time - start_time))
	                if [ $duration -ge $THRESHOLD ]; then
	                    eval "$LONG_ACTION"
	                    touch "$LONG_EXECUTED_FILE"
	                    exit 0
	                fi
	                if [ ! -f "$TIMER_PID_FILE" ]; then exit 0; fi
	                usleep 10000  # 10ms检查一次
	            done
	        ) &
	        echo $! > "$TIMER_PID_FILE"
	        ;;
	    release)
	        # 未执行长按则执行短按，同时终止定时器
	        if [ ! -f "$LONG_EXECUTED_FILE" ]; then
	            eval "$SHORT_ACTION"
	        fi
	        if [ -f "$TIMER_PID_FILE" ]; then
	            kill $(cat "$TIMER_PID_FILE") 2>/dev/null
	            rm -f "$TIMER_PID_FILE"
	        fi
	        rm -f "$LONG_EXECUTED_FILE"
	        ;;
	esac
	```

```shell
chmod +x ~/.mouse_side_key_handler.sh
vim ~/.xbindkeysrc
```

- ```
	# 侧键8：短按回车，长按显示桌面
	"~/.mouse_side_key_handler.sh 8 press"   # 按下时记录时间（参数8指定侧键）
	  b:8
	"~/.mouse_side_key_handler.sh 8 release" # 释放时判断长短按
	  release + b:8
	
	# 侧键9：短按ctrl+shift+c，长按ctrl+shift+v
	"~/.mouse_side_key_handler.sh 9 press"   # 按下时记录时间（参数9指定侧键）
	  b:9
	"~/.mouse_side_key_handler.sh 9 release" # 释放时判断长短按
	  release + b:9
	```

```shell
xbindkeys -p  # 重新加载配置
```



