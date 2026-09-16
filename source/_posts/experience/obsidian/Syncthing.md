---
title: 使用Syncthing多设备同步文件夹
date: 2026-04-03 11:00:00
tags:
  - Syncthing
  - 经验
published: true
hexo-path: /2026/04/03/experience/obsidian/Syncthing
---
#
<!--more-->

# Windows + Windows 用 Syncthing 免费自动同步文件

## 1. 目标

我要解决的问题是：

- 两台 Windows 电脑之间同步 Obsidian
- 不想手动上传下载
- 不想付费
- 尽量自动同步

最终采用：

- Syncthing
- 同步整个 Obsidian vault 文件夹

---


## 3. 安装 Syncthing 时遇到的问题

### 3.1 一开始用了 `Syncthing Windows Setup`

运行后出现一个界面：

```text
Select Installation Zip File
```

这不是报错，而是在要求我提供 Syncthing 本体的 Windows zip 包。

### 3.2 一开始下错了文件

在 GitHub release 页面里，我下载的是：

- `Source code (zip)`
- `Source code (tar.gz)`

这两个都不对。

原因：

- 它们只是安装器项目自己的源码包
- 不是 Syncthing 本体的 Windows 发布包

### 3.3 正确方式

应该下载的文件名类似：

```text
syncthing-windows-amd64-v版本号.zip
```

如果是普通 64 位 Windows，通常选 `amd64` 即可。

### 3.4 更简单的做法

其实没必要卡在 Setup 上。更省事的方法是：

1. 下载官方 Windows zip
2. 解压
3. 双击运行 `syncthing.exe`

---

## 4. 第一次启动后的匿名统计提示


```text
选“否”
```


## 5. 打开 Syncthing 管理页面

Syncthing 启动后，浏览器通常会自动打开：

```text
http://localhost:8384/
```

---



## 7. 两台电脑互相添加设备

两台设备要建立同步关系，必须：

- 每台设备查看自己的 `Device ID`
- 双方互相添加对方

### 7.1 在电脑 A 上查看 Device ID

操作：

1. 打开 `http://localhost:8384/`
2. 右上角点击 `Actions`
3. 点击 `Show ID`
4. 复制这台机器的 Device ID

### 7.2 在电脑 B 上也做同样的事

同样操作：

1. 打开管理页
2. `Actions -> Show ID`
3. 复制电脑 B 的 Device ID

### 7.3 两台电脑互相添加

#### 电脑 A 上

1. 点击 `Add Remote Device`
2. 粘贴电脑 B 的 Device ID
3. Device Name 例如写 `PC-B`
4. 保存

#### 电脑 B 上

1. 点击 `Add Remote Device`
2. 粘贴电脑 A 的 Device ID
3. Device Name 例如写 `PC-A`
4. 保存

做到这里，两台机器会开始建立连接。

---

## 8. 添加并共享 Obsidian vault 文件夹

### 8.1 在电脑 A 上添加文件夹

点击：

```text
Add Folder
```

然后填写：

#### Folder Label

```text
3DGS-Code-Map
```

#### Folder ID

保持默认自动生成即可。

#### Folder Path

选择真实 vault 路径，例如：

```text
C:\Users\用户名\Documents\3DGS-Code-Map
```

#### Folder Type

选择：

```text
Send & Receive
```

这是双向同步模式。

#### Sharing / Devices

勾选刚才添加进来的电脑 B。

最后保存。

---


## 10. 在电脑 B 上接收这个共享文件夹

电脑 B 上也要给这个共享文件夹指定本地路径。

建议先创建一个空文件夹，例如：

```text
C:\Users\用户名\Documents\3DGS-Code-Map
```

然后在电脑 B 上配置同一个共享文件夹。

关键点：

- `Folder ID` 必须和电脑 A 对应的共享文件夹一致
- `Folder Path` 是电脑 B 本地自己的实际路径

也就是说：

- 靠 `Folder ID` 识别这是同一个共享逻辑
- 靠 `Folder Path` 决定它在本机存放的位置

---



## 14. `.obsidian` 文件夹要不要同步

`.obsidian` 里包含：

- 插件配置
- 主题设置
- 工作区配置
- Canvas 相关配置
- 快捷方式等设置
但是有些配置文件，在不同设备填的路径可能不一样，这就需要找到该配置文件，然后在Syncthing分享目录根目录下写一个.stignore，  里面天上应该忽略的文件（比如`.obsidian/plugins/hexo-integration/data.json`）
---

## 15. 两台电脑需要一直开着 Syncthing 吗

### 15.1 结论

不需要 24 小时一直开着。

但如果想做到“这边一改，那边马上同步”，那么两台电脑在某个时间段内都需要：

- 开机
- 运行 Syncthing


### 15.3 开销大吗

对于 Obsidian 这种以 markdown 为主的小型 vault，日常开销通常不大。

开销明显一些的时机主要是：

- 第一次全量同步
- 大量新增/修改文件
- 扫描和哈希校验时
---

## 16. 设置开机自启动


1. 按 `Win + R`
2. 输入：

```text
shell:startup
```

3. 回车，打开当前用户启动文件夹
4. 在这个文件夹里新建快捷方式
5. 目标填写：

```text
你的 syncthing.exe 完整路径 --no-console --no-browser
```

例如：

```text
D:\Syncthing\syncthing.exe --no-console --no-browser
```

这样以后登录 Windows 时，Syncthing 就会自动后台启动。




# ipad上同步
- 安装