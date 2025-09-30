---
title: "3DGS代码调试"
date: 2025-9-20 10:28:00
tags: [3dgs]
categories: [3dgs]
comment: false
toc: true


---

#
<!--more-->

## 1. 调试

- 先在服务器中安装`c/c++`插件

### 1.1 修改文件（跳过）

- 同时调试python和cuda代码需要做额外配置：

```shell
cd ~/code/gaussian-splatting
grep -R "setup.py" -n
```

- 根据输出结果发现这三个目录下有`setup.py`

  - `submodules/simple-knn`

  - `submodules/fused-ssim`

  - `submodules/diff-gaussian-rasterization`

    ```
    scripts/debug.sh:36:  "$PY" setup.py clean  > build.log 2>&1 || true
    scripts/debug.sh:37:  "$PY" setup.py build >> build.log 2>&1
    匹配到二进制文件 .git/modules/submodules/simple-knn/index
    匹配到二进制文件 .git/modules/submodules/fused-ssim/index
    匹配到二进制文件 .git/modules/submodules/diff-gaussian-rasterization/index
    test.py:27:            print(f"[WARN] {b} 里找不到 {WANT} 的 build 目录；请在该子模块用当前 python 重新 `python setup.py build`")
    submodules/simple-knn/simple_knn.egg-info/SOURCES.txt:3:setup.py
    submodules/fused-ssim/fused_ssim.egg-info/SOURCES.txt:4:setup.py
    submodules/fused-ssim/build.log:2:  File "setup.py", line 39, in <module>
    submodules/fused-ssim/build.log:6:  File "setup.py", line 39, in <module>
    submodules/diff-gaussian-rasterization/diff_gaussian_rasterization.egg-info/SOURCES.txt:5:setup.py
    ```

    

- 修改`code/gaussian-splatting/submodules/simple-knn/setup.py`:

  ```python
  from setuptools import setup
  from torch.utils.cpp_extension import CUDAExtension, BuildExtension
  import os
  
  cxx_compiler_flags = []
  
  if os.name == 'nt':
      cxx_compiler_flags.append("/wd4624")
  
  # 判断是否调试
  DEBUG = os.getenv("CUDA_DEBUG_BUILD", "0") == "1"
  if DEBUG:
      extra = {
          # Linux 下把 -fPIC 也加上，更稳
          "cxx":  (cxx_compiler_flags + (["/Od", "/Z7"] if os.name == 'nt' else ["-O0", "-g", "-fPIC"])),
          "nvcc": (["-O0", "-g", "-G", "-lineinfo"] + ([] if os.name == 'nt' else ["-Xcompiler", "-fPIC"])),
      }
  else:
      extra = {"nvcc": [], "cxx": cxx_compiler_flags}
  
  
  setup(
      name="simple_knn",
      ext_modules=[
          CUDAExtension(
              name="simple_knn._C",
              sources=[
              "spatial.cu", 
              "simple_knn.cu",
              "ext.cpp"],
              extra_compile_args=extra,
              )
          ],
      cmdclass={
          'build_ext': BuildExtension
      }
  )
  ```

  

- 修改`code/gaussian-splatting/submodules/fused-ssim/setup.py`：

  ```python
  # from setuptools import setup
  # from torch.utils.cpp_extension import CUDAExtension, BuildExtension
  
  # setup(
  #     name="fused_ssim",
  #     packages=['fused_ssim'],
  #     ext_modules=[
  #         CUDAExtension(
  #             name="fused_ssim_cuda",
  #             sources=[
  #             "ssim.cu",
  #             "ext.cpp"])
  #         ],
  #     cmdclass={
  #         'build_ext': BuildExtension
  #     }
  # )
  
  
  from setuptools import setup
  from torch.utils.cpp_extension import CUDAExtension, BuildExtension
  
  
  
  # 判断是否调试
  import os
  DEBUG = os.getenv("CUDA_DEBUG_BUILD", "0") == "1"
  if DEBUG:
      setup(
          name="fused_ssim",
          packages=['fused_ssim'],
          ext_modules=[
              CUDAExtension(
                  name="fused_ssim_cuda",
                  sources=[
                  "ssim.cu",
                  "ext.cpp"],
                  extra_compile_args={
                      "cxx":  (cxx_compiler_flags + (["/Od", "/Z7"] if os.name == 'nt' else ["-O0", "-g"])),
                      "nvcc": (["-O0", "-g", "-G", "-lineinfo"] + ([] if os.name == 'nt' else ["-Xcompiler", "-fPIC"])),
                  }
                  )
              ],
          cmdclass={
              'build_ext': BuildExtension
          }
      )
  
  else:
      setup(
          name="fused_ssim",
          packages=['fused_ssim'],
          ext_modules=[
              CUDAExtension(
                  name="fused_ssim_cuda",
                  sources=[
                  "ssim.cu",
                  "ext.cpp"])
              ],
          cmdclass={
              'build_ext': BuildExtension
          }
      )
  ```

  

- 修改`code/gaussian-splatting/submodules/diff-gaussian-rasterization/setup.py`：

  ```python
  from setuptools import setup
  from torch.utils.cpp_extension import CUDAExtension, BuildExtension
  import os
  
  this_dir = os.path.dirname(os.path.abspath(__file__))
  glm_inc  = os.path.join(this_dir, "third_party/glm/")
  
  is_windows = os.name == "nt"
  DEBUG = os.getenv("CUDA_DEBUG_BUILD", "0") == "1"
  
  if DEBUG:
      cxx_flags  = (["/Od", "/Z7"] if is_windows else ["-O0", "-g"])
      nvcc_flags = (["-O0", "-g", "-G", "-lineinfo"] +
                    ([] if is_windows else ["-Xcompiler", "-fPIC"]) +
                    ["-I" + glm_inc])
      extra = {"cxx": cxx_flags, "nvcc": nvcc_flags}
  else:
      # Release：保留你原来的 include，不额外加任何 flag
      extra = {"nvcc": ["-I" + glm_inc]}
  
  setup(
      name="diff_gaussian_rasterization",
      packages=["diff_gaussian_rasterization"],
      ext_modules=[
          CUDAExtension(
              name="diff_gaussian_rasterization._C",
              sources=[
                  "cuda_rasterizer/rasterizer_impl.cu",
                  "cuda_rasterizer/forward.cu",
                  "cuda_rasterizer/backward.cu",
                  "rasterize_points.cu",
                  "ext.cpp",
              ],
              extra_compile_args=extra,
          )
      ],
      cmdclass={"build_ext": BuildExtension},
  )
  ```

  

- 新建并运行：

  ```shell
  #!/usr/bin/env bash
  set -euo pipefail
  
  # --- 绝对路径：固定用 3dgs 环境的 python/pip/ninja ---
  ENV_DIR="$HOME/anaconda3/envs/3dgs"
  PY="$ENV_DIR/bin/python"
  PIP="$ENV_DIR/bin/pip"
  NINJA="$ENV_DIR/bin/ninja"
  
  # --- 能用就静默激活 conda：失败也别吵 ---
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" >/dev/null 2>&1 || true
    conda activate 3dgs >/dev/null 2>&1 || true
  fi
  
  # --- ninja：用你自己的源安装；已有就跳过 ---
  if ! "$NINJA" --version >/dev/null 2>&1; then
    conda install -y -n 3dgs ninja >/dev/null 2>&1 || "$PIP" install -q ninja
  fi
  
  # --- 开启调试构建（只 build，不 install） ---
  export CUDA_DEBUG_BUILD=1
  
  # 当前 Python ABI（挑选正确的 build/lib.* 目录）
  PYTAG=$("$PY" - <<'PY'
  import sys; print(f"cpython-{sys.version_info.major}{sys.version_info.minor}")
  PY
  )
  
  # 小工具：安静构建并返回 build/lib.* 路径（只回路径，日志写文件）
  quiet_build() {
    local dir="$1"
    [ -d "$dir" ] || { echo ""; return; }
    cd "$dir"
    # 把构建日志写到 build.log（不污染 stdout）
    "$PY" setup.py clean  > build.log 2>&1 || true
    "$PY" setup.py build >> build.log 2>&1
  
    # 选与当前 Python 匹配的 lib 目录；没有就取最新的
    local best=""
    for p in "$PWD"/build/lib.*; do
      [ -d "$p" ] || continue
      case "$p" in *"$PYTAG"*) best="$p";; esac
    done
    [ -n "$best" ] || best=$(ls -dt "$PWD"/build/lib.* 2>/dev/null | head -n1 || true)
    echo "$best"
  }
  
  ROOT="$HOME/code/gaussian-splatting"
  
  SIMPLE_LIB=$(quiet_build "$ROOT/submodules/simple-knn")
  
  
  FSSIM_LIB=$(quiet_build "$ROOT/submodules/fused-ssim")
  
  
  DGR_LIB=$(quiet_build "$ROOT/submodules/diff-gaussian-rasterization")
  
  # 打印各自路径（纯净）
  echo "$SIMPLE_LIB"
  echo "$FSSIM_LIB"
  echo "$DGR_LIB"
  
  # 组合出干净的 PYTHONPATH（去空）
  PYTHONPATH_VAL=$(printf "%s\n%s\n%s\n" "$SIMPLE_LIB" "$FSSIM_LIB" "$DGR_LIB" | awk 'NF' | paste -sd: -)
  
  echo
  echo "export PYTHONPATH=\"$PYTHONPATH_VAL\""
  echo "(复制上面这一行到当前终端，即可临时使用调试版 .so)"
  echo "注：编译日志保存在各子模块的 build.log（如 submodules/simple-knn/build.log）"
  ```


### 1.1 修改文件



### 1.2 配置文件

```json

{
    "version": "0.2.0",
    "configurations": [
        // 1. Python调试配置（和你原有逻辑一致，仅微调路径映射）
        {
            "name": "Python Attach (6789)",
            "type": "debugpy-old",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 6789
            },
            "justMyCode": false,
            // "pathMappings": [
            //     {
            //         "localRoot": "/home/csu_kcc/code/gaussian-splatting", // 本地项目根目录（VSCode打开的文件夹）
            //         "remoteRoot": "/home/csu_kcc/code/gaussian-splatting" // 服务器上项目的绝对路径（必须写死，避免变量出错）
            //     }
            // ],
            // "stopOnEntry": true // 附加后不自动暂停，方便后续执行到CUDA调用前
        },
        // 2. CUDA调试配置（关键修正）
        {
            "name": "CUDA Attach to Python",
            "type": "cppdbg", // 核心修正：VSCode通过cppdbg类型调用cuda-gdb
            "request": "attach",
            "MIMode": "gdb",
            // 修正：sudo调用cuda-gdb，确保权限（依赖免密码配置）
            "miDebuggerPath": "/usr/local/cuda/bin/cuda-gdb",
            "processId": "${command:pickProcess}", // 选择Python进程（和之前一样）
            "program": "/home/csu_kcc/anaconda3/envs/3dgs/bin/python", // 你的Python解释器（必须绝对路径）
            "environment": [
                {
                    "name": "PYTHONPATH",
                    "value": "/home/csu_kcc/code/gaussian-splatting/submodules/simple-knn/build/lib.linux-x86_64-cpython-37:/home/csu_kcc/code/gaussian-splatting/submodules/fused-ssim/build/lib.linux-x86_64-cpython-37:/home/csu_kcc/code/gaussian-splatting/submodules/diff-gaussian-rasterization/build/lib.linux-x86_64-cpython-37"
                }
            ],
            "setupCommands": [
                {
                    "text": "handle SIGURG nostop noprint pass",
                    "ignoreFailures": true
                },
                // 启用符号美化（方便查看变量）
                { "text": "-enable-pretty-printing", "ignoreFailures": true },
                // 禁止CUDA核函数启动时自动断点（避免一启动就卡）
                { "text": "set cuda break_on_launch none" },
                // 加载CUDA调试符号（避免“无调试符号”警告）
                { "text": "set debug-file-directory /usr/local/cuda/lib/debug" }
            ],
            // 核心修正：用集成终端处理权限（若免密码失效，可在VSCode内输入密码）
            "externalConsole": false,
            "console": "integratedTerminal",
            // 确保调试器能找到CUDA代码（映射.cu文件路径）
            "sourceFileMap": {
                "/home/csu_kcc/code/gaussian-splatting": "/home/csu_kcc/code/gaussian-splatting"
            }
        }
    ]
}

```

- 先运行代码 `python -m debugpy --listen localhost:6789 --wait-for-client train.py -s ~/code/gaussian-splatting/data/input/tandt_db/truck -m ~/code/gaussian-splatting/data/output/tandt_db/truck --eval`

- 然后在左侧调试栏，选择python attach，点击调试按钮

- 当调试到需要进入cuda的部分，在调试栏选择CUDA Attach to Python，然后选择刚刚运行的进程（搜索train.py）

- 选择之后显示：此操作可能会危害计算机需要管理员权限，输入y

- 如果输入y之后直接显示如下内容，则转1.3

  ```shell
  Multiple identities can be used for authentication:
   1.  user,,, (user)
   2.  xxx
  Choose identity to authenticate as (1-2): [1] + Stopped (tty input)        /usr/bin/pkexec "/usr/bin/sudo" --interpreter=mi --tty=${DbgTerm} 0<"/tmp/Microsoft-MIEngine-In-ejoxxylu.2hh" 1>"/tmp/Microsoft-MIEngine-Out-rv5zyxew.go0"
  You have stopped jobs. 
  ```

### 1.3 cuda调试配置

- 修改配置文件

```shell
# 在服务器上执行，允许vs code调试器无需密码使用sudo
echo "$USER ALL=(ALL) NOPASSWD: /usr/bin/gdb, /usr/local/cuda-11.6/bin/cuda-gdb" | sudo tee /etc/sudoers.d/vscode-debug
sudo chmod 0440 /etc/sudoers.d/vscode-debug
sudo visudo -c -f /etc/sudoers.d/vscode-debug
```

- 若后续不需要调试，直接删除该文件即可完全撤销设置： 

  ```shell
  sudo rm /etc/sudoers.d/vscode-debug
  ```

  

- 输入： 如果路径不同，需要修改`sudoers.d/vscode-debug`文件中的路径（修改后记得重新用`visudo -c`验证）。 

```shell
which cuda-gdb  # 输出应与配置中的路径一致
```



- 配置之后如果调试输入y之后仍然直接停止，则：

  ```shell
  sudo nano /etc/polkit-1/localauthority/50-local.d/99-vscode-debug.pkla
  ```

  - 写入：

    ```shell
    [Allow Debuggers without Password]
    Identity=unix-user:<你的用户名>
    Action=org.freedesktop.policykit.exec
    ResultAny=yes
    ResultInactive=yes
    ResultActive=yes
    ```

  - 然后

    ```shell
    sudo chmod 644 /etc/polkit-1/localauthority/50-local.d/99-vscode-debug.pkla
    # 重启PolicyKit守护进程（不同系统可能有差异，尝试以下命令）
    sudo systemctl restart polkit
    # 若上面命令失败，尝试：
    sudo service polkit restart
    ```

    

### 1.4 cuda调试

- 输入y之后进入调试，发现cuda调试进程有一个进程显示exception：

  ```shell
  unable toretrive stack trace. the message is improperly formatted or was damaged in transit
  ```

- 查看CUDA 调试符号路径

  ```shell
  ls /usr/local/cuda-11.6/lib64/debug
  # 或者ls /usr/local/cuda-11.6/lib/debug
  ```

  

- 发现里面不存在debug文件夹，安装cuda调试符号包

```shell
sudo apt install cuda-dbg-11-6
```

- 如果发现 E: 无法定位软件包 cuda-dbg-11-6 

```shell
# 下载 CUDA 11.6 的源配置包（Ubuntu 20.04 对应代号 focal）
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# 将源优先级文件复制到系统目录
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# 下载并安装 CUDA 11.6 源（注意：版本号必须严格匹配 11.6）
wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda-repo-ubuntu2004-11-6-local_11.6.0-510.39.01-1_amd64.deb
# 安装源包（会自动添加 NVIDIA 签名密钥）
sudo dpkg -i cuda-repo-ubuntu2004-11-6-local_11.6.0-510.39.01-1_amd64.deb
# 更新 apt 缓存（让系统识别新添加的源）
cd /var/cuda-repo-ubuntu2004-11-6-local/
sudo apt-key add 7fa2af80.pub
sudo apt update
sudo apt install cuda-dbg-11-6
```

