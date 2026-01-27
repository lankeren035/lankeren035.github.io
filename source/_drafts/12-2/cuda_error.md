> ImportError: .../torch/lib/libtorch_cpu.so: **undefined symbol: iJIT_NotifyEvent**

- PyTorch + MKL 组合问题。当环境里装了 **MKL 2024.1（或更新）** 时，导入 torch 会报 `iJIT_NotifyEvent` 这个符号找不到。官方 issue 里直接说：降级 MKL 到 2024.0.0 就能解决。

- python -c "import torch; print(torch.__version__)"如果输出上面的错误那就是这个问题
- conda list mkl发现版本较高

```shell
conda install "mkl<2024.1"

```







- 下载torch是要注意torch2.5可能是被不同cuda编译的版本，要根据自己的系统的cuda版本来下载