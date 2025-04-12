---
title: python中添加日志记录
date: 2025-3-11 10:00:00
tags: [python,debug]
categories: [python]
comment: true
toc: true

---

#

<!--more-->

# 1. 使用日志

- 需要运行`a.py` ， 想要在`a.py`运行过程中输出内容到文件里，而不是输出到终端。（终端能显示的内容有限）
  - 创建一个文件`b.py`
  - 在`a.py`中import就可以实现修改的print函数，也可以使用日志类

```python
import logging
import os
import builtins
from logging.handlers import RotatingFileHandler
from pathlib import Path

_original_print = builtins.print
_logger_cache = {}

class CustomLogger:
    def __init__(self, name=None, log_file="app.log", console_output=False):  # 修改默认值为False
        # 生成唯一日志器名称
        name = name or f"Logger_{id(self)}"
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False  # 关键：关闭日志传播
        self.log_file = log_file
        self.console_output = console_output
        
        # 清理旧处理器（关键修复）
        if self.logger.handlers:
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
        
        self._setup_handlers()

    def _setup_handlers(self):
        """配置日志处理器"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # 文件处理器
        file_handler = RotatingFileHandler(
            log_dir / self.log_file,
            maxBytes=1024*1024,
            backupCount=5,
            encoding="utf-8"
        )
        
        formatter = logging.Formatter(
            "PID:%(process)-6d | %(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        if self.console_output:  # 按需添加控制台输出
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs, exc_info=True)

def get_logger(log_file):
    if log_file not in _logger_cache:
        _logger_cache[log_file] = CustomLogger(
            name=f"PrintLogger_{log_file}",  # 使用唯一名称
            log_file=log_file,
            console_output=False
        ).logger
    return _logger_cache[log_file]

def custom_print(*args, **kwargs):
    log_file = kwargs.pop('log_file', 'prints.log')
    _original_print(*args, **kwargs)
    
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    message = sep.join(map(str, args)) + end.rstrip('\n')
    
    logger = get_logger(log_file)
    logger.info(message)

builtins.print = custom_print
print("\n")
print( "===== 开始日志记录: 进程 PID {} =====".format(os.getpid()) )
```