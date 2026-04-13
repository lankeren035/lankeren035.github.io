---
title: 使用obsidian学习代码
date: 2026-04-05
tags:
  - 经验
  - 3DGS
categories:
  - 经验
comment: true
toc: true
published: true
hexo-path:
permalink: experience/obsidian
---

#
<!--more-->

## 1. 创建目录结构
- 首先创建一些目录结构，跟原始项目代码保持一致：{% fold "展开查看代码" %}
```python
#!/usr/bin/env python3

# -*- coding: utf-8 -*-

  

"""

根据项目代码结构自动生成 Obsidian 分析脚手架。

  

输出结构（示例）:

output/

├─ maps/

│  └─ project_blueprint.md

├─ files/

│  └─ src/test1.md

├─ modules/

│  └─ src/test1/Cla1.md

└─ functions/

   ├─ src/test1/func1.md

   └─ src/test1/Cla1/method1.md

  

能力说明：

1. Python：使用 ast 精确提取顶层函数、类、方法、__init__ 参数、self.xxx 属性。

2. C/C++：使用启发式静态提取类/结构体、自由函数、方法、构造函数参数、成员变量。

   对模板、宏、函数指针、复杂声明的支持有限，但一般工程代码可用。

3. 生成的 Markdown 默认带有 Obsidian 可点击链接。

4. maps/project_blueprint.md 只生成粗粒度蓝图，不尝试自动分析调用图。

  

注意：

- 该脚本负责“搭脚手架”和“抽取静态签名信息”，不会自动写出真正的逻辑解释。

- “参数解释 / 模块解释 / 输出解释”等内容会预留表格，方便你后续补写。

"""

  

from __future__ import annotations

  

import argparse

import ast

import re

from dataclasses import dataclass, field

from pathlib import Path

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

  
  

# -----------------------------

# 数据结构

# -----------------------------

  

@dataclass

class ParamInfo:

    name: str

    type_hint: str = ""

    desc: str = ""

  
  

@dataclass

class FunctionInfo:

    name: str

    params: List[ParamInfo] = field(default_factory=list)

    return_type: str = ""

    lineno: Optional[int] = None

    doc: str = ""

    is_method: bool = False

    parent_class: Optional[str] = None

    decorators: List[str] = field(default_factory=list)

    synthetic_kind: str = ""

  
  

@dataclass

class AttributeInfo:

    name: str

    type_hint: str = ""

    source_method: str = ""

    desc: str = ""

  
  

@dataclass

class ClassInfo:

    name: str

    lineno: Optional[int] = None

    doc: str = ""

    bases: List[str] = field(default_factory=list)

    init_params: List[ParamInfo] = field(default_factory=list)

    methods: List[FunctionInfo] = field(default_factory=list)

    attributes: List[AttributeInfo] = field(default_factory=list)

  
  

@dataclass

class FileInfo:

    abs_path: Path

    rel_path: Path

    language: str

    classes: List[ClassInfo] = field(default_factory=list)

    functions: List[FunctionInfo] = field(default_factory=list)

    imports: List[str] = field(default_factory=list)

    parse_error: str = ""

  
  

# -----------------------------

# 通用工具

# -----------------------------

  

SUPPORTED_PY = {".py"}

SUPPORTED_CPP = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}

SUPPORTED_CODE = SUPPORTED_PY | SUPPORTED_CPP

  

CONTROL_KEYWORDS = {

    "if", "for", "while", "switch", "catch", "return", "sizeof", "else", "do"

}

  
  

def normalize_exts(items: Sequence[str]) -> set[str]:

    out = set()

    for x in items:

        x = x.strip().lower()

        if not x:

            continue

        if not x.startswith("."):

            x = "." + x

        out.add(x)

    return out

  
  

def ensure_parent(path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

  
  

def write_text(path: Path, text: str) -> None:

    ensure_parent(path)

    path.write_text(text, encoding="utf-8")

  
  

def path_no_suffix(rel_path: Path) -> Path:

    return rel_path.with_suffix("")

  
  

def dotted_rel_path(rel_path_without_suffix: Path) -> str:

    return ".".join(rel_path_without_suffix.parts)

  
  

def normalize_symbol_name(name: str) -> str:

    return "0_main" if name == "main" else name

  
  

def qualified_note_name(rel_path_without_suffix: Path, *names: str) -> str:

    parts = [dotted_rel_path(rel_path_without_suffix)]

    parts.extend([normalize_symbol_name(x) for x in names if x])

    return ".".join(parts)

  
  

def files_note_rel(rel_path_without_suffix: Path) -> Path:

    return Path("files") / rel_path_without_suffix / f"{qualified_note_name(rel_path_without_suffix)}.md"

  
  

def module_note_rel(rel_path_without_suffix: Path, class_name: str) -> Path:

    return Path("modules") / rel_path_without_suffix / f"{qualified_note_name(rel_path_without_suffix, class_name)}.md"

  
  

def function_note_rel(rel_path_without_suffix: Path, func_name: str) -> Path:

    return Path("functions") / rel_path_without_suffix / f"{qualified_note_name(rel_path_without_suffix, func_name)}.md"

  
  

def method_note_rel(rel_path_without_suffix: Path, class_name: str, method_name: str) -> Path:

    return Path("functions") / rel_path_without_suffix / class_name / f"{qualified_note_name(rel_path_without_suffix, class_name, method_name)}.md"

  
  

def obsidian_link(note_rel_path: Path, alias: Optional[str] = None) -> str:

    target = note_rel_path.with_suffix("")

    if alias:

        return f"[[{target.as_posix()}|{alias}]]"

    return f"[[{target.as_posix()}]]"

  
  

def short_doc(doc: str) -> str:

    if not doc:

        return ""

    line = doc.strip().splitlines()[0].strip()

    return re.sub(r"\s+", " ", line)

  
  

def md_escape(text: str) -> str:

    if text is None:

        return ""

    text = str(text)

    return text.replace("|", r"\|").replace("\n", " ").strip()

  
  

def make_param_table(params: List[ParamInfo]) -> str:

    lines = [

        "| 参数名 | 类型 | 解释 |",

        "|---|---|---|",

    ]

    if not params:

        lines.append("| - | - | - |")

    else:

        for p in params:

            lines.append(f"| {md_escape(p.name)} | {md_escape(p.type_hint) or '-'} | {md_escape(p.desc) or '-'} |")

    return "\n".join(lines)

  
  

def make_method_table(rows: List[Tuple[str, str]]) -> str:

    lines = [

        "| 模块名 | 解释 |",

        "|---|---|",

    ]

    if not rows:

        lines.append("| - | - |")

    else:

        for name, desc in rows:

            lines.append(f"| {name} | {md_escape(desc) or '-'} |")

    return "\n".join(lines)

  
  

def make_attr_table(attrs: List[AttributeInfo]) -> str:

    lines = [

        "| 属性名 | 类型 | 来源函数 | 解释 |",

        "|---|---|---|---|",

    ]

    if not attrs:

        lines.append("| - | - | - | - |")

    else:

        for a in attrs:

            lines.append(

                f"| {md_escape(a.name)} | {md_escape(a.type_hint) or '-'} | {md_escape(a.source_method) or '-'} | {md_escape(a.desc) or '-'} |"

            )

    return "\n".join(lines)

  
  

def make_simple_table(headers: List[str], rows: List[List[str]]) -> str:

    head = "| " + " | ".join(headers) + " |"

    sep = "|" + "|".join(["---"] * len(headers)) + "|"

    lines = [head, sep]

    if not rows:

        lines.append("| " + " | ".join(["-"] * len(headers)) + " |")

    else:

        for row in rows:

            row = [(md_escape(x) or "-") for x in row]

            lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)

  
  

# -----------------------------

# Python 解析

# -----------------------------

  
  

def _ast_to_text(node: Optional[ast.AST]) -> str:

    if node is None:

        return ""

    try:

        return ast.unparse(node)  # py>=3.9

    except Exception:

        return ""

  
  

class PythonExtractor:

    def __init__(self, source: str):

        self.source = source

        self.tree = ast.parse(source)

  

    def extract(self, abs_path: Path, rel_path: Path) -> FileInfo:

        file_info = FileInfo(abs_path=abs_path, rel_path=rel_path, language="python")

        file_info.imports = self._extract_imports()

  

        for node in self.tree.body:

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):

                file_info.functions.append(self._parse_function(node, is_method=False, parent_class=None))

            elif isinstance(node, ast.ClassDef):

                file_info.classes.append(self._parse_class(node))

  

        if not any(fn.name == "main" for fn in file_info.functions):

            py_main = self._extract_python_main_guard()

            if py_main is not None:

                file_info.functions.append(py_main)

  

        return file_info

  

    def _extract_imports(self) -> List[str]:

        imports: List[str] = []

        for node in self.tree.body:

            if isinstance(node, ast.Import):

                for alias in node.names:

                    imports.append(alias.name)

            elif isinstance(node, ast.ImportFrom):

                mod = node.module or ""

                names = ", ".join(a.name for a in node.names)

                imports.append(f"from {mod} import {names}")

        return imports

  

    def _extract_python_main_guard(self) -> Optional[FunctionInfo]:

        for node in self.tree.body:

            if not isinstance(node, ast.If):

                continue

            if self._is_python_main_guard(node.test):

                return FunctionInfo(

                    name="main",

                    params=[],

                    return_type="",

                    lineno=getattr(node, "lineno", None),

                    doc="由 if __name__ == '__main__' 入口块生成的入口笔记",

                    is_method=False,

                    parent_class=None,

                    decorators=[],

                    synthetic_kind="python_main_guard",

                )

        return None

  

    def _is_python_main_guard(self, test: ast.AST) -> bool:

        if not isinstance(test, ast.Compare):

            return False

        if len(test.ops) != 1 or len(test.comparators) != 1:

            return False

        if not isinstance(test.ops[0], ast.Eq):

            return False

  

        left_ok = isinstance(test.left, ast.Name) and test.left.id == "__name__"

        right = test.comparators[0]

        right_ok = isinstance(right, ast.Constant) and right.value == "__main__"

        return left_ok and right_ok

  

    def _parse_parameters(self, node: ast.arguments, drop_first: bool) -> List[ParamInfo]:

        params: List[ParamInfo] = []

  

        args = list(node.posonlyargs) + list(node.args)

        defaults = [None] * (len(args) - len(node.defaults)) + list(node.defaults)

  

        items: List[Tuple[ast.arg, Optional[ast.expr], str]] = []

        for arg, default in zip(args, defaults):

            items.append((arg, default, ""))

  

        if node.vararg:

            items.append((node.vararg, None, "*args"))

  

        for kwarg, default in zip(node.kwonlyargs, node.kw_defaults):

            items.append((kwarg, default, ""))

  

        if node.kwarg:

            items.append((node.kwarg, None, "**kwargs"))

  

        if drop_first and items:

            first_name = items[0][0].arg

            if first_name in {"self", "cls"}:

                items = items[1:]

  

        for arg, default, forced_name in items:

            name = forced_name or arg.arg

            typ = _ast_to_text(arg.annotation)

            if default is not None:

                dft = _ast_to_text(default)

                if dft:

                    typ = f"{typ} = {dft}" if typ else f"default={dft}"

            params.append(ParamInfo(name=name, type_hint=typ, desc=""))

        return params

  

    def _decorators(self, node: ast.AST) -> List[str]:

        return [_ast_to_text(d) for d in getattr(node, "decorator_list", [])]

  

    def _parse_function(self, node: ast.AST, is_method: bool, parent_class: Optional[str]) -> FunctionInfo:

        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))

        return FunctionInfo(

            name=node.name,

            params=self._parse_parameters(node.args, drop_first=is_method),

            return_type=_ast_to_text(node.returns),

            lineno=getattr(node, "lineno", None),

            doc=short_doc(ast.get_docstring(node) or ""),

            is_method=is_method,

            parent_class=parent_class,

            decorators=self._decorators(node),

            synthetic_kind="",

        )

  

    def _parse_class(self, node: ast.ClassDef) -> ClassInfo:

        cls = ClassInfo(

            name=node.name,

            lineno=getattr(node, "lineno", None),

            doc=short_doc(ast.get_docstring(node) or ""),

            bases=[_ast_to_text(b) for b in node.bases],

        )

  

        attr_map: Dict[Tuple[str, str], AttributeInfo] = {}

  

        for item in node.body:

            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):

                fn = self._parse_function(item, is_method=True, parent_class=node.name)

                if item.name == "__init__":

                    cls.init_params = fn.params

                else:

                    cls.methods.append(fn)

                self._collect_self_attrs(item, attr_map)

            elif isinstance(item, ast.AnnAssign):

                # 类变量：x: int = 1

                name = self._extract_name(item.target)

                if name:

                    attr_map[(name, "<class>")] = AttributeInfo(

                        name=name,

                        type_hint=_ast_to_text(item.annotation),

                        source_method="<class>",

                        desc="",

                    )

            elif isinstance(item, ast.Assign):

                for t in item.targets:

                    name = self._extract_name(t)

                    if name:

                        attr_map[(name, "<class>")] = AttributeInfo(

                            name=name,

                            type_hint="",

                            source_method="<class>",

                            desc="",

                        )

  

        cls.attributes = sorted(attr_map.values(), key=lambda x: (x.source_method, x.name))

        return cls

  

    def _extract_name(self, node: ast.AST) -> str:

        if isinstance(node, ast.Name):

            return node.id

        return ""

  

    def _collect_self_attrs(self, func_node: ast.AST, attr_map: Dict[Tuple[str, str], AttributeInfo]) -> None:

        assert isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef))

        for child in ast.walk(func_node):

            target_nodes: List[ast.AST] = []

            ann = ""

            if isinstance(child, ast.Assign):

                target_nodes = child.targets

            elif isinstance(child, ast.AnnAssign):

                target_nodes = [child.target]

                ann = _ast_to_text(child.annotation)

            elif isinstance(child, ast.AugAssign):

                target_nodes = [child.target]

  

            for t in target_nodes:

                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":

                    key = (t.attr, func_node.name)

                    if key not in attr_map:

                        attr_map[key] = AttributeInfo(

                            name=t.attr,

                            type_hint=ann,

                            source_method=func_node.name,

                            desc="",

                        )

  
  

# -----------------------------

# C/C++ 解析（启发式）

# -----------------------------

  
  

def strip_comments_and_strings(text: str) -> str:

    # 保留换行，避免行号全乱掉

    pattern = re.compile(

        r'//.*?$|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',

        re.DOTALL | re.MULTILINE,

    )

  

    def repl(match: re.Match) -> str:

        s = match.group(0)

        return "\n" * s.count("\n") if "\n" in s else ""

  

    return pattern.sub(repl, text)

  
  

def find_matching_brace(text: str, start_idx: int) -> int:

    depth = 0

    for i in range(start_idx, len(text)):

        ch = text[i]

        if ch == "{":

            depth += 1

        elif ch == "}":

            depth -= 1

            if depth == 0:

                return i

    return -1

  
  

def split_params_cpp(param_str: str) -> List[str]:

    items: List[str] = []

    cur = []

    depth = 0

    for ch in param_str:

        if ch in "(<[{":

            depth += 1

        elif ch in ")>]}":

            depth -= 1

        if ch == "," and depth == 0:

            item = "".join(cur).strip()

            if item:

                items.append(item)

            cur = []

        else:

            cur.append(ch)

    tail = "".join(cur).strip()

    if tail:

        items.append(tail)

    return items

  
  

def parse_cpp_param(raw: str) -> ParamInfo:

    raw = raw.strip()

    raw = re.sub(r"\s*=\s*.*$", "", raw)

    if not raw or raw == "void":

        return ParamInfo(name="void", type_hint="", desc="")

    if raw == "...":

        return ParamInfo(name="...", type_hint="", desc="")

  

    # 函数指针等复杂参数直接整体保留

    if "(" in raw and ")" in raw and "*" in raw and re.search(r"\(\s*\*\s*\w+\s*\)", raw):

        m = re.search(r"\(\s*\*\s*(\w+)\s*\)", raw)

        name = m.group(1) if m else raw

        return ParamInfo(name=name, type_hint=raw, desc="")

  

    parts = raw.split()

    if len(parts) == 1:

        return ParamInfo(name=parts[0], type_hint="", desc="")

  

    name = parts[-1]

    typ = " ".join(parts[:-1])

  

    # 处理 *name / &name

    while name and name[0] in "*&":

        typ = (typ + " " + name[0]).strip()

        name = name[1:]

  

    # 数组参数 name[]

    m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*(\[.*\])", name)

    if m:

        name = m.group(1)

        typ = (typ + " " + m.group(2)).strip()

  

    return ParamInfo(name=name, type_hint=typ, desc="")

  
  

def parse_cpp_params(param_str: str) -> List[ParamInfo]:

    raw_items = split_params_cpp(param_str)

    params = [parse_cpp_param(x) for x in raw_items if x.strip()]

    if len(params) == 1 and params[0].name == "void":

        return []

    return params

  
  

def parse_cpp_members(class_body: str) -> List[AttributeInfo]:

    attrs: List[AttributeInfo] = []

    for line in class_body.splitlines():

        s = line.strip()

        if not s:

            continue

        if s.endswith(":") and s.rstrip(":") in {"public", "private", "protected"}:

            continue

        if "(" in s or ")" in s:

            continue

        if not s.endswith(";"):

            continue

        s = s[:-1].strip()

        if not s:

            continue

        if s.startswith("using ") or s.startswith("typedef "):

            continue

        parts = s.split()

        if len(parts) < 2:

            continue

        name = parts[-1]

        typ = " ".join(parts[:-1])

        while name and name[0] in "*&":

            typ = (typ + " " + name[0]).strip()

            name = name[1:]

        if name:

            attrs.append(AttributeInfo(name=name, type_hint=typ, source_method="<class>", desc=""))

    return attrs

  
  

def extract_cpp_methods(class_name: str, class_body: str) -> Tuple[List[FunctionInfo], List[ParamInfo]]:

    methods: List[FunctionInfo] = []

    init_params: List[ParamInfo] = []

    seen = set()

  

    method_re = re.compile(

        r"(?P<prefix>[~\w:<>,\*&\s]+?)\b(?P<name>[A-Za-z_~][A-Za-z0-9_]*)\s*\((?P<params>[^;{}()]*(?:\([^)]*\)[^;{}()]*)*)\)\s*(?:const\s*)?(?:noexcept\s*)?(?:=\s*0\s*)?(?:;|\{)",

        re.MULTILINE,

    )

  

    for m in method_re.finditer(class_body):

        name = m.group("name")

        if name in CONTROL_KEYWORDS:

            continue

        params = parse_cpp_params(m.group("params"))

        prefix = re.sub(r"\s+", " ", m.group("prefix")).strip()

        key = (name, tuple((p.name, p.type_hint) for p in params), prefix)

        if key in seen:

            continue

        seen.add(key)

  

        if name == class_name or name == f"~{class_name}":

            if name == class_name and not init_params:

                init_params = params

            continue

  

        return_type = prefix.strip()

        methods.append(

            FunctionInfo(

                name=name,

                params=params,

                return_type=return_type,

                lineno=None,

                doc="",

                is_method=True,

                parent_class=class_name,

                decorators=[],

                synthetic_kind="",

            )

        )

  

    return methods, init_params

  
  

def parse_cpp_file(abs_path: Path, rel_path: Path, text: str) -> FileInfo:

    clean = strip_comments_and_strings(text)

    info = FileInfo(abs_path=abs_path, rel_path=rel_path, language="cpp")

  

    class_ranges: List[Tuple[int, int]] = []

  

    class_head_re = re.compile(r"\b(class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)\b([^\{;]*)\{", re.MULTILINE)

    for m in class_head_re.finditer(clean):

        name = m.group(2)

        brace_start = clean.find("{", m.start())

        brace_end = find_matching_brace(clean, brace_start)

        if brace_start == -1 or brace_end == -1:

            continue

        class_ranges.append((m.start(), brace_end))

        body = clean[brace_start + 1: brace_end]

        bases_raw = m.group(3).strip()

        bases = []

        if ":" in bases_raw:

            inherit_part = bases_raw.split(":", 1)[1]

            for part in inherit_part.split(","):

                bases.append(re.sub(r"\b(public|private|protected|virtual)\b", "", part).strip())

        methods, init_params = extract_cpp_methods(name, body)

        attrs = parse_cpp_members(body)

        info.classes.append(

            ClassInfo(

                name=name,

                lineno=None,

                doc="",

                bases=[x for x in bases if x],

                init_params=init_params,

                methods=methods,

                attributes=attrs,

            )

        )

  

    # 去掉类体，再解析自由函数

    masked = list(clean)

    for s, e in class_ranges:

        for i in range(s, min(e + 1, len(masked))):

            if masked[i] != "\n":

                masked[i] = " "

    free_text = "".join(masked)

  

    func_re = re.compile(

        r"(?P<ret>[\w:<>,\*&\s~]+?)\b(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\((?P<params>[^;{}()]*(?:\([^)]*\)[^;{}()]*)*)\)\s*(?:const\s*)?(?:noexcept\s*)?(?:->\s*[\w:<>,\*&\s]+)?\s*(?:;|\{)",

        re.MULTILINE,

    )

  

    seen = set()

    for m in func_re.finditer(free_text):

        name = m.group("name")

        ret = re.sub(r"\s+", " ", m.group("ret")).strip()

        if name in CONTROL_KEYWORDS:

            continue

        if re.search(r"\b(class|struct|enum|namespace|typedef|using)\b", ret):

            continue

        params = parse_cpp_params(m.group("params"))

        key = (name, tuple((p.name, p.type_hint) for p in params), ret)

        if key in seen:

            continue

        seen.add(key)

        info.functions.append(

            FunctionInfo(

                name=name,

                params=params,

                return_type=ret,

                lineno=None,

                doc="",

                is_method=False,

                parent_class=None,

                decorators=[],

                synthetic_kind="",

            )

        )

  

    if not any(fn.name == "main" for fn in info.functions):

        main_match = re.search(

            r"\b(?P<ret>[A-Za-z_][\w:<>,\*&\s]*)\bmain\s*\((?P<params>[^)]*)\)",

            free_text,

            re.MULTILINE,

        )

        if main_match:

            info.functions.append(

                FunctionInfo(

                    name="main",

                    params=parse_cpp_params(main_match.group("params")),

                    return_type=re.sub(r"\s+", " ", main_match.group("ret")).strip(),

                    lineno=None,

                    doc="由入口检测补充生成的 main 入口笔记",

                    is_method=False,

                    parent_class=None,

                    decorators=[],

                    synthetic_kind="detected_main",

                )

            )

  

    return info

  
  

# -----------------------------

# 单文件处理

# -----------------------------

  
  

def analyze_file(root: Path, file_path: Path) -> Optional[FileInfo]:

    rel_path = file_path.relative_to(root)

    suffix = file_path.suffix.lower()

  

    try:

        text = file_path.read_text(encoding="utf-8")

    except UnicodeDecodeError:

        try:

            text = file_path.read_text(encoding="utf-8-sig")

        except UnicodeDecodeError:

            try:

                text = file_path.read_text(encoding="gb18030")

            except Exception as e:

                lang = "text"

                return FileInfo(abs_path=file_path, rel_path=rel_path, language=lang, parse_error=f"读取失败: {e}")

    except Exception as e:

        return FileInfo(abs_path=file_path, rel_path=rel_path, language="text", parse_error=f"读取失败: {e}")

  

    if suffix in SUPPORTED_PY:

        try:

            return PythonExtractor(text).extract(file_path, rel_path)

        except Exception as e:

            return FileInfo(abs_path=file_path, rel_path=rel_path, language="python", parse_error=f"Python 解析失败: {e}")

  

    if suffix in SUPPORTED_CPP:

        try:

            return parse_cpp_file(file_path, rel_path, text)

        except Exception as e:

            return FileInfo(abs_path=file_path, rel_path=rel_path, language="cpp", parse_error=f"C/C++ 解析失败: {e}")

  

    return FileInfo(abs_path=file_path, rel_path=rel_path, language="text")

  
  

# -----------------------------

# Markdown 生成

# -----------------------------

  
  

def build_file_md(fi: FileInfo) -> str:

    rel_no_suffix = path_no_suffix(fi.rel_path)

    title = qualified_note_name(rel_no_suffix)

  

    class_rows = []

    for c in fi.classes:

        link = obsidian_link(module_note_rel(rel_no_suffix, c.name), alias=c.name)

        class_rows.append([link, c.doc, str(c.lineno or "-")])

  

    func_rows = []

    for f in fi.functions:

        link = obsidian_link(function_note_rel(rel_no_suffix, f.name), alias=f.name)

        func_rows.append([link, f.return_type, str(f.lineno or "-"), f.doc])

  

    imports_text = "\n".join([f"- `{x}`" for x in fi.imports]) if fi.imports else "- -"

  

    lines = [

        f"# {title}",

        "",

        "<!--more-->",

        "",

        "## 1. 文件信息",

        "",

        f"- 相对路径：`{fi.rel_path.as_posix()}`",

        f"- 语言：`{fi.language}`",

        f"- 源文件：`{fi.abs_path}`",

    ]

  

    if fi.parse_error:

        lines += [

            f"- 解析状态：`{md_escape(fi.parse_error)}`",

            "",

        ]

    else:

        lines += [

            "- 解析状态：`ok`",

            "",

        ]

  

    lines += [

        "## 2. 包含的类",

        "",

        make_simple_table(["类名", "说明", "行号"], class_rows),

        "",

        "## 3. 包含的函数",

        "",

        make_simple_table(["函数名", "返回", "行号", "说明"], func_rows),

        "",

        "## 4. 依赖 / 导入",

        "",

        imports_text,

        "",

        "## 5. 备注",

        "",

        "- 这里用于补充该文件在整体工程中的职责、调用入口、与其它文件的关系。",

        "",

    ]

    return "\n".join(lines)

  

def build_function_md(fi: FileInfo, fn: FunctionInfo) -> str:

    rel_no_suffix = path_no_suffix(fi.rel_path)

    if fn.parent_class:

        note_name = qualified_note_name(rel_no_suffix, fn.parent_class, fn.name)

    else:

        note_name = qualified_note_name(rel_no_suffix, fn.name)

    display_name = fn.name

    file_link = obsidian_link(files_note_rel(rel_no_suffix), alias=qualified_note_name(rel_no_suffix))

    parent_link = ""

    if fn.parent_class:

        parent_link = obsidian_link(module_note_rel(rel_no_suffix, fn.parent_class), alias=fn.parent_class)

  

    lines = [

        f"# {note_name}",

        "",

        "<!--more-->",

        "",

        "## 0. 概览",

        "",

        f"- 所属文件：{file_link}",

    ]

    if parent_link:

        lines.append(f"- 所属类：{parent_link}")

    lines += [

        f"- 名称：`{display_name}`",

        f"- 返回：`{fn.return_type or '-'}`",

        f"- 行号：`{fn.lineno or '-'}`",

    ]

    if fn.decorators:

        lines.append(f"- 装饰器：`{', '.join(fn.decorators)}`")

    if fn.doc:

        lines.append(f"- 摘要：{fn.doc}")

    if fn.synthetic_kind:

        lines.append(f"- 类型：`{fn.synthetic_kind}`")

    lines += [

        "",

        "## 1. 输入",

        "",

        make_param_table(fn.params),

        "",

        "## 2. 输出",

        "",

        make_simple_table(["返回类型", "解释"], [[fn.return_type or "-", "-"]]),

        "",

        "## 3. 操作逻辑",

        "",

        "1. 待补充",

        "",

        "## 4. 备注",

        "",

        "- 这里可补充：副作用、异常、关键分支、调用链位置。",

        "",

    ]

    return "\n".join(lines)

  

def build_class_md(fi: FileInfo, cls: ClassInfo) -> str:

    rel_no_suffix = path_no_suffix(fi.rel_path)

    title = qualified_note_name(rel_no_suffix, cls.name)

    file_link = obsidian_link(files_note_rel(rel_no_suffix), alias=qualified_note_name(rel_no_suffix))

  

    method_rows = []

    for m in cls.methods:

        link = obsidian_link(method_note_rel(rel_no_suffix, cls.name, m.name), alias=m.name)

        method_rows.append((link, m.doc))

  

    lines = [

        f"# {title}",

        "",

        "<!--more-->",

        "",

        "## 0. 概览",

        "",

        f"- 所属文件：{file_link}",

        f"- 类名：`{cls.name}`",

        f"- 行号：`{cls.lineno or '-'}`",

        f"- 继承：`{', '.join(cls.bases) if cls.bases else '-'}`",

    ]

    if cls.doc:

        lines.append(f"- 摘要：{cls.doc}")

  

    lines += [

        "",

        "## 1. 输入",

        "",

        make_param_table(cls.init_params),

        "",

        "## 2. 属性",

        "",

        make_attr_table(cls.attributes),

        "",

        "## 3. 模块",

        "",

        make_method_table(method_rows),

        "",

    ]

  

    for idx, m in enumerate(cls.methods, start=1):

        lines += [

            f"### 3.{idx} {m.name}",

            "",

            make_param_table(m.params),

            "",

        ]

  

    lines += [

        "## 4. 备注",

        "",

        "- 这里可补充：该类的状态机、生命周期、与其它模块的协作关系。",

        "",

    ]

    return "\n".join(lines)

  

def build_project_blueprint(root: Path, file_infos: List[FileInfo]) -> str:

    # 只做粗粒度目录蓝图

    top_stats: Dict[str, Dict[str, int]] = {}

    total_classes = 0

    total_functions = 0

    code_files = 0

  

    for fi in file_infos:

        top = fi.rel_path.parts[0] if fi.rel_path.parts else "."

        bucket = top_stats.setdefault(top, {"files": 0, "code_files": 0, "classes": 0, "functions": 0})

        bucket["files"] += 1

        if fi.language in {"python", "cpp"}:

            bucket["code_files"] += 1

            code_files += 1

        bucket["classes"] += len(fi.classes)

        bucket["functions"] += len(fi.functions) + sum(len(c.methods) for c in fi.classes)

        total_classes += len(fi.classes)

        total_functions += len(fi.functions) + sum(len(c.methods) for c in fi.classes)

  

    rows = []

    for top, stats in sorted(top_stats.items()):

        rows.append([

            top,

            str(stats["files"]),

            str(stats["code_files"]),

            str(stats["classes"]),

            str(stats["functions"]),

        ])

  

    mermaid_lines = [

        "flowchart TD",

        f'    ROOT["{root.name}"]',

    ]

    for top in sorted(top_stats):

        mermaid_lines.append(f'    ROOT --> {re.sub(r"[^A-Za-z0-9_]", "_", top)}["{top}"]')

  

    lines = [

        f"# {root.name} 粗粒度蓝图",

        "",

        "<!--more-->",

        "",

        "## 1. 总览",

        "",

        f"- 项目根目录：`{root}`",

        f"- 扫描文件数：`{len(file_infos)}`",

        f"- 代码文件数：`{code_files}`",

        f"- 类数量：`{total_classes}`",

        f"- 函数 / 方法数量：`{total_functions}`",

        "",

        "## 2. 顶层目录统计",

        "",

        make_simple_table(["顶层目录", "文件数", "代码文件数", "类数", "函数/方法数"], rows),

        "",

        "## 3. 粗粒度结构图",

        "",

        "```mermaid",

        *mermaid_lines,

        "```",

        "",

        "## 4. 使用建议",

        "",

        "1. 先从本文件查看顶层目录划分。",

        "2. 再进入 `files/` 看每个文件包含哪些类和函数。",

        "3. 然后进入 `modules/` / `functions/` 补全具体逻辑解释。",

        "4. 如果后续你手动画了调用图、训练流程图、数据流图，可以继续放到 `maps/` 目录下。",

        "",

    ]

    return "\n".join(lines)

  
  

# -----------------------------

# 输出执行

# -----------------------------

  
  

def emit_docs(output_root: Path, fi: FileInfo) -> None:

    rel_no_suffix = path_no_suffix(fi.rel_path)

  

    # files/

    file_md_path = output_root / files_note_rel(rel_no_suffix)

    write_text(file_md_path, build_file_md(fi))

  

    # 非代码文件只生成 files 卡片

    if fi.language not in {"python", "cpp"}:

        return

  

    # modules/<file_stem>/

    modules_dir = output_root / "modules" / rel_no_suffix

    modules_dir.mkdir(parents=True, exist_ok=True)

  

    # functions/<file_stem>/

    functions_dir = output_root / "functions" / rel_no_suffix

    functions_dir.mkdir(parents=True, exist_ok=True)

  

    # 顶层函数

    for fn in fi.functions:

        fn_path = output_root / function_note_rel(rel_no_suffix, fn.name)

        write_text(fn_path, build_function_md(fi, fn))

  

    # 类及其方法

    for cls in fi.classes:

        cls_path = output_root / module_note_rel(rel_no_suffix, cls.name)

        write_text(cls_path, build_class_md(fi, cls))

  

        if cls.methods:

            cls_func_dir = output_root / "functions" / rel_no_suffix / cls.name

            cls_func_dir.mkdir(parents=True, exist_ok=True)

            for m in cls.methods:

                method_path = output_root / method_note_rel(rel_no_suffix, cls.name, m.name)

                write_text(method_path, build_function_md(fi, m))

  
  

# -----------------------------

# 主流程

# -----------------------------

  
  

def iter_target_files(root: Path, exclude_exts: set[str], exclude_dirs: set[str]) -> Iterable[Path]:

    for p in root.rglob("*"):

        if not p.is_file():

            continue

        if any(part in exclude_dirs for part in p.relative_to(root).parts):

            continue

        if p.suffix.lower() in exclude_exts:

            continue

        yield p

  
  

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="为项目代码自动生成 Obsidian 分析脚手架")

    parser.add_argument("--input", required=True, help="项目根目录")

    parser.add_argument("--output", required=True, help="输出根目录")

    parser.add_argument(

        "--exclude-ext",

        nargs="*",

        default=[".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".avi", ".mov", ".pt", ".ckpt", ".bin"],

        help="要排除的后缀名，可写 .png .mp4 或 png mp4",

    )

    parser.add_argument(

        "--exclude-dir",

        nargs="*",

        default=[".git", ".idea", ".vscode", "__pycache__", "node_modules", "build", "dist"],

        help="要排除的目录名",

    )

    parser.add_argument(

        "--no-maps",

        action="store_true",

        help="不生成 maps/project_blueprint.md",

    )

    return parser.parse_args()

  
  

def main() -> None:

    args = parse_args()

    input_root = Path(args.input).resolve()

    output_root = Path(args.output).resolve()

    exclude_exts = normalize_exts(args.exclude_ext)

    exclude_dirs = set(args.exclude_dir)

  

    if not input_root.exists() or not input_root.is_dir():

        raise SystemExit(f"输入路径不存在或不是目录: {input_root}")

  

    output_root.mkdir(parents=True, exist_ok=True)

  

    file_infos: List[FileInfo] = []

    for p in iter_target_files(input_root, exclude_exts=exclude_exts, exclude_dirs=exclude_dirs):

        fi = analyze_file(input_root, p)

        if fi is None:

            continue

        file_infos.append(fi)

        emit_docs(output_root, fi)

  

    if not args.no_maps:

        maps_dir = output_root / "maps"

        maps_dir.mkdir(parents=True, exist_ok=True)

        blueprint_path = maps_dir / "project_blueprint.md"

        write_text(blueprint_path, build_project_blueprint(input_root, file_infos))

  

    print(f"[DONE] scanned={len(file_infos)} output={output_root}")

  
  

if __name__ == "__main__":

    main()
```
{% endfold %}

- 创建之后会生成四个文件夹：
	- files，存放目录结构，只是将原始的代码文件换成md文件，用于解释每一份文件
	- functions，解释每个函数
	- maps，存放一些跳转图
	- modules，解释每个类
