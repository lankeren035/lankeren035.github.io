import re
import argparse
from pathlib import Path

# 匹配：没有语言标记的 fenced code block
# 开始行：``` 后只能有空白，然后换行（不允许 ```python）
# 结束行：单独一行 ```
PATTERN = re.compile(r'(?ms)^[ \t]*```[ \t]*\r?\n(.*?)^[ \t]*```[ \t]*\r?$')

def codeblock_to_quote(md: str) -> str:
    def repl(m: re.Match) -> str:
        content = m.group(1).rstrip("\n")
        lines = content.splitlines()
        # 每行加 >，空行也变成 >
        return "\n".join(("> " + line) if line.strip() != "" else ">" for line in lines)

    return PATTERN.sub(repl, md)

def main():
    parser = argparse.ArgumentParser(
        description="Convert markdown fenced code blocks without language (```\\n...\\n```) into blockquotes (>)"
    )
    parser.add_argument("--input", default="D:\\blog\\blog\\source\\_posts\\leecode\\1_base\\1_相向双指针.md")
    parser.add_argument("-o", "--output", help="Output file path (default: overwrite input)")
    parser.add_argument("--dry-run", action="store_true", help="Do not write file, just report changes")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"File not found: {in_path}")

    text = in_path.read_text(encoding="utf-8")
    new_text = codeblock_to_quote(text)

    if new_text == text:
        print("No changes made (no matching code blocks found).")
        return

    if args.dry_run:
        print("Changes would be made. (dry-run: not writing)")
        return

    out_path = Path(args.output) if args.output else in_path
    out_path.write_text(new_text, encoding="utf-8")
    print(f"Wrote updated markdown to: {out_path}")

if __name__ == "__main__":
    main()
