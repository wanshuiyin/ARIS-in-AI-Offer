#!/usr/bin/env python3
"""Render every docs/tutorials/code/*.py as a readable single-file HTML page.

The hub's code badges used to jump to GitHub's blob view, which is the one
ugly hop on an otherwise self-hosted site. This writes docs/code/<name>.html:
the tutorial top bar, a link back to the sheet the script belongs to, GitHub /
raw-download buttons, and the source with highlight.js and line numbers.

Usage:
    python3 tools/build_code_pages.py
"""

from __future__ import annotations

import html
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = ROOT / "docs" / "tutorials" / "code"
OUT_DIR = ROOT / "docs" / "code"

spec = importlib.util.spec_from_file_location("build_index", ROOT / "tools" / "build_index.py")
build_index = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_index)
REPO, SHORT, CATALOG = build_index.REPO, build_index.SHORT, build_index.CATALOG
RAW = REPO.replace("https://github.com/", "https://raw.githubusercontent.com/") + "/main"

e = html.escape

PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{name} · ARIS in AI Offer</title>
<meta name="description" content="{name} — {desc}">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/atom-one-light.min.css">
<script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/highlight.min.js"></script>
<style>
:root{{--bg:#fdfcf7;--bg-soft:#f4f1ea;--ink:#1a1a1a;--ink-soft:#4a4a4a;--ink-muted:#6b6b6b;--primary:#1a4a8c;--border:#d6d0c0;--border-soft:#e8e3d5}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"PingFang SC","Segoe UI",sans-serif}}
a{{color:var(--primary);text-decoration:none}}
a:hover{{text-decoration:underline}}
.wrap{{max-width:1120px;margin:0 auto;padding:0 16px}}
header.top{{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:10px 0 0}}
header.top a,.btn{{display:inline-flex;align-items:center;gap:6px;height:36px;padding:0 12px;border:1px solid var(--border);border-radius:8px;background:var(--bg-soft);color:var(--ink);white-space:nowrap;font-size:14px}}
header.top a:hover,.btn:hover{{border-color:var(--primary);color:var(--primary);text-decoration:none}}
header.top a small{{color:var(--ink-muted);font-size:12px}}
h1{{font:600 22px/1.3 ui-monospace,Menlo,Consolas,monospace;margin:22px 0 6px;overflow-wrap:anywhere}}
.from{{color:var(--ink-soft);margin:0 0 12px}}
.desc{{color:var(--ink-soft);margin:0 0 14px;max-width:80ch}}
.actions{{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 16px}}
.run{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;color:var(--ink-muted);align-self:center}}
.code{{display:flex;border:1px solid var(--border);border-radius:8px;background:#fafaf6;overflow:hidden;margin:0 0 28px}}
.code pre{{margin:0;font:12.5px/1.6 ui-monospace,Menlo,Consolas,monospace;padding:12px 0}}
.code .ln{{color:var(--ink-muted);text-align:right;padding:12px 10px 12px 12px;border-right:1px solid var(--border-soft);background:var(--bg-soft);user-select:none;flex:0 0 auto}}
.code .src{{flex:1;min-width:0;overflow-x:auto}}
.code .src code{{display:block;padding:0 14px;background:transparent}}
footer{{margin:0 0 28px;padding-top:14px;border-top:1px solid var(--border-soft);font-size:13px;color:var(--ink-muted)}}
footer a{{margin-right:14px}}
@media (max-width:640px){{h1{{font-size:18px}}.code pre{{font-size:12px}}}}
@media print{{header.top,.actions,footer{{display:none}}}}
</style>
</head>
<body>
<div class="wrap">
<header class="top">
  <a href="{short}/">⌂ 全部教程 <small>easyaioffer.github.io</small></a>
  <a href="{repo}">⭐ Star</a>
</header>

<h1>{name}</h1>
<p class="from">配套教程：<a href="../tutorials/{slug}.html">{cn}</a> · <a href="../tutorials/{slug}_en.html">EN</a></p>
<p class="desc">{desc}</p>
<div class="actions">
  <a class="btn" href="{repo}/blob/main/docs/tutorials/code/{name}">在 GitHub 查看</a>
  <a class="btn" href="{raw}/docs/tutorials/code/{name}">原始 .py</a>
  <button class="btn" id="copy" type="button">复制全部</button>
  <span class="run">python {name}</span>
</div>

<div class="code">
<pre class="ln">{lines}</pre>
<pre class="src"><code class="language-python" id="src">{source}</code></pre>
</div>

<footer>
  <a href="{short}/">ARIS in AI Offer</a>
  <a href="{repo}">GitHub</a>
  <a href="{repo}/blob/main/docs/tutorials/code/README.md">全部脚本说明</a>
</footer>
</div>
<script>
hljs.highlightAll();
document.getElementById('copy').addEventListener('click', function () {{
  var b = this;
  navigator.clipboard.writeText(document.getElementById('src').textContent).then(function () {{
    b.textContent = '已复制'; setTimeout(function () {{ b.textContent = '复制全部'; }}, 1500);
  }});
}});
</script>
</body>
</html>
"""


def docstring_summary(src: str) -> str:
    """First non-empty, non-underline line of the module docstring, else ''."""
    stripped = src.lstrip()
    if not stripped.startswith(('"""', "'''")):
        return ""
    body = stripped[3:].split(stripped[:3], 1)[0]
    for line in body.splitlines():
        line = line.strip()
        if line and not set(line) <= {"=", "-"}:
            return line
    return ""


def main() -> int:
    owner = {c: t for t in CATALOG for c in t["code"]}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for py in sorted(CODE_DIR.glob("*.py")):
        t = owner.get(py.name)
        if t is None:
            print(f"skip {py.name}: no tutorial lists it in build_index.CATALOG", file=sys.stderr)
            continue
        src = py.read_text(encoding="utf-8")
        count = src.count("\n") + (0 if src.endswith("\n") else 1)
        page = PAGE.format(
            name=e(py.name), desc=e(docstring_summary(src)), slug=e(t["slug"]), cn=e(t["cn"]),
            repo=REPO, short=SHORT, raw=RAW,
            lines="\n".join(str(i) for i in range(1, count + 1)),
            source=e(src.rstrip("\n")),
        )
        (OUT_DIR / f"{py.name}.html").write_text(page, encoding="utf-8")
        n += 1
    print(f"wrote {n} code pages -> {OUT_DIR.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
