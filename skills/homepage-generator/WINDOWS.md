# ARIS-Homepage Windows 零基础上手指南

[`SKILL.md`](SKILL.md) 是完整契约文档，面向已经熟悉命令行的用户。这篇是给**第一次用 Windows 电脑、没装过 Python、没配过任何开发环境**的人看的补充版——把实际踩过的坑和对应的解决办法都记下来了。

> English: [`WINDOWS_en.md`](WINDOWS_en.md)
>
> 适用对象：Windows 10/11，只有一台普通电脑，没装过 Python / Git / 任何命令行工具。

---

## 0. 你需要准备的东西

- 一份 CV，格式 `.pdf` / `.docx` / `.txt` 都行
- 一个能跑 Claude Code（或其他 LLM agent）的终端 —— 因为 CV → 结构化信息这一步**必须由 AI 读一遍 CV 并填表**，`aris_homepage.py` 这个脚本本身不调用任何 LLM —— 只有「读懂 CV、填成结构化 JSON」这一步要交给 agent，其余的转文本、落地成源文件、核对、渲染都是脚本自己做

---

## 1. 装 Python（如果还没有）

打开 PowerShell（开始菜单搜"PowerShell"），输入：

```powershell
python --version
```

- 如果显示版本号，确认**不低于 3.10**（`aris_homepage.py` 要求 Python 3.10+），然后跳到第 2 步。低于 3.10 就按下一条去装个新的。
- 如果提示"找不到命令"或弹出 Microsoft Store，去 [python.org/downloads](https://www.python.org/downloads/) 下载安装包，**安装时一定要勾选 "Add python.exe to PATH"**，否则后面所有命令都会失败。

> ⚠️ **不要用 `python3`**：Windows 上 `python3` 这个名字经常被系统占用成一个空壳（Microsoft Store 的"应用执行别名"），装了 Python 也可能打不开，报 `找不到命令` / exit code 9009。**统一用 `python`。**

---

## 2. 装脚本需要的依赖

```powershell
python -m pip install pyyaml
```

`pyyaml` 是必须的 —— 没装的话 `finalize` 和 `render` 都会直接停在 `pyyaml not installed`。

如果之后 DBLP 论文核对总是报 SSL 证书错误（`CERTIFICATE_VERIFY_FAILED`），再补一个包（这是 Windows 上 Python.org 版本的通病，不是这个脚本的 bug）：

```powershell
python -m pip install pip-system-certs
```

装完这个包后，Python 会直接用 Windows 系统自带的证书库去验证 HTTPS，DBLP 就能连上了。

---

## 3. 命令怎么调用（这里最容易踩坑）

进入仓库目录后，**永远用这个格式**：

```powershell
python .\tools\aris_homepage.py <子命令> <参数...>
```

❌ **不要**直接双击或者省略 `python` 直接敲 `.\tools\aris_homepage.py ...` —— 在很多 Windows 环境下，`.py` 文件关联没配好，敲了完全没反应、没输出、也不报错，看起来像是"卡住了"，其实是命令根本没被执行。

❌ **不要**用 `python3`（原因见第 1 步）。

❌ **不要**敲 `aris-homepage` —— 这个命令不存在。仓库没有提供任何安装器，脚本永远只能用 `python <路径>\aris_homepage.py` 的方式调。旧文档或博客里如果看到这种写法，那是过时的。另外注意文件名是**下划线** `aris_homepage.py`，不是连字符。

✅ 正确示例：

```powershell
python .\tools\aris_homepage.py init --from-cv .\cv\my_cv.pdf --out .\my_site
```

---

## 4. 选哪种 CV 格式（用 `.pdf` 的话路径别有中文）

这是实测踩到的坑：**如果 CV 文件所在的文件夹名字含中文**（比如 `CV合集`），部分 Windows 版 `pdftotext.exe` 会把命令行参数里的中文乱码成 `??`，于是根本找不到文件。这跟文件名本身是不是英文无关——路径中任意一级文件夹名含中文都可能触发。

⚠️ **屏幕上看到的不是 `pdftotext` 那句人话报错。** 脚本是用 `capture_output=True, check=True` 调它的，`pdftotext` 自己那句

```
I/O Error: Couldn't open file '.\CV??\...': No such file or directory.
```

会被吞掉，你实际看到的是一条 Python traceback，末尾是：

```
subprocess.CalledProcessError: Command '['pdftotext', ...]' returned non-zero exit status 1.
```

所以**看到 `CalledProcessError` 又刚好用了 `.pdf`，先怀疑路径里有中文**，不用去找脚本的 bug。

**解决办法**：把 CV 复制到一个纯英文/数字路径下再跑，比如：

```powershell
mkdir cv_input
copy "..\CV合集\我的简历.pdf" ".\cv_input\my_cv.pdf"
python .\tools\aris_homepage.py init --from-cv ".\cv_input\my_cv.pdf" --out .\my_site
```

如果你的电脑本来就没装 `pdftotext`（大部分 Windows 电脑默认没有），`init` 处理 `.pdf` 时会直接报错提示装 poppler。这时候三种格式的真实代价是：

| CV 格式 | Windows 上还需要装什么 |
|---|---|
| `.txt` | **什么都不用装** ——唯一开箱即用的 |
| `.docx` | 要 `python -m pip install python-docx`。脚本会先找 macOS 的 `textutil`（Windows 上没有），再退回 `python-docx`，都没有就报 `Cannot read .docx: install python-docx OR run on macOS (textutil).` |
| `.pdf` | 要 poppler（提供 `pdftotext`），还得躲开上面的中文路径问题 |

**所以最省事的办法是把 CV 另存为 `.txt`。** 网上很多说法是"存成 docx 就行"——那是 macOS 的经验，Windows 上 `.docx` 还要多装一个包。

---

## 5. 完整跑一遍的四步命令

以下命令都在仓库根目录、用 PowerShell 执行：

```powershell
# 第 1 步：把 CV 转成纯文字，生成待填写的提取模板
python .\tools\aris_homepage.py init --from-cv ".\cv_input\my_cv.txt" --out .\my_site

# 第 2 步：这一步不是敲命令，而是让 Claude Code（或你在用的 AI agent）
#   读 my_site\.aris-homepage\EXTRACTION_HANDOFF.md 里的说明，
#   读 my_site\.aris-homepage\cv.txt 的 CV 正文，
#   按 schema 把信息填进 my_site\.aris-homepage\extraction.json
#   —— 直接跟 Claude Code 说"帮我完成 CV 提取"即可，不需要你自己写 JSON

# 第 3 步：把 extraction.json 落地成可编辑的源文件
cd .\my_site
python ..\tools\aris_homepage.py finalize

# 第 4 步：渲染网页（会自动查 DBLP 核对论文信息；查不到只是 WARN，不挡生成。
#          但年份/venue 对不上这类硬错会判 BLOCKED，直接不生成 —— 见第 6 节）
python ..\tools\aris_homepage.py render --persona theory-minimal
```

跑完后 `my_site\index.html` 就是可以直接拖进浏览器打开的个人主页，`audit-report.md` 是核对报告，`EXTRACTION_REVIEW.md` 列出了 AI 从 CV 里提取时不太确定、需要你人工确认的地方（比如合著者名单、模糊日期）——**发布前一定要看一眼这个文件**。

---

## 6. 主仓库 README 演示图里的论文链接 / 缩略图，是工具自动生成的吗？

**不是。** 主 README 里那个 [live demo](https://wanshuiyin.github.io/) 是拿 CV **加上**维护者原来那份手写主页当编辑参考做出来的；只喂一份 CV 得到的结果一定更朴素，这是设计如此。先说结论：

- **不会**自动去网上搜这篇论文，也**不会**自动生成 `[Paper]` `[Code]` `[Slides]` 这些链接。
- **不会**自动截图、抓取论文首页图、生成或下载任何缩略图。
- 核对那一步（`check` / `render` 时自动跑）**只查 DBLP**，而且只查你在 `profile.yml` 的 `selected_publications` 里列出来的那几篇：用标题去搜，比对年份和 venue；**不核对作者名单**，也**不会**把查到的链接写回你的文件。所谓 "arXiv fallback" 不联网，只是看你 bib 里有没有 `eprint` 或 `archiveprefix = arxiv`。
- **核对结果会挡生成，不只是报告上的一个标记。** 三档：全过是 `PASS`，查不到 / 连不上 / 有歧义算 `WARN`（照常生成），而 bibkey 不存在、年份对不上、venue 对不上、声称了 Best Paper / Spotlight / Oral / Outstanding 之类奖项却没填可核实链接（只认 `arxiv` `paper` `pdf` `project` `openreview` 这五个字段，填 `code` 或 `slides` 不算），都算硬错 → verdict `BLOCKED` → **`render` 直接退出，`index.html` 根本不写**。这时候只有 `audit-report.md` 会生成，照着它改完再跑，或者用 `--override-all` 强推（会在报告里留痕）。

README 演示图里那些论文卡片上的 `[Paper]` `[arXiv]` `[Code]` 链接和右侧缩略图，全部来自 `profile.yml` 里手写的字段——具体是 `publications_meta.<bibkey>.links` 和 `publications_meta.<bibkey>.thumbnail`，例如：

```yaml
publications_meta:
  chen2024fum:
    links:
      paper: "https://arxiv.org/abs/xxxx.xxxxx"
      code: "https://github.com/你的用户名/项目名"
    thumbnail: "https://raw.githubusercontent.com/.../thumb.png"   # 或本地路径
```

也就是说：

1. **链接**：渲染代码（`build_publications_section`）只读 `profile.yml` 里 `publications_meta.<bibkey>.links` 这一个字段来生成 `[Paper]` `[Code]` 按钮，`publications.bib` 里就算写了 `url`/`eprint` 字段也**不会**被拿来渲染成按钮。如果你（或帮你做提取的 AI）在 CV 或聊天里没提供论文的 arXiv/项目主页/代码仓库地址，`publications_meta` 里就不会有 `links`，渲染出来的论文条目下面就只有标题+作者+venue，没有任何按钮——这是正常现象，不是渲染失败。想要这些按钮，必须自己把每篇论文的真实链接手动写进 `profile.yml` 的 `publications_meta.<bibkey>.links`。
2. **缩略图**：同理，`thumbnail` 字段完全靠手动填一个图片 URL 或本地路径（比如把配图放进 `my_site/assets/` 再写相对路径）。工具不会替你去论文里"抠"一张图出来。
3. 头像（`identity.photo`）也是同样逻辑——本地路径或远程 URL，工具只负责把它嵌进 HTML，不会帮你生成或美化。

所以对第一次用这个工具的人，合理预期是：**跑完 `init → finalize → render` 拿到的是一个"文字齐全、但论文卡片下面大概率没有按钮和配图"的初版主页**。如果想要 README 演示图那种效果，需要在 `profile.yml` 的 `publications_meta` 里手动把每篇论文的链接（`links`）和缩略图路径（`thumbnail`）补上，再重新 `render` 一次。

---

## 7. 常见报错对照表

| 报错 / 现象 | 原因 | 解决 |
|---|---|---|
| 敲了 `.\tools\aris_homepage.py ...` 完全没反应 | 没加 `python` 前缀，`.py` 文件关联没生效 | 改成 `python .\tools\aris_homepage.py ...` |
| `python3` 提示找不到命令 / 弹 Store | Windows 上 `python3` 是空壳别名 | 统一用 `python` |
| `aris-homepage` 提示不是命令 | 这个命令在任何平台都不存在，仓库没有安装器 | `python <路径>\aris_homepage.py` |
| `找不到文件 'aris-homepage.py'` | 文件名拼错了——实际文件名是下划线 `aris_homepage.py`，不是连字符 | 用下划线 |
| `pyyaml not installed` | 依赖没装 | `python -m pip install pyyaml` |
| 用 `.pdf` 时抛 `subprocess.CalledProcessError ... pdftotext ... exit status 1` | `pdftotext` 以非零退出（它自己的报错被脚本吞了，所以看不到原因）。路径含中文是最常见的一种 | 先把 CV 挪到纯英文路径下重试；还不行就手动跑一遍 `pdftotext` 看它到底说什么 |
| `Cannot read .pdf: install poppler-utils` | 电脑上没装 `pdftotext` | 换成 `.txt` 格式的 CV（最省事），或自行安装 poppler for Windows 并加入 PATH |
| `Cannot read .docx: install python-docx OR run on macOS (textutil)` | Windows 上没有 `textutil`，读 `.docx` 要 `python-docx` | `python -m pip install python-docx`，或干脆换 `.txt` |
| `render` 打印 `✗ Audit BLOCKED`，没有 `index.html` | 论文核对出现硬错（年份/venue 不符、bibkey 缺失、奖项没链接） | 照 `audit-report.md` 改 `profile.yml` / `publications.bib`，或 `--override-all` 强推 |
| 各种莫名其妙的运行时报错，且 Python 版本低于 3.10 | 仓库声明的最低版本是 3.10（脚本没有在启动时检查，所以旧版不会给你一句干脆的提示） | 装 Python 3.10 以上 |
| DBLP 核对报 `CERTIFICATE_VERIFY_FAILED` | Windows 版 Python 没用系统证书库 | `python -m pip install pip-system-certs` |
| DBLP 核对报 `Remote end closed connection` | 短时间内请求太多被限速，多为偶发 | 重跑一次 `render` 或 `check` 一般就好 |
| `extraction.json not found` | 还没做第 2 步（AI 提取） | 先让 Claude Code 读 `EXTRACTION_HANDOFF.md` 填好 `extraction.json`，再 `finalize` |

---

## 8. 用了 Anaconda / conda 环境怎么办

如果你的电脑装的是 Anaconda，而不是 python.org 的原生 Python，用法完全一样，只是把 `python` 换成对应环境的完整路径，或者先 `conda activate <环境名>` 再照常用 `python`。

⚠️ 换成完整路径之后**别忘了脚本路径是跟着当前目录走的**：`render` 必须在站点目录（有 `profile.yml` 那个）里跑，此时脚本在上一级，要写 `..\tools\aris_homepage.py`。（`finalize` 宽松一些，它支持 `--out <目录>`，可以待在仓库根目录跑。）

```powershell
$py = "C:\ProgramData\anaconda3\envs\<你的环境名>\python.exe"

& $py -m pip install pyyaml
cd .\my_site
& $py ..\tools\aris_homepage.py render --persona theory-minimal
```

（在仓库根目录直接 `& $py .\tools\aris_homepage.py render` 是跑不通的 —— `render` 把当前目录当站点目录，根目录下没有 `profile.yml`。）

---

有问题去 [ARIS 主仓库](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep) 找社区群，或者在本仓库开 issue。
