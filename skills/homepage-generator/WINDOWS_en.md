# ARIS-Homepage on Windows — zero-setup quick start

[`SKILL.md`](SKILL.md) is the full contract, written for people already comfortable with a command line. This is the supplement for someone on **Windows who has never installed Python or used a terminal** — the pitfalls below were all hit in a real debugging session.

> 中文版：[`WINDOWS.md`](WINDOWS.md)
>
> Applies to: Windows 10/11, an ordinary machine with no Python / Git / command-line tooling installed.

---

## 0. What you need

- A CV in `.pdf`, `.docx`, or `.txt`
- A terminal running Claude Code (or another LLM agent) — the CV → structured-data step **must** be done by an AI reading your CV and filling in a form. `aris_homepage.py` never calls an LLM itself — only "read the CV, turn it into structured JSON" is delegated to the agent; extracting the text, persisting the source files, fact-checking, and rendering are all the script's own work.

---

## 1. Install Python

Open PowerShell (search "PowerShell" in the Start menu):

```powershell
python --version
```

- If it prints a version, check it is **3.10 or newer** (`aris_homepage.py` requires Python 3.10+), then go to step 2.
- If it says "not found" or opens the Microsoft Store, install from [python.org/downloads](https://www.python.org/downloads/) and **tick "Add python.exe to PATH"** during setup, or every command below will fail.

> ⚠️ **Don't use `python3`.** On stock Windows, `python3` is usually the Microsoft Store's App Execution Alias — a stub that fails with "command not found" / exit code 9009 even when Python is installed. **Use `python`.**

---

## 2. Install dependencies

```powershell
python -m pip install pyyaml
```

`pyyaml` is required — `finalize` and `render` both stop with `pyyaml not installed` without it.

If DBLP lookups later fail with an SSL certificate error (`CERTIFICATE_VERIFY_FAILED`), install one more package. This is a known python.org-on-Windows issue, not a bug in the script:

```powershell
python -m pip install pip-system-certs
```

It makes Python verify HTTPS against the Windows certificate store, after which DBLP becomes reachable.

---

## 3. How to invoke the script

Once you are in the repo directory, **always use this form**:

```powershell
python .\tools\aris_homepage.py <subcommand> <args...>
```

❌ **Don't** double-click the file or drop the `python` prefix. On many Windows setups the `.py` file association isn't wired for synchronous console execution, so `.\tools\aris_homepage.py ...` produces no output, no error, and no exit code — it looks like a hang, but the command simply never ran.

❌ **Don't** use `python3` (see step 1).

❌ **Don't** type `aris-homepage`. Despite what the main README's quick start shows, there is no such command — the repo ships no installer, so the script is only ever reachable as `python <path>\aris_homepage.py`. Note the **underscore** in the filename, not a hyphen.

✅ Correct:

```powershell
python .\tools\aris_homepage.py init --from-cv .\cv_input\my_cv.pdf --out .\my_site
```

---

## 4. Choosing a CV format (and keeping the path ASCII-only)

What each format actually costs you on Windows:

| CV format | What else you must install |
|---|---|
| `.txt` | **Nothing** — the only format that works out of the box |
| `.docx` | `python -m pip install python-docx`. The script looks for macOS `textutil` first (absent on Windows), falls back to `python-docx`, and otherwise dies with `Cannot read .docx: install python-docx OR run on macOS (textutil).` |
| `.pdf` | poppler (which provides `pdftotext`), plus the ASCII-path caveat below |

**Saving your CV as `.txt` is the path of least resistance.** Advice you'll find elsewhere that "just use .docx" is macOS experience — on Windows `.docx` needs an extra package.

If you do use `.pdf`: **keep every folder in the path ASCII-only.** Some Windows builds of `pdftotext.exe` mangle non-ASCII arguments into `??` and then can't find the file. It doesn't matter whether the filename itself is English — any Chinese (or otherwise non-ASCII) folder name along the path can trigger it.

⚠️ **You won't see `pdftotext`'s own error message.** The script invokes it with `capture_output=True, check=True`, so its `I/O Error: Couldn't open file '.\CV??\...'` is swallowed and what reaches your screen is a Python traceback ending in:

```
subprocess.CalledProcessError: Command '['pdftotext', ...]' returned non-zero exit status 1.
```

So: **`CalledProcessError` + a `.pdf` CV → suspect a non-ASCII path first**, not a bug in the script. Copy the CV somewhere ASCII-only and re-run:

```powershell
mkdir cv_input
copy "..\CV合集\我的简历.pdf" ".\cv_input\my_cv.pdf"
python .\tools\aris_homepage.py init --from-cv ".\cv_input\my_cv.pdf" --out .\my_site
```

---

## 5. The full four-step run

All from the repo root, in PowerShell:

```powershell
# 1 — CV to plain text; writes the extraction template the agent will fill
python .\tools\aris_homepage.py init --from-cv ".\cv_input\my_cv.txt" --out .\my_site

# 2 — NOT a shell command. Point Claude Code (or your agent) at
#     my_site\.aris-homepage\EXTRACTION_HANDOFF.md for the instructions and
#     my_site\.aris-homepage\cv.txt for the CV text, and have it fill
#     my_site\.aris-homepage\extraction.json. Just ask it to "do the CV
#     extraction" — you never write that JSON by hand.

# 3 — persist extraction.json into editable source files
cd .\my_site
python ..\tools\aris_homepage.py finalize

# 4 — render (automatically checks publications against DBLP; a miss is only a
#     WARN, but a hard mismatch is BLOCKED and stops the render — see §6)
python ..\tools\aris_homepage.py render --persona theory-minimal
```

Afterwards `my_site\index.html` is a homepage you can drag straight into a browser, `audit-report.md` is the fact-check report, and `EXTRACTION_REVIEW.md` lists everything the AI was unsure about while reading your CV (co-author lists, vague dates) — **read that file before publishing.**

---

## 6. Where the demo's paper links and thumbnails come from

The [live demo](https://wanshuiyin.github.io/) in the main README was built from a CV **plus** the maintainer's previous manual homepage as editorial reference. Running from a bare CV will give you something sparser, by design:

- **The tool never fetches or generates paper links.** It doesn't search the web for your publications, and it never writes a URL it found back into your files. (The unrelated `init --from-repos` flag does fetch GitHub metadata, but it never touches publication entries.)
- **The tool never generates or downloads thumbnails**, and never creates or retouches your photo.
- **The fact-check only queries DBLP**, and only for the papers listed in `profile.yml`'s `selected_publications`. It searches by your title and compares year and venue; it does **not** check author lists. The so-called "arXiv fallback" doesn't hit the network at all — it just notices whether your BibTeX entry carries an `eprint` field or an `archiveprefix` of `arxiv`.
- **The fact-check result can stop the render — it is not just a mark in a report.** Three outcomes: everything verified is `PASS`; not found / unreachable / ambiguous is `WARN` and renders normally; but a missing bibkey, a year mismatch, a venue mismatch, or an award like Best Paper / Spotlight / Oral / Outstanding claimed without a verifiable link (only `arxiv`, `paper`, `pdf`, `project`, `openreview` count — a `code` or `slides` link does not) is a hard failure → verdict `BLOCKED` → **`render` exits without writing `index.html`.** Only `audit-report.md` appears. Fix what it lists and re-run, or force it through with `--override-all` (loudly recorded in the report).

`[Paper]` / `[arXiv]` / `[Code]` buttons and thumbnails come from exactly one place — hand-written fields in `profile.yml`:

```yaml
publications_meta:
  chen2024fum:
    links:
      paper: "https://arxiv.org/abs/xxxx.xxxxx"
      code: "https://github.com/<you>/<project>"
    thumbnail: "https://raw.githubusercontent.com/.../thumb.png"   # or a local path
```

A `url` or `eprint` field in `publications.bib` is perfectly valid BibTeX but is **never** read for rendering those buttons — only `publications_meta.<bibkey>.links` is. Your headshot (`identity.photo`) works the same way: a local path or a remote URL you supply, embedded as-is.

Most CVs list nothing but "Title, Venue, Year", so a first `init → finalize → render` pass typically produces accurate text with **no** paper buttons and **no** thumbnails. That is expected editorial work, not a failed pipeline: fill in `links` / `thumbnail` per bibkey and render again. Full field reference: [`PROFILE_SCHEMA.md`](PROFILE_SCHEMA.md).

---

## 7. Error reference

| Error / symptom | Cause | Fix |
|---|---|---|
| `.\tools\aris_homepage.py ...` does nothing at all | Missing `python` prefix; `.py` association not wired | Use `python .\tools\aris_homepage.py ...` |
| `python3` not found / opens the Store | On Windows `python3` is an alias stub | Use `python` |
| `aris-homepage` not found | No such command exists anywhere | `python <path>\aris_homepage.py` |
| Can't find `aris-homepage.py` | The filename uses an underscore | `aris_homepage.py` |
| `pyyaml not installed` | Dependency missing | `python -m pip install pyyaml` |
| Obscure runtime errors on a Python older than 3.10 | 3.10 is the declared floor; the script does no version check, so an old interpreter fails without saying why | Install Python 3.10+ |
| `subprocess.CalledProcessError ... pdftotext ... exit status 1` | `pdftotext` exited non-zero and its own message was swallowed; a non-ASCII path is the most common reason | Retry from an ASCII-only path; if it persists, run `pdftotext` by hand to see the real error |
| `Cannot read .pdf: install poppler-utils` | No `pdftotext` on this machine | Use a `.txt` CV, or install poppler for Windows and add it to PATH |
| `Cannot read .docx: install python-docx OR run on macOS (textutil)` | No `textutil` on Windows; `.docx` needs `python-docx` | `python -m pip install python-docx`, or switch to `.txt` |
| `CERTIFICATE_VERIFY_FAILED` during the DBLP check | Windows Python isn't using the OS certificate store | `python -m pip install pip-system-certs` |
| DBLP check reports `Remote end closed connection` | Transient rate-limiting | Re-run `render` / `check` |
| `render` prints `✗ Audit BLOCKED` and writes no `index.html` | A hard fact-check failure | Fix what `audit-report.md` lists, or `--override-all` |
| `extraction.json not found` | Step 2 (the AI extraction) hasn't happened yet | Have the agent fill `extraction.json` per `EXTRACTION_HANDOFF.md`, then `finalize` |

---

## 8. Using Anaconda / conda

Same commands — either `conda activate <env>` and use `python` as usual, or spell out the environment's interpreter.

⚠️ If you spell out the interpreter, remember **the script path is still relative to your current directory**. `render` must run from the site directory (the one holding `profile.yml`), where the script is one level up. (`finalize` is more forgiving — it accepts `--out <dir>` and can run from the repo root.)

```powershell
$py = "C:\ProgramData\anaconda3\envs\<env-name>\python.exe"

& $py -m pip install pyyaml
cd .\my_site
& $py ..\tools\aris_homepage.py render --persona theory-minimal
```

Running `& $py .\tools\aris_homepage.py render` from the repo root does not work — `render` treats the current directory as the site directory, and there is no `profile.yml` there.

---

Questions: the [ARIS main repo](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep) community group, or open an issue here.
