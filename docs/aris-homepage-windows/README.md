# ARIS-Homepage — Windows Quick Start (Python-explicit)

> Community supplement to the [**ARIS-Homepage**](../../README.md#-aris-homepage--fact-checked-academic-homepage-from-cv) section of the main README. The main README's "Quick start" snippet assumes a Unix-style shell where `aris-homepage` resolves as a command; on Windows there is no such alias, and a couple of Windows-only pitfalls (`.py` file association, SSL verification) will trip up a first-time user. This folder gives the corrected, tested command sequence for Windows + PowerShell, plus two things a first-time user should know before comparing their output to the live demo.
>
> 中文版排错细节（PowerShell 逐条实测踩坑记录）见 [`docs/aris-homepage-windows-quickstart_CN.md`](../aris-homepage-windows-quickstart_CN.md)。本文件是精简版 quick start，供 PR 进主 README 或单独查阅。

## Quick start (Windows / PowerShell)

`aris-homepage` is **not** an installed command on Windows — there is no `aris-homepage.exe` and no shell alias set up by default. Every invocation must go through `python` explicitly, against the actual script filename `aris_homepage.py` (underscore, not hyphen):

```powershell
python tools\aris_homepage.py init --from-cv .\cv.pdf --out .\site
cd .\site
# Calling agent (Claude Code or another LLM agent) fills .aris-homepage/extraction.json
# per .aris-homepage\EXTRACTION_HANDOFF.md — this step is not a shell command.
python ..\tools\aris_homepage.py finalize
notepad profile.yml             # tweak editorial choices (any editor works)
python ..\tools\aris_homepage.py render --persona theory-minimal
```

Two Windows-specific gotchas that aren't obvious from the error output:

- **Never omit `python`.** Running `.\tools\aris_homepage.py ...` directly (relying on the `.py` file association) can silently do nothing on Windows — no output, no error, no exit code — because the association isn't reliably wired for synchronous console execution. Always prefix with `python`.
- **`python3` may not exist.** On stock Windows installs, `python3` often resolves to the Microsoft Store's App Execution Alias stub rather than a real interpreter, and fails with "command not found" (exit code 9009). Use `python`, not `python3`.
- **Avoid non-ASCII characters anywhere in the CV path** (e.g. a Chinese folder name). The Windows builds of `pdftotext` (used for `.pdf` → text extraction) mangle non-ASCII path components into `??`, causing `I/O Error: Couldn't open file`. Copy the CV into an ASCII-only path (e.g. `.\cv_input\my_cv.pdf`) before running `init` if your CV lives under a non-English folder name.

## Windows SSL certificate error during fact-check

`render` and `check` run a DBLP/arXiv fact-check pass automatically. On Windows, this step commonly fails with:

```
⚠ DBLP unreachable for '...': <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]
  certificate verify failed: unable to get local issuer certificate (_ssl.c:1000)>
```

This is a known Windows/Python.org issue, not a bug in `aris_homepage.py`: `urllib.request`'s default SSL context on Windows does not use the OS certificate store, so on some machines the bundled root CA list is stale or incomplete. It's non-fatal — the audit just falls back to `WARN` for every paper and `index.html` still renders — but fact-checking won't actually verify anything until it's fixed.

Fix:

```powershell
python -m pip install pip-system-certs
```

This patches Python's SSL context to use the Windows certificate store. Re-run `render`/`check` afterward; DBLP should become reachable and correctly-indexed papers will move from `WARN` to a verified ✅ `PASS`.

If you're on a conda/Anaconda environment rather than python.org's Python, install into that environment specifically:

```powershell
& "C:\path\to\your\conda\envs\<env-name>\python.exe" -m pip install pyyaml pip-system-certs
& "C:\path\to\your\conda\envs\<env-name>\python.exe" tools\aris_homepage.py render --persona theory-minimal
```

## Your first CV-only site will look sparser than the live demo

The [live demo](https://wanshuiyin.github.io/) shown in the main README's preview strip was built from a CV **plus** the maintainer's previous manual homepage as editorial reference — that's where its paper `[Paper]`/`[Code]`/`[arXiv]` links and thumbnail images come from. If you run the pipeline from a bare CV only, don't expect those to appear automatically. Concretely:

- **The tool never fetches or generates links.** It doesn't search the web for your papers, and DBLP/arXiv fact-checking only verifies that a title/venue/year combination is *correct* (plus a sanity check that award badges like "Best Paper" have a verifiable URL) — it does **not** verify author lists, and it never writes a discovered URL back into your files. `[Paper]` / `[Code]` / `[Slides]` links are read from exactly one place at render time — `profile.yml`'s `publications_meta.<bibkey>.links` (see `build_publications_section`) — a `url`/`eprint` field in `publications.bib` is **not** picked up for rendering, even though it's valid BibTeX.
- **The tool never generates or downloads thumbnails.** `publications_meta.<bibkey>.thumbnail` is a plain path/URL field you fill in by hand (e.g. point it at an image you exported and dropped into `site/assets/`). Same for `identity.photo` — a local path or remote URL you supply, not something the tool creates.
- **A CV alone typically doesn't list co-authors, arXiv IDs, or repo links per-paper** — most CVs just have "Title, Venue, Year." So a first `init → finalize → render` pass from a CV alone will correctly produce accurate text (name, bio, education, publication titles/venues/years) but usually **no** paper action links and **no** thumbnails, and often with `[ ]` items in `EXTRACTION_REVIEW.md` asking you to confirm co-authors it couldn't find.

To match the demo's richer look, add the missing links/images by hand in `profile.yml`'s `publications_meta` (`links` / `thumbnail` per bibkey), then re-run `render`. This is expected editorial work, not a failure of the pipeline — see [`skills/homepage-generator/PROFILE_SCHEMA.md`](../../skills/homepage-generator/PROFILE_SCHEMA.md) for the full field reference.

---

Full skill contract: [`skills/homepage-generator/SKILL.md`](../../skills/homepage-generator/SKILL.md). Implementation: [`tools/aris_homepage.py`](../../tools/aris_homepage.py).
