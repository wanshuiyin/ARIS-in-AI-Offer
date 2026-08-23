#!/usr/bin/env python3
"""aris_homepage.py — generate fact-checked academic homepages from a CV.

See skills/homepage-generator/SKILL.md for the full contract.

Pipeline:
    init --from-cv  →  textutil → emit extraction handoff JSON → user's calling LLM
                       fills .aris-homepage/extraction.json → this script persists
                       to profile.yml + publications.bib + bio.md + news.md
    render          →  load → fact-check (DBLP) → Python builds HTML chunks
                       → injects into homepage-<persona>.html template
    check           →  fact-check only, update audit-report.md
    doctor          →  environment diagnostic

Round-2 cross-model design (Codex GPT-5.5 xhigh + Gemini auto-gemini-3) converged on:
  - macOS textutil for .docx; python-docx as optional fallback
  - LLM extraction by calling agent (not here in Python); persistence handled by this script
  - JSON-schema-constrained extraction output
  - Idempotency: bail by default if profile.yml exists
  - DBLP direct API call (no third-party lib)
  - Two-layer override: per-paper YAML + --override-all CLI
  - Template approach: Python builds per-section HTML chunks; template is shell

External deps: pyyaml only (BibTeX is parsed via the stdlib parser in this file).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html as html_lib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1
DBLP_API = "https://dblp.org/search/publ/api"


def short_path(p: Path, from_dir: Path | None = None) -> str:
    """Whichever of the relative / absolute form of ``p`` is shorter to read.

    A relative path is friendlier from inside the repo, but climbing out of a
    site workspace on another branch of the filesystem produces ../../../..
    noise that is worse than just printing the absolute path.
    """
    rel = os.path.relpath(p, from_dir or Path.cwd())
    return rel if len(rel) < len(str(p)) else str(p)


def invocation(from_dir: Path | None = None) -> str:
    """How to re-run this script from ``from_dir`` (default: the current directory).

    There is no installed ``aris-homepage`` command — this repo ships no entry
    point — so every "run this next" message has to spell out the interpreter
    and a path valid from where the user will actually be standing. That is not
    always where they are now: ``render`` and ``check`` operate on the site
    workspace, so their commands must be written relative to it.
    """
    return f"python {short_path(Path(__file__).resolve(), from_dir)}"


DEFAULT_FILES = {
    "profile": "profile.yml",
    "bib": "publications.bib",
    "bio": "bio.md",
    "news": "news.md",
    "extraction_review": "EXTRACTION_REVIEW.md",
    "audit_report": "audit-report.md",
    "output_html": "index.html",
}

PERSONAS = ("theory-minimal", "active-researcher")
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# Friendly labels for social/scholar links rendered in the masthead.
# (Email is excluded — it's rendered separately as a mailto: line above the strip.)
LINK_LABELS = {
    "google_scholar": "Scholar",
    "semantic_scholar": "Semantic Scholar",
    "dblp": "DBLP",
    "github": "GitHub",
    "twitter": "Twitter",
    "bluesky": "Bluesky",
    "linkedin": "LinkedIn",
    "orcid": "ORCID",
    "homepage": "Homepage",
    "cv": "CV",
}

# DBLP local cache — avoids re-querying same titles + rate-limit hell
DBLP_CACHE_DIR = ".aris-homepage"
DBLP_CACHE_FILE = "dblp-cache.json"


# ---------------------------------------------------------------------------
# Stdlib BibTeX parser
# ---------------------------------------------------------------------------
# Handles the BibTeX subset that real-world ML papers actually use:
#   @inproceedings{key, author = {...}, title = {...}, booktitle = "...", year = 2024}
# Brace nesting, quoted values, and common LaTeX escapes (\&, \"o) are supported.
# Does NOT support: @string macros, @preamble, @comment with embedded entries,
# crossref resolution, full LaTeX-to-Unicode normalization.

_RE_ENTRY_HEADER = re.compile(r"@\s*([A-Za-z]+)\s*\{\s*([^,\s]+)\s*,")
_LATEX_ESCAPES = {
    r"\&": "&", r"\%": "%", r"\$": "$", r"\#": "#", r"\_": "_",
    r"\{": "{", r"\}": "}", r"\textasciitilde": "~",
}


def parse_bibtex_str(text: str) -> dict[str, dict[str, str]]:
    """Parse BibTeX text → {bibkey: {field: value, ..., 'ENTRYTYPE': type}}.

    Designed to be forgiving — silently skips malformed entries rather than crashing.
    """
    entries: dict[str, dict[str, str]] = {}
    i = 0
    n = len(text)
    while i < n:
        m = _RE_ENTRY_HEADER.search(text, i)
        if not m:
            break
        entry_type = m.group(1).lower()
        key = m.group(2).strip()
        i = m.end()
        # Parse fields until matching closing brace of the entry.
        entry: dict[str, str] = {"ENTRYTYPE": entry_type}
        depth = 1
        while i < n and depth > 0:
            # Skip whitespace + commas
            while i < n and text[i] in " \t\n\r,":
                i += 1
            if i >= n:
                break
            if text[i] == "}":
                depth -= 1
                i += 1
                continue
            # Parse field name
            fm = re.match(r"([A-Za-z][A-Za-z0-9_-]*)\s*=\s*", text[i:])
            if not fm:
                # Recover: skip to next entry boundary
                i = text.find("\n", i) + 1 if "\n" in text[i:] else n
                continue
            fname = fm.group(1).lower()
            i += fm.end()
            # Parse field value: brace-balanced, quoted, or bare number
            if text[i] == "{":
                value, i = _read_braced(text, i)
            elif text[i] == '"':
                value, i = _read_quoted(text, i)
            else:
                vm = re.match(r"[A-Za-z0-9._:/-]+", text[i:])
                value = vm.group(0) if vm else ""
                i += len(value)
            entry[fname] = _clean_bib_value(value)
        if key:
            entries[key] = entry
    return entries


def _read_braced(text: str, i: int) -> tuple[str, int]:
    """Read a brace-delimited value starting at text[i] == '{'. Returns (content, new_i)."""
    assert text[i] == "{"
    depth = 1
    i += 1
    start = i
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "\\" and i + 1 < len(text):
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i], i + 1
        i += 1
    return text[start:i], i


def _read_quoted(text: str, i: int) -> tuple[str, int]:
    """Read a quoted value starting at text[i] == '"'. Returns (content, new_i)."""
    assert text[i] == '"'
    i += 1
    start = i
    depth = 0  # supports embedded {...}
    while i < len(text):
        c = text[i]
        if c == "\\" and i + 1 < len(text):
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == '"' and depth == 0:
            return text[start:i], i + 1
        i += 1
    return text[start:i], i


def _clean_bib_value(s: str) -> str:
    """Strip leading/trailing whitespace, collapse internal whitespace, apply LaTeX escapes."""
    s = s.strip()
    for k, v in _LATEX_ESCAPES.items():
        s = s.replace(k, v)
    # Collapse internal newlines + spaces
    s = re.sub(r"\s+", " ", s)
    # Strip outer braces (common protection brace: {Title})
    while s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    return s


def split_authors(author_field: str) -> list[str]:
    """Split BibTeX author field on ' and '. Convert 'Last, First' → 'First Last'."""
    if not author_field:
        return []
    raw = re.split(r"\s+and\s+", author_field)
    normalized = []
    for a in raw:
        a = a.strip()
        if "," in a:
            parts = [p.strip() for p in a.split(",", 1)]
            if len(parts) == 2 and parts[1]:
                a = f"{parts[1]} {parts[0]}"
        normalized.append(a)
    return normalized


# ---------------------------------------------------------------------------
# Minimal Markdown subset for bio.md and news.md
# ---------------------------------------------------------------------------

def md_inline(s: str) -> str:
    """Escape HTML then apply minimal inline Markdown: links, bold, italic, code.

    Allows raw <a ...>...</a> and <img ...> tags to pass through unescaped — needed for
    embedded badges (e.g. star-history SVG inside a news bullet)."""
    stash: list[str] = []
    def _stash(m):
        stash.append(m.group(0))
        return f"§MD{len(stash)-1}§"
    # Stash <a>-with-content first (may wrap an <img>); then bare <img>.
    s = re.sub(r"<a\s[^>]*?>.*?</a>", _stash, s, flags=re.DOTALL)
    s = re.sub(r"<img\s[^>]*?/?>", _stash, s)
    s = html_lib.escape(s, quote=False)
    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)",
               r'<a href="\2" rel="noopener">\1</a>', s)
    s = re.sub(r"\*\*([^*\n]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", s)
    s = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", s)
    for i, tag in enumerate(stash):
        s = s.replace(f"§MD{i}§", tag)
    return s


def md_to_paragraphs(text: str) -> str:
    """Render bio-style Markdown to <p> blocks. Blank line = paragraph break."""
    out = []
    for para in re.split(r"\n\s*\n", text.strip()):
        if para.strip():
            out.append(f"<p>{md_inline(para.strip())}</p>")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Command: init --from-cv
# ---------------------------------------------------------------------------

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "profile": {
            "type": "object",
            "properties": {
                "identity": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "name_native": {"type": ["string", "null"]},
                        "title": {"type": ["string", "null"]},
                        "email": {"type": ["string", "null"]},
                    },
                    "required": ["name"],
                },
                "affiliations": {
                    "type": "object",
                    "properties": {
                        "current": {"type": "array", "items": {"type": "object"}},
                        "past": {"type": "array", "items": {"type": "object"}},
                    },
                },
                "education": {"type": "array", "items": {"type": "object"}},
                "research": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "interests": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "links": {"type": "object"},
                "awards": {"type": "array", "items": {"type": "object"}},
                "talks": {"type": "array", "items": {"type": "object"}},
                "teaching": {"type": "array", "items": {"type": "object"}},
                "selected_publications": {"type": "array", "items": {"type": "string"}},
                "publications_meta": {"type": "object"},
            },
            "required": ["identity", "research"],
        },
        "bibtex_entries": {"type": "string",
                           "description": "Full text of publications.bib (BibTeX format)."},
        "bio_md": {"type": "string", "description": "1-3 paragraph bio in Markdown."},
        "news_md": {"type": "string",
                    "description": "Reverse-chrono bullets: '- 2026-05: Did X'."},
        "uncertain": {"type": "array", "items": {"type": "string"},
                      "description": "Claims the agent could not verify from the CV."},
    },
    "required": ["profile", "bibtex_entries"],
}


def cmd_init(args: argparse.Namespace) -> int:
    """Bootstrap workspace by extracting a CV into structured editable files."""
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cv_path = Path(args.from_cv).resolve()
    if not cv_path.exists():
        die(f"CV file not found: {cv_path}")
    if (out_dir / DEFAULT_FILES["profile"]).exists():
        if args.force:
            backup_existing(out_dir)
        elif args.merge:
            die("--merge is not yet implemented (v1.1).")
        else:
            die(f"{DEFAULT_FILES['profile']} already exists at {out_dir}. "
                f"Use --force (backup .bak-TIMESTAMP) or --merge (v1.1).")

    cv_txt = extract_text_from_cv(cv_path)
    txt_path = out_dir / ".aris-homepage" / "cv.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(cv_txt, encoding="utf-8")

    # Optional: fetch GitHub repo snapshots for additional context (v1.1).
    # See issue #2: user wants to merge repo timelines into homepage news + projects.
    repos_path = None
    if getattr(args, "from_repos", None):
        repo_specs = [r.strip() for r in args.from_repos.split(",") if r.strip()]
        repos_data = fetch_github_repos(repo_specs,
                                         include_private=getattr(args, "include_private", False))
        if repos_data:
            repos_path = out_dir / ".aris-homepage" / "github_repos.json"
            repos_path.write_text(json.dumps(repos_data, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
            print(f"✓ {len(repos_data)} GitHub repo snapshots → {repos_path}")

    print_extraction_handoff(txt_path, out_dir, repos_path=repos_path)
    return 0


def fetch_github_repos(repos: list[str], *, include_private: bool = False) -> list[dict]:
    """Fetch repo snapshot via `gh` CLI for each `owner/repo` spec.

    Returns a list of dicts (one per successfully fetched repo) with shape:
        {repo, snapshot_at, url, homepage_url, description, stars, forks,
         primary_language, topics, is_archived, created_at, pushed_at,
         releases: [{tag, name, date, url, description}], latest_commit,
         readme_excerpt}

    Private repos are skipped by default — pass include_private=True to override.
    """
    if not shutil.which("gh"):
        die("`gh` CLI not installed. Install via `brew install gh` or see https://cli.github.com/")
    auth = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
    if auth.returncode != 0:
        die("`gh` CLI not authenticated. Run: `gh auth login`")

    gql = """query($owner:String!, $name:String!) {
        repository(owner:$owner, name:$name) {
            nameWithOwner description url homepageUrl
            isPrivate isArchived
            stargazerCount forkCount
            createdAt pushedAt
            primaryLanguage { name }
            repositoryTopics(first: 10) { nodes { topic { name } } }
            releases(first: 8, orderBy: {field: CREATED_AT, direction: DESC}) {
                nodes { name tagName publishedAt url description }
            }
            defaultBranchRef {
                name
                target {
                    ... on Commit {
                        latest: history(first: 1) {
                            nodes { committedDate messageHeadline url oid }
                        }
                    }
                }
            }
        }
    }"""
    results: list[dict] = []
    for spec in repos:
        if "/" not in spec:
            print(f"  ⚠ invalid repo spec '{spec}' (need owner/repo); skipping", file=sys.stderr)
            continue
        owner, name = spec.split("/", 1)
        try:
            r = subprocess.run(
                ["gh", "api", "graphql",
                 "-f", f"owner={owner}", "-f", f"name={name}",
                 "-f", f"query={gql}"],
                capture_output=True, text=True, check=True,
            )
            data = json.loads(r.stdout).get("data", {}).get("repository")
        except subprocess.CalledProcessError as e:
            print(f"  ⚠ gh api failed for {spec}: {(e.stderr or '')[:200]}", file=sys.stderr)
            continue
        except json.JSONDecodeError:
            print(f"  ⚠ malformed JSON from gh for {spec}", file=sys.stderr)
            continue
        if not data:
            print(f"  ⚠ repo not found or no access: {spec}", file=sys.stderr)
            continue
        if data.get("isPrivate") and not include_private:
            print(f"  ⚠ skipping PRIVATE repo {spec} (use --include-private to override)",
                  file=sys.stderr)
            continue

        # README — separate REST call, truncated to 20KB
        readme_text = ""
        try:
            rm = subprocess.run(
                ["gh", "api", f"/repos/{owner}/{name}/readme",
                 "-H", "Accept: application/vnd.github.raw"],
                capture_output=True, text=True, check=False,
            )
            if rm.returncode == 0:
                readme_text = rm.stdout[:20000]
        except Exception:
            pass

        topics = [n["topic"]["name"]
                  for n in (data.get("repositoryTopics") or {}).get("nodes", [])]
        commits = (((data.get("defaultBranchRef") or {})
                    .get("target") or {})
                   .get("latest", {}).get("nodes", []))
        latest_commit = commits[0] if commits else None

        snap = {
            "repo": data["nameWithOwner"],
            "snapshot_at": datetime.now(timezone.utc).isoformat(),
            "url": data["url"],
            "homepage_url": data.get("homepageUrl"),
            "description": data.get("description"),
            "stars": data.get("stargazerCount", 0),
            "forks": data.get("forkCount", 0),
            "primary_language": (data.get("primaryLanguage") or {}).get("name"),
            "topics": topics,
            "is_archived": data.get("isArchived", False),
            "is_private": data.get("isPrivate", False),
            "created_at": data.get("createdAt"),
            "pushed_at": data.get("pushedAt"),
            "releases": [
                {
                    "tag": rel["tagName"],
                    "name": rel.get("name"),
                    "date": rel["publishedAt"],
                    "url": rel["url"],
                    "description": (rel.get("description") or "")[:500],
                }
                for rel in (data.get("releases") or {}).get("nodes", [])[:8]
            ],
            "latest_commit": (
                {
                    "date": latest_commit["committedDate"],
                    "message": latest_commit["messageHeadline"],
                    "url": latest_commit["url"],
                    "sha": latest_commit["oid"][:7],
                } if latest_commit else None
            ),
            "readme_excerpt": readme_text,
        }
        results.append(snap)
        rel_count = len(snap["releases"])
        print(f"  ✓ {spec} — {snap['stars']} ★ · {rel_count} releases · "
              f"{snap['primary_language'] or '—'}")
    return results


def extract_text_from_cv(path: Path) -> str:
    """Convert CV → plain text. macOS textutil → python-docx → pdftotext fallback chain."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8")
    if suffix == ".docx":
        if shutil.which("textutil"):
            r = subprocess.run(["textutil", "-convert", "txt", "-stdout", str(path)],
                               capture_output=True, text=True, check=True)
            return r.stdout
        try:
            import docx  # type: ignore
            doc = docx.Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            die("Cannot read .docx: install python-docx OR run on macOS (textutil).")
    if suffix == ".pdf":
        if shutil.which("pdftotext"):
            r = subprocess.run(["pdftotext", str(path), "-"],
                               capture_output=True, text=True, check=True)
            return r.stdout
        die("Cannot read .pdf: install poppler-utils (provides pdftotext).")
    die(f"Unsupported CV format: {suffix}. Use .txt, .docx, or .pdf.")


def print_extraction_handoff(txt_path: Path, out_dir: Path,
                              repos_path: Path | None = None) -> None:
    """Emit instructions for the calling LLM agent to do extraction.

    This script does NOT call an LLM — the calling agent (Claude/Gemini)
    reads cv.txt + optional github_repos.json, fills the JSON-schema-constrained
    output, and writes .aris-homepage/extraction.json. A follow-up
    ``<this script> finalize`` ingests that JSON and writes the editable sources.
    """
    handoff_path = out_dir / ".aris-homepage" / "EXTRACTION_HANDOFF.md"
    schema_path = out_dir / ".aris-homepage" / "extraction.schema.json"
    schema_path.write_text(json.dumps(EXTRACTION_SCHEMA, indent=2), encoding="utf-8")

    repos_block = ""
    if repos_path is not None and repos_path.exists():
        repos_block = f"""

## Optional source: GitHub repo snapshots (v1.1)

A snapshot of user-selected GitHub repos has been written to:
  {repos_path.relative_to(out_dir)}

It contains per-repo: description / stars / forks / topics / primary_language /
created_at / pushed_at / up to 8 releases / latest commit / README excerpt
(truncated 20KB).

### How to use github_repos.json in extraction

- Surface each repo as a `featured_projects[]` entry (or extend an existing one)
  with a new `github:` subobject carrying the snapshot data (stars, forks,
  primary_language, topics, created_at, pushed_at, releases summary, latest_commit).
- Merge each repo's **timeline** into `news_md` with these rules:
  * Take at most 2 events per repo (releases highest priority, then created_at).
  * Cap total github-derived news at 6 across all repos.
  * If a CV news entry already mentions a release / project on the same
    date OR same YYYY-MM with overlapping title — DEDUP: keep the CV's
    human phrasing and append the release URL; do not list both.
  * stars/forks go to project stats, NOT into news.
- Use README excerpt to verify the project description matches the CV's claim
  (catches misrepresentation).
- Mark anything you cannot confirm from `github_repos.json` as `uncertain[]`.
"""

    handoff_path.write_text(f"""# ARIS Homepage — Extraction Handoff

The CV has been converted to plain text at:
  {txt_path.relative_to(out_dir)}
{repos_block}
## Next step (LLM agent task)

Read the CV text{' + the github_repos.json snapshot' if repos_block else ''} and
emit JSON conforming to the schema at:
  {schema_path.relative_to(out_dir)}

Write the JSON output to:
  .aris-homepage/extraction.json

Then run:
  cd {short_path(out_dir)}
  {invocation(out_dir)} finalize

…to persist the structured fields into profile.yml + publications.bib + bio.md + news.md.

## Extraction prompt template (use as a starting point)

> You are extracting structured data from an academic CV for a homepage generator.
> Read the CV text from {txt_path.name}. Emit valid JSON matching extraction.schema.json.
>
> **Rules:**
> - Use field names from the schema verbatim (no synonyms).
> - For each publication, generate a unique bibkey of the form `lastname{{year}}{{keyword}}`
>   (e.g., yang2024fewshot). Bibkeys MUST be valid BibTeX keys (no spaces, no special chars).
> - In bibtex_entries, emit ONE full BibTeX entry per paper, comma-separated within each entry.
> - Bio prose goes in bio_md (Markdown). Keep it 1-3 paragraphs, third-person OR first-person.
> - News goes in news_md as bullets prefixed with date: `- 2026-05: Joined NUS as intern`.
> - For uncertain claims (e.g., guessed dates, inferred awards), add them to `uncertain[]`
>   with a note. These will become checklist items in EXTRACTION_REVIEW.md.
> - Do NOT invent claims. If the CV doesn't say something, leave the field null/empty.
> - Identify the CV owner. Their name goes in profile.identity.name + name_native (if bilingual).
{('> - If github_repos.json exists, merge its timeline into news_md following the rules in the section above.' + chr(10)) if repos_block else ''}>
> After writing the JSON, instruct the user to `cd` into `{short_path(out_dir)}` and run `{invocation(out_dir)} finalize`.
""", encoding="utf-8")
    print(f"✓ CV text extracted to: {txt_path}")
    print(f"✓ Extraction handoff written to: {handoff_path}")
    print()
    print("Next: read the handoff doc and fill .aris-homepage/extraction.json, then:")
    print(f"  cd {short_path(out_dir)}")
    print(f"  {invocation(out_dir)} finalize")


def cmd_finalize(args: argparse.Namespace) -> int:
    """Ingest .aris-homepage/extraction.json → profile.yml + publications.bib + bio.md + news.md."""
    workspace = Path(args.out).resolve() if args.out else Path(".").resolve()
    extraction_path = workspace / ".aris-homepage" / "extraction.json"
    if not extraction_path.exists():
        die(f"{extraction_path} not found. Run `{invocation()} init --from-cv ...` first, "
            f"then have the calling agent fill the extraction JSON.")

    data = json.loads(extraction_path.read_text(encoding="utf-8"))
    profile = data.get("profile", {})
    bib_text = data.get("bibtex_entries", "")
    bio_md = data.get("bio_md", "")
    news_md = data.get("news_md", "")
    uncertain = data.get("uncertain", [])

    # Persist
    if not HAS_YAML:
        die("pyyaml not installed. Run: pip install pyyaml --break-system-packages "
            "(or use a venv).")
    profile["schema_version"] = SCHEMA_VERSION
    (workspace / DEFAULT_FILES["profile"]).write_text(
        yaml.dump(profile, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (workspace / DEFAULT_FILES["bib"]).write_text(bib_text, encoding="utf-8")
    if bio_md:
        (workspace / DEFAULT_FILES["bio"]).write_text(bio_md, encoding="utf-8")
    if news_md:
        (workspace / DEFAULT_FILES["news"]).write_text(news_md, encoding="utf-8")

    # Extraction review
    review_lines = ["# Extraction Review", "",
                    "These claims were extracted from your CV but could not be auto-verified.",
                    "Edit the corresponding files (profile.yml, publications.bib, bio.md, news.md)",
                    f"as needed, then run `{invocation(workspace)} render --persona theory-minimal`",
                    "from inside this directory.", ""]
    if uncertain:
        review_lines.append("## ⚠️ Uncertain claims (review before rendering)")
        review_lines.append("")
        for u in uncertain:
            review_lines.append(f"- [ ] {u}")
    else:
        review_lines.append("✓ No uncertain claims flagged by extraction.")
    (workspace / DEFAULT_FILES["extraction_review"]).write_text(
        "\n".join(review_lines) + "\n", encoding="utf-8")

    print(f"✓ Wrote {DEFAULT_FILES['profile']}")
    print(f"✓ Wrote {DEFAULT_FILES['bib']}")
    if bio_md:
        print(f"✓ Wrote {DEFAULT_FILES['bio']}")
    if news_md:
        print(f"✓ Wrote {DEFAULT_FILES['news']}")
    print(f"✓ Wrote {DEFAULT_FILES['extraction_review']}")
    print()
    print("Edit the files as needed, then render from the site directory:")
    print(f"  cd {short_path(workspace)}")
    print(f"  {invocation(workspace)} render --persona theory-minimal")
    return 0


# ---------------------------------------------------------------------------
# Command: render
# ---------------------------------------------------------------------------

def cmd_render(args: argparse.Namespace) -> int:
    """Render homepage HTML + audit-report.md from profile.yml + publications.bib."""
    if args.persona not in PERSONAS:
        die(f"Unknown persona: {args.persona}. Choose from {PERSONAS}.")
    if args.persona == "active-researcher":
        die("active-researcher template ships in v1.1. v1 only supports theory-minimal.")

    workspace = Path(".").resolve()
    profile = load_profile(workspace / DEFAULT_FILES["profile"])
    bib = parse_bibtex(workspace / DEFAULT_FILES["bib"])
    bio_md = read_optional(workspace / DEFAULT_FILES["bio"])
    news_md = read_optional(workspace / DEFAULT_FILES["news"])

    # Fact-check
    audit = None
    if not args.no_audit:
        audit = run_fact_check(profile, bib, override_all=args.override_all, workspace=workspace)
        write_audit_report(workspace / DEFAULT_FILES["audit_report"], audit, profile, args.persona)
        if audit.verdict == "BLOCKED" and not args.override_all:
            print(f"\n✗ Audit BLOCKED. See {DEFAULT_FILES['audit_report']}.", file=sys.stderr)
            print("  Fix the failing claims OR re-run with --override-all (logged loudly).",
                  file=sys.stderr)
            return 2

    # Render
    chunks = build_section_chunks(profile, bib, bio_md, news_md, persona=args.persona,
                                   workspace=workspace)
    template_path = TEMPLATES_DIR / f"homepage-{args.persona}.html"
    if not template_path.exists():
        die(f"Template not found: {template_path}")
    template = template_path.read_text(encoding="utf-8")
    html = render_template(template, chunks, profile, audit, workspace)

    out_path = Path(args.out) if args.out else workspace / DEFAULT_FILES["output_html"]
    out_path.write_text(html, encoding="utf-8")
    print(f"✓ Rendered: {out_path}  ({out_path.stat().st_size // 1024} KB)")
    if audit:
        print(f"✓ Audit: {audit.verdict}  "
              f"(PASS {len(audit.passed)} · WARN {len(audit.warned)} · "
              f"FAIL {len(audit.failed)} · OVERRIDDEN {len(audit.overridden)})")
    return 0


def load_profile(path: Path) -> dict[str, Any]:
    """Load + validate profile.yml. Required: schema_version, identity.name."""
    if not HAS_YAML:
        die("pyyaml not installed. Run: pip install pyyaml --break-system-packages")
    if not path.exists():
        die(f"{path} not found. Run `{invocation()} init --from-cv <cv>` first.")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        die(f"{path}: expected a YAML mapping at top level.")
    sv = data.get("schema_version")
    if sv != SCHEMA_VERSION:
        die(f"schema_version mismatch: file has {sv!r}, expected {SCHEMA_VERSION}.")
    if not data.get("identity", {}).get("name"):
        die(f"{path}: identity.name is required.")
    return data


def parse_bibtex(path: Path) -> dict[str, dict[str, str]]:
    """Load publications.bib via stdlib parser."""
    if not path.exists():
        return {}
    return parse_bibtex_str(path.read_text(encoding="utf-8"))


def read_optional(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


# ---------------------------------------------------------------------------
# Section chunk builders
# ---------------------------------------------------------------------------
# Each returns "" when the corresponding profile field is empty (graceful
# degradation — no LLM filler, per Gemini's risk #3).

def build_section_chunks(profile: dict, bib: dict, bio_md: str, news_md: str,
                          *, persona: str, workspace: Path) -> dict[str, str]:
    return {
        "PHOTO_HTML": build_photo_html(profile, workspace),
        "PERSON_NAME": esc(profile["identity"]["name"]),
        "PERSON_NAME_NATIVE_HTML": build_native_name_html(profile),
        "TITLE_AFFILIATION_HTML": build_title_affil_html(profile),
        "EMAIL_HTML": build_email_html(profile),
        "SOCIAL_LINKS_HTML": build_social_html(profile),
        "BIO_SECTION_HTML": build_bio_section(profile, bio_md),
        "RESEARCH_EXPERIENCES_SECTION_HTML": build_research_experiences_section(profile),
        "FEATURED_PROJECTS_SECTION_HTML": build_featured_projects_section(profile),
        "RESEARCH_SECTION_HTML": build_research_section(profile),
        "EDUCATION_SECTION_HTML": build_education_section(profile),
        "NEWS_SECTION_HTML": build_news_section(news_md),
        "PUBLICATIONS_SECTION_HTML": build_publications_section(profile, bib, persona),
        "AWARDS_SECTION_HTML": build_awards_section(profile),
        "TALKS_SECTION_HTML": build_talks_section(profile),
        "PROFESSIONAL_SERVICES_SECTION_HTML": build_professional_services_section(profile),
        "TEACHING_SECTION_HTML": build_teaching_section(profile),
        "AUDIT_LINK_HTML": f" · <a href=\"{DEFAULT_FILES['audit_report']}\">audit</a>",
    }


def build_photo_html(profile: dict, workspace: Path) -> str:
    """Render headshot. Supports local path (base64-inline) OR remote URL
    (e.g. raw.githubusercontent.com — embeds the URL directly, no fetch)."""
    photo_path_str = profile.get("identity", {}).get("photo")
    if not photo_path_str:
        return ""
    # Remote URL — embed directly. Single-file HTML keeps the URL but doesn't fetch+inline.
    if photo_path_str.startswith(("http://", "https://")):
        return f'<img class="photo" src="{esc(photo_path_str)}" alt="" loading="lazy">'
    # Local path — base64-inline
    photo = workspace / photo_path_str
    if not photo.exists():
        print(f"  ⚠ photo not found: {photo}", file=sys.stderr)
        return ""
    if photo.stat().st_size > 500_000:
        print(f"  ⚠ photo >500KB ({photo.stat().st_size//1024}KB), embedding anyway. "
              f"Consider downscaling.", file=sys.stderr)
    mime, _ = mimetypes.guess_type(photo.name)
    mime = mime or "image/jpeg"
    data = base64.b64encode(photo.read_bytes()).decode("ascii")
    return f'<img class="photo" src="data:{mime};base64,{data}" alt="">'


def build_native_name_html(profile: dict) -> str:
    native = profile.get("identity", {}).get("name_native")
    if not native:
        return ""
    return f' <span class="name-native">{esc(native)}</span>'


def build_title_affil_html(profile: dict) -> str:
    """Render title + PRIMARY current affiliation only (single line, like manual academic pages).

    Secondary affiliations (e.g., visiting positions) are intentionally NOT shown in the
    masthead — they belong in the bio prose where context can be given.
    """
    title = profile.get("identity", {}).get("title")
    current = profile.get("affiliations", {}).get("current", []) or []
    primary_parts = []
    if title:
        primary_parts.append(esc(title))
    if current:
        a = current[0]
        affil_parts = [esc(a[k]) for k in ("institution", "department") if a.get(k)]
        if affil_parts:
            primary_parts.append(", ".join(affil_parts))
    return " · ".join(primary_parts) if primary_parts else ""


def build_email_html(profile: dict) -> str:
    """Render multi-line contact block: Email + WeChat + Office (one per line).
    Mirrors the manual-academic-homepage convention of stacked plain-text contact lines."""
    identity = profile.get("identity", {}) or {}
    lines: list[str] = []
    email = identity.get("email")
    wechat = identity.get("wechat")
    office = identity.get("office")
    if email:
        lines.append(f'Email: <a href="mailto:{esc(email)}">{esc(email)}</a>')
    if wechat:
        lines.append(f'WeChat: {esc(wechat)}')
    if office:
        lines.append(f'Office: {esc(office)}')
    return "<br>".join(lines)


def build_social_html(profile: dict) -> str:
    """Render social/scholar link strip. Suppresses self-referential Homepage link
    (if links.homepage matches the user's GitHub Pages URL — output target)."""
    links = profile.get("links", {}) or {}
    suppress: set[str] = set()
    homepage = links.get("homepage", "")
    # Heuristic: if homepage points to <user>.github.io, it's self-referential
    if homepage and (".github.io" in homepage or "github.io/" in homepage):
        suppress.add("homepage")
    out = []
    for key, label in LINK_LABELS.items():
        url = links.get(key)
        if not url or key in suppress:
            continue
        out.append(f'<a href="{esc(url)}" rel="noopener">{label}</a>')
    return "\n      ".join(out)


def build_bio_section(profile: dict, bio_md: str) -> str:
    body = bio_md.strip() or profile.get("research", {}).get("summary", "")
    if not body:
        return ""
    inner = md_to_paragraphs(body) if "\n" in body or "[" in body else f"<p>{md_inline(body)}</p>"
    return f'<section class="bio"><h2>Bio</h2>\n{inner}\n</section>'


def build_research_section(profile: dict) -> str:
    research = profile.get("research", {}) or {}
    interests = research.get("interests") or []
    if not interests:
        return ""
    items = "\n".join(f"  <li>{md_inline(str(i))}</li>" for i in interests)
    return (f'<section class="research"><h2>Research Interests</h2>\n'
            f'<ul>\n{items}\n</ul>\n</section>')


def build_education_section(profile: dict) -> str:
    edus = profile.get("education", []) or []
    if not edus:
        return ""
    rows = []
    for e in edus:
        degree = esc(e.get("degree", ""))
        inst = esc(e.get("institution", ""))
        adv = e.get("advisor") or []
        adv_html = ""
        if adv:
            adv_str = ", ".join(adv) if isinstance(adv, list) else str(adv)
            adv_html = f" · advised by {esc(adv_str)}"
        dates = format_dates(e.get("start"), e.get("end"))
        rows.append(f'<div class="edu-item"><span class="degree">{degree}'
                    f'{" · " + inst if inst else ""}{adv_html}</span>'
                    f'<span class="dates">{esc(dates)}</span></div>')
    return (f'<section class="education"><h2>Education</h2>\n' +
            "\n".join(rows) + "\n</section>")


def _normalize_selected_publications(selected: Any) -> list[dict]:
    """Accepts flat list OR list of {group: str, keys: [bibkey]} dicts.
    Returns list of groups: [{'group': str|None, 'keys': [bibkey, ...]}, ...]."""
    if not selected:
        return []
    if all(isinstance(x, str) for x in selected):
        return [{"group": None, "keys": list(selected)}]
    groups = []
    for item in selected:
        if isinstance(item, str):
            groups.append({"group": None, "keys": [item]})
        elif isinstance(item, dict):
            groups.append({"group": item.get("group"), "keys": item.get("keys", [])})
    return groups


def build_publications_section(profile: dict, bib: dict, persona: str) -> str:
    selected = profile.get("selected_publications") or []
    if not selected:
        return ""
    groups = _normalize_selected_publications(selected)
    meta = profile.get("publications_meta", {}) or {}
    user_name = profile.get("identity", {}).get("name", "")
    name_tokens = [t.lower() for t in re.findall(r"\w+", user_name) if t]
    preamble = (profile.get("publications", {}) or {}).get("preamble", "")

    missing: list[str] = []
    section_chunks: list[str] = []

    def render_paper_row(bibkey: str) -> str:
        if bibkey not in bib:
            missing.append(bibkey)
            return ""
        entry = bib[bibkey]
        m = meta.get(bibkey, {}) or {}
        spotlight_cls = " pub-spotlight" if m.get("spotlight") else ""
        title = esc(entry.get("title", "(untitled)"))
        authors_str = render_authors(entry.get("author", ""), name_tokens,
                                      co_first=m.get("co_first"))
        venue = render_venue(entry)
        link_html_parts = []
        plinks = m.get("links", {}) or {}
        # Known link types in canonical display order.
        # User flagged "abs" as too cryptic when the link is just the paper — use "Paper" / "arXiv".
        known_link_order = [
            ("paper", "Paper"),
            ("arxiv", "arXiv"),
            ("pdf", "PDF"),
            ("openreview", "OpenReview"),
            ("project", "Project"),
            ("code", "Code"),
            ("slides", "Slides"),
            ("talk_slides", "Talk Slides"),
            ("html_intro", "Intro"),
            ("html", "HTML"),
            ("video", "Video"),
            ("poster", "Poster"),
            ("paperweekly", "PaperWeekly"),
            ("bibtex", "BibTeX"),
        ]
        seen_link_keys: set[str] = set()
        for key, label in known_link_order:
            url = plinks.get(key)
            if url:
                link_html_parts.append(f'<a href="{esc(url)}" rel="noopener">[{label}]</a>')
                seen_link_keys.add(key)
        # Custom/unknown keys keep their key name as label
        for key, url in plinks.items():
            if key in seen_link_keys or not url:
                continue
            link_html_parts.append(f'<a href="{esc(url)}" rel="noopener">[{esc(key)}]</a>')
        # Awards/badges — support both singular `award: str` (back-compat) and plural `awards: [str]`
        award_items: list[str] = []
        if m.get("awards"):
            award_items = [str(a) for a in m["awards"] if a]
        elif m.get("award"):
            award_items = [str(m["award"])]
        award_html = "".join(f'<span class="pub-badge">{esc(a)}</span>' for a in award_items)
        links_html = (f'<div class="pub-links">{"".join(link_html_parts)}{award_html}</div>'
                      if link_html_parts or award_items else "")
        description = m.get("description", "")
        description_html = (f'<div class="pub-description">{md_inline(description)}</div>'
                            if description else "")
        content_html = (f'<div class="pub-title">{title}</div>'
                        f'<div class="pub-authors">{authors_str}</div>'
                        f'<div class="pub-venue">{venue}</div>'
                        f'{links_html}')
        # 2-column layout when thumbnail provided. Description renders FULL-WIDTH
        # below the thumb+content row via CSS grid-template-areas — matches the
        # DFS-GRPO single-column treatment but kept inside the 2-col grid.
        thumb = m.get("thumbnail")
        if thumb:
            thumb_html = (f'<div class="pub-thumb-wrap">'
                          f'<img class="pub-thumb" src="{esc(thumb)}" alt="" loading="lazy">'
                          f'</div>')
            return (f'<div class="pub-item pub-item-with-thumb{spotlight_cls}">'
                    f'{thumb_html}<div class="pub-content">{content_html}</div>'
                    f'{description_html}</div>')
        return f'<div class="pub-item{spotlight_cls}">{content_html}{description_html}</div>'

    for group in groups:
        if group["group"]:
            section_chunks.append(f'<h3 class="pub-group">{esc(group["group"])}</h3>')
        for bibkey in group["keys"]:
            row = render_paper_row(bibkey)
            if row:
                section_chunks.append(row)

    if missing:
        print(f"  ⚠ {len(missing)} bibkey(s) in selected_publications not found in .bib: "
              f"{', '.join(missing)}", file=sys.stderr)

    # Scholar deflection
    scholar = profile.get("links", {}).get("google_scholar")
    deflect = (f'<div class="scholar-deflect">Full list on '
               f'<a href="{esc(scholar)}" rel="noopener">Google Scholar</a>.</div>'
               if scholar else "")

    preamble_html = (f'<p class="pub-preamble">{md_inline(preamble)}</p>\n'
                     if preamble else "")
    return (f'<section class="pubs"><h2>Selected Publications</h2>\n' +
            preamble_html +
            "\n".join(section_chunks) + (f"\n{deflect}" if deflect else "") + "\n</section>")


def render_authors(author_field: str, user_name_tokens: list[str],
                    co_first: list[str] | None = None) -> str:
    """Render BibTeX authors → HTML. Bold the user's name; mark co-first with asterisk."""
    if not author_field:
        return ""
    authors = split_authors(author_field)
    out = []
    co_first_set = {a.lower().strip() for a in (co_first or [])}
    has_co_first = bool(co_first_set)
    for a in authors:
        a_norm = a.strip()
        # BibTeX convention: `and others` → render as `et al.`
        if a_norm.lower() in ("others", "et al.", "et al"):
            out.append('<em>et al.</em>')
            continue
        a_tokens = {t.lower() for t in re.findall(r"\w+", a_norm)}
        is_user = bool(user_name_tokens) and all(t in a_tokens for t in user_name_tokens)
        marker = "*" if has_co_first and a_norm.lower() in co_first_set else ""
        rendered = esc(a_norm) + marker
        if is_user:
            rendered = f'<span class="you">{rendered}</span>'
        out.append(rendered)
    s = ", ".join(out)
    if has_co_first:
        s += ' <em>(* equal contribution)</em>'
    return s


def render_venue(entry: dict) -> str:
    """Render BibTeX venue string. Falls back through booktitle → journal → howpublished.
    The howpublished fallback is essential for @misc preprint entries."""
    venue = (entry.get("booktitle") or entry.get("journal")
             or entry.get("howpublished") or entry.get("venue") or "")
    year = entry.get("year") or ""
    parts = [esc(v) for v in [venue, str(year)] if v]
    return ", ".join(parts) if parts else "—"


def build_news_section(news_md: str) -> str:
    if not news_md.strip():
        return ""
    # Parse bullets of form "- YYYY[-MM[-DD]]: Content"
    items = []
    for line in news_md.splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        body = line.lstrip("-").strip()
        m = re.match(r"(\d{4}(?:-\d{1,2}(?:-\d{1,2})?)?)\s*[:：]\s*(.+)", body)
        if m:
            date, content = m.group(1), m.group(2)
        else:
            date, content = "", body
        items.append(f'<div class="news-item"><span class="date">{esc(date)}</span>'
                     f'<span class="content">{md_inline(content)}</span></div>')
    items = items[:10]  # cap to last 10
    if not items:
        return ""
    return (f'<section class="news"><h2>News</h2>\n' +
            "\n".join(items) + "\n</section>")


def build_awards_section(profile: dict) -> str:
    awards = profile.get("awards", []) or []
    if not awards:
        return ""
    heading = (profile.get("ship", {}) or {}).get("awards_heading", "Awards")
    rows = []
    for a in awards:
        title = esc(a.get("title", ""))
        issuer = esc(a.get("issuer", ""))
        year = a.get("year", "")
        url = a.get("url")
        title_html = f'<a href="{esc(url)}" rel="noopener">{title}</a>' if url else title
        meta_parts = [p for p in [issuer, str(year) if year else ""] if p]
        meta = f' <span class="award-meta">— {", ".join(meta_parts)}</span>' if meta_parts else ""
        rows.append(f'<div class="award-item"><span class="award-title">{title_html}</span>{meta}</div>')
    return (f'<section class="awards"><h2>{esc(heading)}</h2>\n' +
            "\n".join(rows) + "\n</section>")


def build_featured_projects_section(profile: dict) -> str:
    """Render featured-project cards. Each is a dedicated section with stats grid +
    primary links cluster + sub-projects strip. Distinct from publications/bio.

    Schema (per profile.yml):
        featured_projects:
          - id: aris
            name: "ARIS — Auto Research in Sleep"
            tagline: "..."
            subtitle: "..."  # optional
            logo: "https://..."  # optional (URL or local path)
            stats: ["~10K stars", "HF Daily #1", ...]  # list of strings OR
                  # list of {label, value, source_url} for richer rendering
            links:
              repo: "..."
              paper: "..."
              ...
            subprojects:
              - {name, url, tagline}
            elevator: "..."  # 1-2 sentence description
    """
    fps = profile.get("featured_projects", []) or []
    if not fps:
        return ""

    cards = []
    for fp in fps:
        name = esc(fp.get("name", ""))
        tagline = md_inline(fp.get("tagline", "")) if fp.get("tagline") else ""
        subtitle = esc(fp.get("subtitle", "")) if fp.get("subtitle") else ""
        elevator = md_inline(fp.get("elevator", "")) if fp.get("elevator") else ""
        logo = fp.get("logo")

        # Stats — accept either list of strings or list of dicts
        stats = fp.get("stats", []) or []
        stat_items = []
        for s in stats:
            if isinstance(s, str):
                stat_items.append(f'<li>{esc(s)}</li>')
            elif isinstance(s, dict):
                label = esc(s.get("label", ""))
                value = esc(s.get("value", ""))
                url = s.get("source_url")
                v_html = f'<a href="{esc(url)}" rel="noopener">{value}</a>' if url else value
                stat_items.append(f'<li><span class="stat-label">{label}</span>'
                                  f'<span class="stat-value">{v_html}</span></li>')
        stats_html = (f'<ul class="fp-stats">{"".join(stat_items)}</ul>'
                      if stat_items else "")

        # Primary links
        links = fp.get("links", {}) or {}
        primary_link_order = [
            ("repo", "GitHub"), ("paper", "Paper"), ("arxiv", "arXiv"),
            ("talk_slides", "Talk Slides"), ("html_intro", "Intro (HTML)"),
            ("paperweekly", "PaperWeekly"), ("awesome_list", "awesome-list"),
            ("releases", "Releases"), ("star_history", "Star History"),
        ]
        link_chunks = []
        for key, label in primary_link_order:
            url = links.get(key)
            if url:
                link_chunks.append(f'<a href="{esc(url)}" rel="noopener">{label}</a>')
        for key, url in links.items():
            if key in {k for k, _ in primary_link_order}:
                continue
            if url:
                link_chunks.append(f'<a href="{esc(url)}" rel="noopener">{esc(key)}</a>')
        links_html = (f'<nav class="fp-links">{" · ".join(link_chunks)}</nav>'
                      if link_chunks else "")

        elevator_html = f'<p class="fp-elevator">{elevator}</p>' if elevator else ""

        # Sub-projects
        subprojects = fp.get("subprojects", []) or []
        sub_items = []
        for sp in subprojects:
            sp_name = esc(sp.get("name", ""))
            sp_url = sp.get("url")
            sp_tag = md_inline(sp.get("tagline", "")) if sp.get("tagline") else ""
            if sp_url:
                sub_items.append(f'<li><a href="{esc(sp_url)}" rel="noopener">'
                                 f'<strong>{sp_name}</strong></a> — {sp_tag}</li>')
            else:
                sub_items.append(f'<li><strong>{sp_name}</strong> — {sp_tag}</li>')
        sub_html = (f'<div class="fp-subprojects-label">Sub-projects:</div>'
                    f'<ul class="fp-subprojects">{"".join(sub_items)}</ul>'
                    if sub_items else "")

        # Open problems (CV's research-direction prompts)
        open_problems = fp.get("open_problems", []) or []
        op_items = []
        for op in open_problems:
            op_items.append(f'<li>{md_inline(str(op))}</li>')
        op_html = (f'<div class="fp-op-label">Open problems explored in this line of work:</div>'
                   f'<ol class="fp-open-problems">{"".join(op_items)}</ol>'
                   if op_items else "")

        # Assemble text-column content (everything except the logo image)
        title_block_parts = [f'<h2 class="fp-name">{name}</h2>']
        if tagline:
            title_block_parts.append(f'<div class="fp-tagline">{tagline}</div>')
        if subtitle:
            title_block_parts.append(f'<div class="fp-subtitle">{subtitle}</div>')
        title_block = "".join(title_block_parts)

        text_col = (f'{title_block}{stats_html}{links_html}'
                    f'{elevator_html}{sub_html}{op_html}')

        # Float-right image (user-requested: 2-col only until image ends, then 1-col continues)
        if logo:
            logo_html = (f'<img class="fp-logo-float" src="{esc(logo)}" '
                         f'alt="ARIS workflow diagram" loading="lazy">')
            cards.append(
                f'<section class="featured-project" id="fp-{esc(fp.get("id", "x"))}">'
                f'{logo_html}{text_col}</section>'
            )
        else:
            cards.append(
                f'<section class="featured-project" id="fp-{esc(fp.get("id", "x"))}">'
                f'{text_col}</section>'
            )

    return "\n".join(cards)


def build_talks_section(profile: dict) -> str:
    """Combined 'Blogs & Tutorials & Talks' section — mirrors manual academic homepages.

    Renders blogs_tutorials[] items first (with [html][pdf][slides] link triplet)
    then talks[] items. If only one type present, heading adapts.
    """
    blogs = profile.get("blogs_tutorials", []) or []
    talks = profile.get("talks", []) or []
    if not blogs and not talks:
        return ""

    if blogs and talks:
        heading = "Blogs & Tutorials & Talks"
    elif blogs:
        heading = "Blogs & Tutorials"
    else:
        heading = "Talks"

    rows = []
    # Blogs / tutorials first (typically text/PDF/slides artifacts)
    for b in blogs:
        title = esc(b.get("title", ""))
        category = b.get("category")
        cat_html = f' <em class="category-tag">({esc(category)})</em>' if category else ''
        links = b.get("links", {}) or {}
        link_order = [("html", "html"), ("pdf", "pdf"), ("slides", "slides"), ("video", "video"),
                       ("project", "project"), ("code", "code")]
        link_parts = []
        for label, key in link_order:
            url = links.get(key)
            if url:
                link_parts.append(f'<a href="{esc(url)}" rel="noopener">[{label}]</a>')
        links_html = (' ' + ' '.join(link_parts)) if link_parts else ''
        note = b.get("note")
        note_html = f' <span class="award-meta">— {esc(note)}</span>' if note else ''
        rows.append(f'<div class="award-item"><span class="award-title">{title}{cat_html}</span>'
                    f'{note_html}{links_html}</div>')
    # Talks
    for t in talks:
        title = esc(t.get("title", ""))
        venue = esc(t.get("venue", ""))
        date = esc(t.get("date", ""))
        url = t.get("url")
        slides = t.get("slides")
        title_html = f'<a href="{esc(url)}" rel="noopener">{title}</a>' if url else title
        meta_parts = [p for p in [venue, date] if p]
        meta = f' <span class="award-meta">— {", ".join(meta_parts)}</span>' if meta_parts else ""
        slides_html = (f' <a href="{esc(slides)}" rel="noopener">[slides]</a>'
                       if slides else "")
        rows.append(f'<div class="award-item"><span class="award-title">{title_html}</span>'
                    f'{meta}{slides_html}</div>')

    return (f'<section class="awards"><h2>{esc(heading)}</h2>\n' +
            "\n".join(rows) + "\n</section>")


def build_teaching_section(profile: dict) -> str:
    teaching = profile.get("teaching", []) or []
    if not teaching:
        return ""
    rows = []
    for t in teaching:
        course = esc(t.get("course", ""))
        role = esc(t.get("role", ""))
        term = esc(t.get("term", ""))
        meta = f" · {role}" if role else ""
        meta += f" · {term}" if term else ""
        rows.append(f'<div class="award-item"><span class="award-title">{course}</span>'
                    f'<span class="award-meta">{meta}</span></div>')
    return (f'<section class="awards"><h2>Teaching</h2>\n' +
            "\n".join(rows) + "\n</section>")


def build_professional_services_section(profile: dict) -> str:
    """Render 'Professional Services' section (Conference Reviewer list etc).

    Schema:
        professional_services:
          preamble: "Conference Reviewer for:"
          items: ["AAMAS 2022", "NeurIPS 2024, 2025", ...]
    """
    ps = profile.get("professional_services") or {}
    items = ps.get("items") or []
    if not items:
        return ""
    preamble = ps.get("preamble", "")
    parts = []
    if preamble:
        parts.append(f'<p class="ps-preamble">{md_inline(preamble)}</p>')
    item_html = "".join(f'<div class="award-item">{md_inline(str(item))}</div>' for item in items)
    parts.append(item_html)
    return (f'<section class="awards"><h2>Professional Services</h2>\n' +
            "\n".join(parts) + "\n</section>")


def build_research_experiences_section(profile: dict) -> str:
    """Render past affiliations as a dedicated 'Research Experiences' section.

    Mirrors the user's manual homepage convention — these are prominent placements
    that recruiters/peers look for, not just CV-style affiliation history.
    """
    past = profile.get("affiliations", {}).get("past", []) or []
    if not past:
        return ""
    rows = []
    for a in past:
        role = esc(a.get("role", ""))
        inst = esc(a.get("institution", ""))
        dept = esc(a.get("department", "")) if a.get("department") else ""
        dates = format_dates(a.get("start"), a.get("end"))
        title_parts = [role] if role else []
        if inst:
            title_parts.append(f"at <b>{inst}</b>")
        title_line = " ".join(title_parts)
        meta_parts = []
        if dept:
            meta_parts.append(dept)
        if dates:
            meta_parts.append(esc(dates))
        meta = f' <span class="award-meta">— {" · ".join(meta_parts)}</span>' if meta_parts else ""
        rows.append(f'<div class="award-item"><span class="award-title">{title_line}</span>{meta}</div>')
    return (f'<section class="awards"><h2>Research Experiences</h2>\n' +
            "\n".join(rows) + "\n</section>")


def format_dates(start: str | None, end: str | None) -> str:
    if not start:
        return esc(str(end)) if end else ""
    start = str(start)
    if end is None:
        return f"{start} – present"
    return f"{start} – {end}"


# ---------------------------------------------------------------------------
# Fact-check execution
# ---------------------------------------------------------------------------

class AuditResult:
    def __init__(self) -> None:
        self.passed: list[dict] = []
        self.warned: list[dict] = []
        self.failed: list[dict] = []
        self.overridden: list[dict] = []

    @property
    def verdict(self) -> str:
        if self.failed:
            return "BLOCKED"
        if self.warned:
            return "WARN"
        return "PASS"


def run_fact_check(profile: dict, bib: dict, *, override_all: bool = False,
                    workspace: Path | None = None) -> AuditResult:
    """Hard-fail + soft-warn checks. See SKILL.md for the canonical list."""
    result = AuditResult()
    overrides = profile.get("audit", {}).get("overrides", {}) or {}
    # Normalize: flatten grouped selected_publications back to a flat list of bibkeys
    selected_groups = _normalize_selected_publications(
        profile.get("selected_publications", []) or [])
    selected = [key for g in selected_groups for key in g["keys"]]
    meta = profile.get("publications_meta", {}) or {}
    workspace = workspace or Path(".")
    today_iso = datetime.now(timezone.utc).date().isoformat()

    # Hard-fail 1: bibkey existence in bib
    for key in selected:
        if key not in bib:
            result.failed.append({
                "kind": "missing_bibkey", "key": key,
                "detail": "bibkey listed in selected_publications but not in publications.bib",
            })

    # Per-paper DBLP check
    for key in selected:
        if key not in bib:
            continue
        entry = bib[key]
        # Check overrides first
        ov = overrides.get(key, {})
        if ov:
            if not _override_active(ov, today_iso):
                result.failed.append({
                    "kind": "override_expired", "key": key,
                    "detail": f"override expired ({ov.get('expires')})",
                })
                continue
            result.overridden.append({
                "key": key, "fields": [k for k in ov if k not in ("reason", "expires")],
                "reason": ov.get("reason", ""), "expires": ov.get("expires", "no expiry"),
            })
            continue

        # DBLP query
        title = entry.get("title", "")
        if not title:
            result.warned.append({"kind": "no_title", "key": key, "detail": "BibTeX has no title"})
            continue
        hits = dblp_search_title(title, workspace=workspace)
        if not hits:
            # arXiv fallback heuristic: if entry has eprint/arxiv ID, accept as warn
            if entry.get("eprint") or "arxiv" in (entry.get("archiveprefix", "").lower()):
                result.warned.append({
                    "kind": "arxiv_only", "key": key,
                    "detail": "No DBLP hit; arXiv eprint present.",
                })
            else:
                result.warned.append({
                    "kind": "no_dblp_hit", "key": key,
                    "detail": "No DBLP hit; venue may be workshop/preprint/non-indexed.",
                })
            continue

        # Multiple hits → ambiguous warn
        if len(hits) > 1:
            # Try to disambiguate by year match
            entry_year = str(entry.get("year", ""))
            year_matches = [h for h in hits if _hit_year(h) == entry_year]
            if len(year_matches) == 1:
                hits = year_matches
            else:
                result.warned.append({
                    "kind": "ambiguous_dblp", "key": key,
                    "detail": f"{len(hits)} DBLP candidates; cannot uniquely match.",
                })
                continue

        hit = hits[0]
        # Year check
        dblp_year = _hit_year(hit)
        entry_year = str(entry.get("year", ""))
        if dblp_year and entry_year and dblp_year != entry_year:
            result.failed.append({
                "kind": "year_mismatch", "key": key,
                "detail": f"BibTeX year={entry_year}, DBLP year={dblp_year}.",
            })
            continue
        # Venue check (fuzzy — DBLP venue strings vary)
        dblp_venue = _hit_venue(hit)
        entry_venue = (entry.get("booktitle") or entry.get("journal") or "").strip()
        if dblp_venue and entry_venue and not _venue_match(entry_venue, dblp_venue):
            result.failed.append({
                "kind": "venue_mismatch", "key": key,
                "detail": f"BibTeX venue='{entry_venue}', DBLP venue='{dblp_venue}'.",
            })
            continue
        # Passed
        result.passed.append({
            "key": key, "detail": f"verified by DBLP ({_hit_url(hit)})",
        })

    # Award badge sanity: claims like "Best Paper" / "Spotlight" / "Oral" need a URL.
    # Supports both `award` (singular, back-compat) and `awards` (plural list, v2+).
    for key, m in meta.items():
        m = m or {}
        award_pool: list[str] = []
        if m.get("awards"):
            award_pool = [str(a) for a in m["awards"] if a]
        elif m.get("award"):
            award_pool = [str(m["award"])]
        flag_terms = ("best paper", "spotlight", "oral", "outstanding")
        links = m.get("links", {}) or {}
        has_verifier = bool(
            links.get("arxiv") or links.get("paper") or links.get("pdf")
            or links.get("project") or links.get("openreview")
        )
        for award in award_pool:
            if any(t in award.lower() for t in flag_terms) and not has_verifier:
                result.failed.append({
                    "kind": "award_unverifiable", "key": key,
                    "detail": f"Award '{award}' claimed but no verifiable URL.",
                })

    if override_all:
        # Promote failures to overridden (still flagged loudly)
        for f in result.failed:
            result.overridden.append({
                "key": f.get("key", "?"),
                "fields": [f.get("kind", "?")],
                "reason": "--override-all CLI flag",
                "expires": "immediate (no expiry recorded)",
            })
        result.failed = []

    _save_dblp_cache(workspace)
    return result


def _override_active(ov: dict, today_iso: str) -> bool:
    expires = ov.get("expires")
    if not expires:
        return True  # no expiration set
    return str(expires) >= today_iso


_dblp_cache: dict[str, list[dict]] | None = None


def _load_dblp_cache(workspace: Path) -> dict[str, list[dict]]:
    """Load DBLP cache from workspace. Cache persists across runs to dodge rate limits."""
    global _dblp_cache
    if _dblp_cache is not None:
        return _dblp_cache
    cache_path = workspace / DBLP_CACHE_DIR / DBLP_CACHE_FILE
    if cache_path.exists():
        try:
            _dblp_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            _dblp_cache = {}
    else:
        _dblp_cache = {}
    return _dblp_cache


def _save_dblp_cache(workspace: Path) -> None:
    if _dblp_cache is None:
        return
    cache_dir = workspace / DBLP_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / DBLP_CACHE_FILE).write_text(
        json.dumps(_dblp_cache, ensure_ascii=False, indent=2), encoding="utf-8")


def dblp_search_title(title: str, workspace: Path | None = None) -> list[dict]:
    """Search DBLP by title with rate-limit backoff + local cache (cache survives across runs)."""
    if workspace is None:
        workspace = Path(".")
    cache = _load_dblp_cache(workspace)
    cache_key = title.strip().lower()
    if cache_key in cache:
        return cache[cache_key]

    q = urllib.parse.urlencode({"q": title, "format": "json", "h": 5})
    url = f"{DBLP_API}?{q}"
    # DBLP politeness: 2s minimum between queries; exponential backoff on retry
    for attempt in range(4):
        time.sleep(2.0 * (1 + attempt))
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ARIS-Homepage/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.load(r)
            hits = data.get("result", {}).get("hits", {}).get("hit", [])
            hits = hits if isinstance(hits, list) else [hits]
            cache[cache_key] = hits
            return hits
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:
                continue
            print(f"  ⚠ DBLP HTTP {e.code} for '{title[:60]}...' (giving up)", file=sys.stderr)
            cache[cache_key] = []  # cache the failure to avoid retry-storm next render
            return []
        except (urllib.error.URLError, ConnectionResetError, TimeoutError) as e:
            if attempt < 3:
                continue
            print(f"  ⚠ DBLP unreachable for '{title[:60]}...': {e}", file=sys.stderr)
            return []  # don't cache transient network failures
        except Exception as e:
            print(f"  ⚠ DBLP query failed for '{title[:60]}...': {e}", file=sys.stderr)
            return []
    return []


def _hit_year(hit: dict) -> str:
    return str(hit.get("info", {}).get("year", "")).strip()


def _hit_venue(hit: dict) -> str:
    info = hit.get("info", {})
    return (info.get("venue") or "").strip()


def _hit_url(hit: dict) -> str:
    info = hit.get("info", {})
    return (info.get("key") or info.get("url") or "").strip()


def _venue_match(claimed: str, dblp: str) -> bool:
    """Fuzzy venue match. Handles 'NeurIPS' ↔ 'NIPS' ↔ 'Advances in Neural...' families."""
    norm = lambda s: re.sub(r"\W+", "", s.lower())
    a, b = norm(claimed), norm(dblp)
    if a == b or a in b or b in a:
        return True
    # Common aliases
    aliases = [
        {"neurips", "nips", "advancesinneuralinformationprocessingsystems"},
        {"iclr", "internationalconferenceonlearningrepresentations"},
        {"icml", "internationalconferenceonmachinelearning"},
        {"aaai", "aaaiconferenceonartificialintelligence"},
        {"cvpr", "ieeeconferenceoncomputervisionandpatternrecognition"},
        {"acl", "annualmeetingoftheassociationforcomputationallinguistics"},
        {"aistats", "internationalconferenceonartificialintelligenceandstatistics"},
    ]
    for alias_set in aliases:
        if a in alias_set and b in alias_set:
            return True
    return False


def write_audit_report(path: Path, audit: AuditResult, profile: dict, persona: str) -> None:
    """Write Markdown audit report next to the HTML."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    verdict_emoji = {"PASS": "✅", "WARN": "⚠️", "BLOCKED": "❌"}
    lines = [
        f"# ARIS Homepage Audit Report",
        f"",
        f"Generated: {now}",
        f"Persona: {persona}",
        f"Verdict: {verdict_emoji.get(audit.verdict, '?')} {audit.verdict}",
        f"",
        f"## Summary",
        f"",
        f"PASS {len(audit.passed)} · WARN {len(audit.warned)} · "
        f"FAIL {len(audit.failed)} · OVERRIDDEN {len(audit.overridden)}",
        f"",
    ]
    if audit.failed:
        lines.append("## ❌ Failures (must fix or `--override-all`)\n")
        for f in audit.failed:
            lines.append(f"- **{f.get('key', '?')}** [`{f.get('kind', '?')}`]: "
                         f"{f.get('detail', '')}")
        lines.append("")
    if audit.warned:
        lines.append("## ⚠️ Warnings (review, but render proceeds)\n")
        for w in audit.warned:
            lines.append(f"- **{w.get('key', '?')}** [`{w.get('kind', '?')}`]: "
                         f"{w.get('detail', '')}")
        lines.append("")
    if audit.overridden:
        lines.append("## 🔐 Overrides applied\n")
        for o in audit.overridden:
            fields = ", ".join(o.get("fields", [])) or "all"
            lines.append(f"- **{o.get('key', '?')}** [{fields}] — expires "
                         f"{o.get('expires', '?')} — reason: {o.get('reason', '')}")
        lines.append("")
    if audit.passed:
        lines.append("## ✅ Verified\n")
        for p in audit.passed:
            lines.append(f"- **{p.get('key', '?')}**: {p.get('detail', '')}")
        lines.append("")
    lines.append("---")
    lines.append("Generated by [ARIS-Homepage](https://github.com/wanshuiyin/ARIS-in-AI-Offer). "
                 f"Re-run `{invocation(path.parent)} check` from this directory after edits.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Template substitution
# ---------------------------------------------------------------------------

def render_template(template: str, chunks: dict[str, str], profile: dict,
                     audit: AuditResult | None, workspace: Path) -> str:
    now = datetime.now(timezone.utc)
    ship = profile.get("ship", {}) or {}
    base_vars = {
        "LANG": ship.get("lang", "en"),
        "PAGE_TITLE": f"{profile['identity']['name']} — Homepage",
        "SOURCE_PATH": DEFAULT_FILES["profile"],
        "SOURCE_SHA256": file_sha256(workspace / DEFAULT_FILES["profile"]),
        "GENERATED_AT": now.isoformat(),
        "GENERATED_AT_DISPLAY": now.strftime("%Y-%m-%d"),
        "ACCENT_COLOR": ship.get("accent_color", "#1a4a8c"),
        "HEAD_CDN": "",
        "AUDIT_VERDICT": audit.verdict if audit else "NONE",
    }
    base_vars.update(chunks)
    out = template
    for k, v in base_vars.items():
        out = out.replace("{{" + k + "}}", v if v is not None else "")
    return out


# ---------------------------------------------------------------------------
# Command: check
# ---------------------------------------------------------------------------

def cmd_check(args: argparse.Namespace) -> int:
    workspace = Path(".").resolve()
    profile = load_profile(workspace / DEFAULT_FILES["profile"])
    bib = parse_bibtex(workspace / DEFAULT_FILES["bib"])
    audit = run_fact_check(profile, bib, workspace=workspace)
    write_audit_report(workspace / DEFAULT_FILES["audit_report"], audit, profile,
                       persona="(check only)")
    print(f"Audit: {audit.verdict}  "
          f"(PASS {len(audit.passed)} · WARN {len(audit.warned)} · "
          f"FAIL {len(audit.failed)} · OVERRIDDEN {len(audit.overridden)})")
    if args.strict and audit.warned:
        return 1
    return 0 if audit.verdict != "BLOCKED" else 1


# ---------------------------------------------------------------------------
# Command: doctor
# ---------------------------------------------------------------------------

def cmd_doctor(args: argparse.Namespace) -> int:
    print("ARIS Homepage Doctor")
    print(f"  Python:        {sys.version.split()[0]}")
    print(f"  pyyaml:        {'✓' if HAS_YAML else '✗  (pip install pyyaml --break-system-packages)'}")
    print(f"  bibtex parser: ✓ stdlib (bundled)")
    print(f"  textutil:      {'✓' if shutil.which('textutil') else '✗  (macOS only; install poppler for PDF)'}")
    print(f"  pdftotext:     {'✓' if shutil.which('pdftotext') else '✗  (brew install poppler)'}")
    print(f"  templates dir: {'✓' if TEMPLATES_DIR.exists() else '✗ ' + str(TEMPLATES_DIR)}")
    print("  DBLP reachable: ", end="", flush=True)
    try:
        with urllib.request.urlopen("https://dblp.org/", timeout=5) as r:
            print("✓" if r.status == 200 else f"⚠ HTTP {r.status}")
    except Exception as e:
        print(f"✗  ({e})")
    return 0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def esc(s: Any) -> str:
    return html_lib.escape(str(s), quote=True)


def file_sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def backup_existing(out_dir: Path) -> None:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    for key in ("profile", "bib", "bio", "news", "extraction_review"):
        f = out_dir / DEFAULT_FILES[key]
        if f.exists():
            f.rename(f.with_suffix(f.suffix + f".bak-{ts}"))


def die(msg: str) -> None:
    print(f"✗ {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main / argparse
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        prog=invocation(),
        description="Generate a fact-checked academic homepage from a CV.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="extract CV to text, emit extraction handoff")
    p_init.add_argument("--from-cv", required=True, help=".docx / .pdf / .txt CV path")
    p_init.add_argument("--out", default=".", help="output directory (default: cwd)")
    p_init.add_argument("--from-repos",
                         help="optional: comma-separated owner/repo list to snapshot via gh CLI "
                              "(e.g. wanshuiyin/ARIS-in-AI-Offer,wanshuiyin/Auto-claude-code-research-in-sleep)")
    p_init.add_argument("--include-private", action="store_true",
                         help="allow private repos in --from-repos snapshots (default: skip)")
    g = p_init.add_mutually_exclusive_group()
    g.add_argument("--force", action="store_true", help="overwrite existing profile.yml")
    g.add_argument("--merge", action="store_true", help="fill only missing fields (v1.1)")
    p_init.set_defaults(func=cmd_init)

    p_final = sub.add_parser("finalize",
                              help="ingest .aris-homepage/extraction.json → profile.yml + .bib + .md")
    p_final.add_argument("--out", default=".", help="workspace directory (default: cwd)")
    p_final.set_defaults(func=cmd_finalize)

    p_render = sub.add_parser("render", help="render homepage HTML (runs fact-check)")
    p_render.add_argument("--persona", default="theory-minimal", choices=PERSONAS)
    p_render.add_argument("--out", help="output HTML path (default: index.html)")
    p_render.add_argument("--override-all", action="store_true",
                          help="ship past hard-fail (loudly logged in audit-report)")
    p_render.add_argument("--no-audit", action="store_true", help="skip fact-check (fast iter)")
    p_render.add_argument("--offline", action="store_true", help="skip CDN refs")
    p_render.set_defaults(func=cmd_render)

    p_check = sub.add_parser("check", help="fact-check only, no render")
    p_check.add_argument("--strict", action="store_true", help="treat WARN as FAIL")
    p_check.set_defaults(func=cmd_check)

    p_doctor = sub.add_parser("doctor", help="environment + dependency diagnostic")
    p_doctor.set_defaults(func=cmd_doctor)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
