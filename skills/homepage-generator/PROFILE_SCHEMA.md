# `profile.yml` — Complete Schema Reference

Single source of truth for all `profile.yml` fields supported by `aris_homepage.py`.

All examples use generic placeholders. Do not copy real names, institutions, paper titles, or any other personal data from the maintainer's dogfood demo into your `profile.yml`.

## Top-level shape

```yaml
schema_version: 1                       # required, int. Bumped on breaking changes.

identity: { ... }                       # required — see §1
affiliations: { current: [...], past: [...] }   # required current; past optional — see §2
education: [...]                        # optional — see §3
research: { summary, interests }        # required — see §4
links: { ... }                          # optional — see §5
featured_projects: [...]                # optional, but ARIS-tier flagship projects belong here — see §6
awards: [...]                           # optional — see §7
talks: [...]                            # optional — see §8
blogs_tutorials: [...]                  # optional, combined with talks under one H2 — see §8
teaching: [...]                         # optional — see §9
professional_services: { ... }          # optional — see §10
publications: { preamble }              # optional intro before publications H3s
selected_publications: [...]            # flat list OR ordered topic groups — see §11
publications_meta: { <bibkey>: { ... } } # per-paper augmentation — see §12
audit: { overrides: { <bibkey>: { ... } } }    # per-paper override bypass — see §13
ship: { ... }                           # render-time config — see §14
```

---

## §1 `identity` (required)

```yaml
identity:
  name: "Jane Doe"                      # required — display name
  name_native: "简·多伊"                  # optional — bilingual name rendered next to H1
  title: "Ph.D. Candidate"              # optional — e.g. "Assistant Professor"
  email: "jane@example.edu"             # optional — rendered as mailto in masthead
  wechat: "yourwechat"                  # optional — multi-line contact stack
  office: "Room 1, Building 2, ..."     # optional — multi-line contact stack
  photo: "assets/photo.jpg"             # optional — local path OR https:// URL.
                                        #   If URL, embedded as <img src> directly (no fetch).
                                        #   If local, base64-inlined (warning if >500KB).
```

## §2 `affiliations`

```yaml
affiliations:
  current:
    - institution: "Example University"
      department: "Department of CS"
      role: "Ph.D. Candidate"
      start: "2022-09"                  # YYYY-MM or YYYY
      end: null                         # null = present
  past:
    - institution: "Example Industry Lab"
      department: "Research Group"
      role: "Research Intern"
      start: "2024-06"
      end: "2024-12"
```

**Masthead rendering**: only the **first** entry in `current` is shown next to the H1. Additional current affiliations (visiting positions, joint appointments) belong in the bio prose. Past affiliations render as a separate "Research Experiences" section.

## §3 `education`

```yaml
education:
  - degree: "Ph.D. in Computer Science"
    institution: "Example University"
    department: "Department of CS"
    advisor: ["Prof. Example Advisor"]  # optional, list of strings
    start: "2022-09"
    end: "2027-06 (expected)"           # free-form string allowed
```

## §4 `research` (required)

```yaml
research:
  summary: "One-sentence description of research focus."   # required
  interests:                                               # required, list of strings
    - "Topic 1"
    - "Topic 2 (with **bold** Markdown support)"
```

## §5 `links`

```yaml
links:
  google_scholar: "https://scholar.google.com/citations?user=..."
  semantic_scholar: "https://www.semanticscholar.org/author/..."
  github: "https://github.com/example"
  dblp: "https://dblp.org/pid/.../..."
  twitter: "https://twitter.com/..."
  bluesky: "https://bsky.app/profile/..."
  linkedin: "https://linkedin.com/in/..."
  orcid: "https://orcid.org/..."
  homepage: "https://example.com"       # ⚠️ auto-suppressed if it points to <user>.github.io
  cv: "https://.../cv.pdf"
  email: "you@example.edu"              # rendered as Email label in social strip (separate from identity.email)
```

Rendered as a small social strip below the contact lines in masthead.

## §6 `featured_projects` (flagship OSS / large projects)

```yaml
featured_projects:
  - id: example-project                 # short slug for HTML anchor
    name: "Example Project Name"
    tagline: "One-sentence value prop."
    subtitle: "Optional subtitle (e.g. invited talk venue)"
    logo: "https://.../hero.svg"        # local path OR https URL. Floats right; text wraps around.
    stats:
      - {label: "GitHub Stars", value: "10K+", source_url: "https://..."}
      - "Featured · Some Outlet"        # plain string entries also accepted
    links:
      repo: "https://github.com/..."
      paper: "https://..."
      arxiv: "https://arxiv.org/abs/..."
      talk_slides: "https://.../slides.pdf"
      html_intro: "https://.../intro.html"
      # any custom key works; canonical labels handled for common ones
    elevator: "1-2 sentence description of what the project is and why it matters."
    subprojects:
      - {name: "Sub-Project A", url: "https://...", tagline: "What it does."}
      - {name: "Sub-Project B", url: null, tagline: "(Upcoming) Planned sub-project."}
    open_problems:                      # optional — renders as numbered list under "Open problems explored in this line of work"
      - "Problem 1 statement — what it is, why it matters, why it's open."
      - "Problem 2 statement."
    github:                             # (v1.1) snapshot populated by `init --from-repos`
      repo: "owner/repo"                #   provenance: which repo this snapshot is from
      snapshot_at: "2026-05-24T10:00:00Z"
      stars: 128
      forks: 9
      primary_language: "Python"
      topics: ["academic-homepage", "llm-agent"]
      created_at: "2026-05-01T00:00:00Z"
      pushed_at: "2026-05-23T00:00:00Z"
      latest_release:
        tag: "v1.0"
        date: "2026-05-23T00:00:00Z"
        url: "https://github.com/owner/repo/releases/tag/v1.0"
```

> **Note on `github:` subobject** (v1.1): populated by `python tools/aris_homepage.py init --from-repos owner/repo,...`. Written to `.aris-homepage/github_repos.json` first; calling LLM agent then maps it into `featured_projects[*].github` during the extraction step. Renderer does NOT auto-fetch — what's in profile.yml is what gets rendered (provenance lives in `snapshot_at`).

**Visual treatment**: the logo image floats right; text content (stats / links / elevator / sub-projects / open_problems) wraps around it, then continues full-width below the image. Use for ONE flagship project; the renderer supports multiple but visual weight stacks.

## §7 `awards`

```yaml
awards:
  - title: "Example Award"
    issuer: "Issuing Organization"
    year: 2025                          # int OR string ("2019, 2021")
    url: null                           # optional verification URL
```

Heading defaults to "Awards"; override via `ship.awards_heading` (see §14).

## §8 `talks` + `blogs_tutorials` (combined under one H2)

```yaml
blogs_tutorials:                        # rendered FIRST under the combined heading
  - title: "Blog Title"
    category: "(Theory)"                # optional, italic tag after title
    note: "Brief context, e.g. 'Coming Soon'"   # optional, rendered as gray meta-line after title
    links:
      html: "https://..."
      pdf: "https://..."
      slides: "https://..."

talks:
  - title: "Talk Title"
    venue: "Conference / Workshop"
    date: "2025"                        # free-form string
    url: null                           # talk URL
    slides: "https://.../slides.pdf"    # slides link, rendered as [slides]
```

Combined heading: `"Blogs & Tutorials & Talks"` if both present, else `"Blogs & Tutorials"` or `"Talks"`.

## §9 `teaching`

```yaml
teaching:
  - course: "Course Code: Course Title"
    institution: "Example University"
    role: "Teaching Assistant"
    term: "Spring 2024"
```

## §10 `professional_services`

```yaml
professional_services:
  preamble: "Conference Reviewer for:"
  items:
    - "International Conference X 2024, 2025"
    - "Conference Y 2025"
```

## §11 `selected_publications`

Two accepted forms — **topic-grouped (recommended for portfolios with distinct research lines)**:

```yaml
selected_publications:
  - group: "Topic Group A"
    keys:
      - example2026paper1
      - example2025paper2
  - group: "Topic Group B"
    keys:
      - example2024paper3
```

Or **flat list** (simpler, for small portfolios):

```yaml
selected_publications:
  - example2026paper1
  - example2025paper2
```

Group headings render as small-caps H3 above each cluster.

## §11b `publications.preamble`

```yaml
publications:
  preamble: "One-sentence framing of the research areas before the first topic group."
```

## §12 `publications_meta` (per-paper augmentation, keyed by bibkey)

```yaml
publications_meta:
  example2026paper1:
    spotlight: true                     # 4px accent border-left (vs 2px default)
    thumbnail: "https://.../thumb.png"  # 2-col layout when present
    description: |
      Optional **Markdown** blurb (long form). Renders as a full-width blue-quoted box
      below the thumb+content row. Use for highlighting a key project / internship blurb.
    awards:                             # PLURAL list of badges (each renders as a blue-bordered pill)
      - "Best Paper · Conference X 2025"
      - "Spotlight"
    award: "Best Paper"                 # singular form (back-compat) — use awards (list) for new entries
    co_first:                           # list of names to mark with * and emit "(* equal contribution)"
      - "Jane Doe"
      - "John Smith"
    links:
      arxiv: "https://arxiv.org/abs/..."
      paper: "https://..."              # any of these keys render with canonical labels
      pdf: "https://..."
      openreview: "https://openreview.net/forum?id=..."
      code: "https://github.com/..."
      project: "https://..."
      slides: "https://.../slides.pdf"
      talk_slides: "https://..."
      html_intro: "https://..."
      video: "https://..."
      poster: "https://..."
      paperweekly: "https://..."
      bibtex: "https://..."
      # unknown keys render with the key name as the label
```

**Canonical link order** (when present): Paper → arXiv → PDF → OpenReview → Project → Code → Slides → Talk Slides → Intro → HTML → Video → Poster → PaperWeekly → BibTeX → (custom keys).

## §13 `audit.overrides` (per-paper field bypass)

```yaml
audit:
  overrides:
    example2026paper1:
      venue: true                       # bypass venue-mismatch hard-fail
      authors: true                     # bypass author-list mismatch (use sparingly)
      reason: "Workshop attribution; DBLP only indexes main conference."   # REQUIRED
      expires: "2026-12-31"             # OPTIONAL — auto-expires; after this date override fails
```

Required: `reason`. Recommended: `expires` (forces re-verification annually).

## §14 `ship` (render-time config)

```yaml
ship:
  persona: theory-minimal               # currently the only fully-shipping persona
  accent_color: "#1a4a8c"               # one knob for visual accent (navy by default)
  lang: en                              # "en" | "zh" | (future) "bilingual"
  awards_heading: "Awards"              # default; override to "Honors" / "Rewards" / etc.
```

---

## Cheat-sheet: minimal valid profile.yml

```yaml
schema_version: 1
identity:
  name: "Jane Doe"
  email: "jane@example.edu"
affiliations:
  current:
    - institution: "Example University"
      role: "Ph.D. Candidate"
research:
  summary: "Studying X."
  interests: ["Topic 1", "Topic 2"]
selected_publications:
  - example2024paper
```

This is enough to render — sections without data simply disappear (graceful degradation).
