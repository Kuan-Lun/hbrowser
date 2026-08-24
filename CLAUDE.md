# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HBrowser is a Python library for browser automation on E-Hentai/ExHentai
websites. It uses `zendriver` (Chrome DevTools Protocol) for browser
automation. FlareSolverr can automatically solve Cloudflare managed challenges
and the embedded Turnstile on the Forums login form; unsupported or failed
interactive challenges fall back to manual interaction in GUI mode.

## Communication

- Claude 必須以繁體中文回答所有對話內容，不論使用者以何種語言提問；程式碼、指令、檔名、專有名詞等仍維持原文。

## Git Workflow

- Do not create or switch to a development branch.
- Perform all development work directly on the repository's primary branch
  (`master`).

## Build & Development Commands

```bash
# Install dependencies
uv pip install -e .

# Run the full Python finalizer over all project Python files
bash scripts/hooks/finalize-python.sh

# Run the full Markdown finalizer over all project Markdown files
bash scripts/hooks/finalize-markdown.sh

# Linting with ruff (rules in pyproject.toml: E, F, I, UP)
uv run ruff check .

# Formatting with black (88 char line length)
uv run black .
```

## Running Python

- Always use `uv run python` to run scripts, tests, or ad-hoc snippets.

## Environment Variables

- `EH_USERNAME` / `EH_PASSWORD` - Login credentials for E-Hentai
- `HBROWSER_LOG_DIR` - Private diagnostics and append-only
  `events-NNNNNN.jsonl` segments; the application selects the exact run
  directory before its one explicit `configure_logging()` lifecycle. Use
  `require_file_sink=False` only when trace loss may be reported as degraded
  through `logging_health()` rather than treated as a startup failure
- `HBROWSER_LOG_FORWARD_ENDPOINT` / `HBROWSER_LOG_FORWARD_TOKEN` are internal,
  one-purpose child capabilities. They are injected only by
  `start_owned_process(..., forward_logging=True)` and consumed by
  `configure_forwarded_logging()`; users must not configure them directly.
- `FLARESOLVERR_URL` - Optional: FlareSolverr 3.5.0+ endpoint (e.g. `http://127.0.0.1:8191/v1`) for automated Cloudflare challenge solving; unset disables it. For local testing, run `docker run -d -p 8191:8191 ghcr.io/flaresolverr/flaresolverr` and set this to `http://127.0.0.1:8191/v1`

## Architecture

### Package Structure

- **`hbrowser`** - Gallery browsing automation
  - `gallery/driver_base.py` - Abstract `Driver` base class with `zendriver` integration, login flow, and CAPTCHA handling
  - `gallery/eh_driver.py` - `EHDriver` for E-Hentai
  - `gallery/exh_driver.py` - `ExHDriver` for ExHentai
  - `gallery/search_models.py` - Public bounded-search and exact-GID lookup request/result models
  - `gallery/captcha/` - CAPTCHA detection plus automatic Turnstile and GUI-manual resolution policy
  - `gallery/browser/` - Browser factory built on `zendriver`; also owns proxy/Tor rotation, persistent FlareSolverr sessions, and ban detection. Use `find hbrowser/gallery/browser -name '*.py'` for the current file list rather than relying on this doc.

### Key Patterns

**Logging composition**: Library modules use `logging.getLogger(__name__)`.
Only the explicit parent composition lifecycle configures the `hbrowser`,
`battle`, `hvbrowser`, and `hvbattle` namespaces and owns append-only segments. Opted-in
Python children enqueue bounded authenticated JSON for a dedicated sender to
forward to the parent; ordinary logger calls perform no file or socket I/O.
Chrome, Tor, and short-lived external helpers never inherit that capability.
Forwarding and optional trace failures are reported through `logging_health()`
without stopping business work; explicit terminal and close boundaries wait
only for their documented bounded drain deadlines.

**Child environments**: Owned processes inherit only the documented OS,
profile/temp, locale/display/XDG, and Python-runtime allowlists. Credentials,
battle/EH control variables, log-directory variables, and arbitrary inherited
state are excluded; reserved values are also stripped from explicit overlays.

**Driver Inheritance**: `Driver` (ABC) → `EHDriver` → `ExHDriver`

**Context Manager**: All drivers support `async with` for automatic login and cleanup.

**Gallery Search**: `EHDriver` and `ExHDriver` accept `SearchRequest` objects.
Initial results and pagination use trusted URL GETs with a main-frame loader
barrier; missing/invalid pagination fails closed. `lookup_gid()` requires two
independent explicit empty searches before returning `ConfirmedGalleryMissing`.

### External Dependencies

- `h2h-galleryinfo-parser` - Gallery metadata parsing
- `zendriver` - CDP-based browser automation

## Coding Guidelines

This is a solo, pre-1.0 project with no external consumers pinned to current APIs. Do not optimize for minimal diffs or backward compatibility:

- Freely rename, restructure, or delete code when it improves the design — there are no external callers to break.
- Do not keep deprecated aliases, compatibility shims, or old code paths "just in case."
- Prefer the cleanest end state over the smallest diff to get there.

Follow SOLID principles when writing code:

- **Single Responsibility** - Each class/module should have one reason to change
- **Open/Closed** - Open for extension, closed for modification (use inheritance/composition)
- **Liskov Substitution** - Subtypes must be substitutable for their base types
- **Interface Segregation** - Prefer small, specific interfaces over large ones
- **Dependency Inversion** - Depend on abstractions (ABC), not concrete implementations

## Code Style

- **Sync obligation for tooling configuration:** the IDE save pipeline and the Stop hook pipeline are kept in lockstep across the locations below. Any change to one of them requires matching updates to the others in the same change.
  - Python formatting/lint/type-check: [.vscode/settings.json](.vscode/settings.json) (`[python]` block), the `[tool.ruff]` section of [pyproject.toml](pyproject.toml), [mypy.ini](mypy.ini), and [scripts/hooks/finalize-python.sh](scripts/hooks/finalize-python.sh).
  - Markdown formatting: [.vscode/settings.json](.vscode/settings.json) (`[markdown]` block) and [scripts/hooks/finalize-markdown.sh](scripts/hooks/finalize-markdown.sh).
  - Tool versions: the `dev` group of `[project.optional-dependencies]` in [pyproject.toml](pyproject.toml) pins `black`, `ruff`, `mypy`, and `pymarkdownlnt`. Both the IDE pipeline (when invoked via `uv run`) and the Stop hooks resolve to these venv-installed versions, so bumping any of them must be done here — not via Homebrew or any other system-wide install.
- Python version range: refer to `requires-python` in [pyproject.toml](pyproject.toml)
