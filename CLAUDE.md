# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HBrowser is a Python library for browser automation on E-Hentai/ExHentai websites and HentaiVerse game. It uses `zendriver` (Chrome DevTools Protocol) for browser automation. Cloudflare managed challenges can be solved automatically via FlareSolverr; CAPTCHA challenges still require manual user interaction.

## Communication

- Claude 必須以繁體中文回答所有對話內容，不論使用者以何種語言提問；程式碼、指令、檔名、專有名詞等仍維持原文。

## Build & Development Commands

```bash
# Install dependencies
uv pip install -e .

# Type checking (strict mode configured in mypy.ini)
uv run mypy hbrowser hvbrowser

# Linting with ruff (rules in pyproject.toml: E, F, I, UP)
uv run ruff check .

# Formatting with black (88 char line length)
uv run black .
```

## Running Python

- Always use `uv run python` to run scripts, tests, or ad-hoc snippets.

## Environment Variables

- `EH_USERNAME` / `EH_PASSWORD` - Login credentials for E-Hentai
- `HBROWSER_LOG_LEVEL` - Optional: DEBUG, INFO, WARNING, ERROR (default: INFO)
- `FLARESOLVERR_URL` - Optional: FlareSolverr endpoint (e.g. `http://127.0.0.1:8191/v1`) for automated Cloudflare challenge solving; unset disables it. For local testing, run `docker run -d -p 8191:8191 ghcr.io/flaresolverr/flaresolverr` and set this to `http://127.0.0.1:8191/v1`

## Architecture

### Package Structure

- **`hbrowser`** - Gallery browsing automation
  - `gallery/driver_base.py` - Abstract `Driver` base class with `zendriver` integration, login flow, and CAPTCHA handling
  - `gallery/eh_driver.py` - `EHDriver` for E-Hentai
  - `gallery/exh_driver.py` - `ExHDriver` for ExHentai
  - `gallery/captcha/` - CAPTCHA detection (manual user resolution)
  - `gallery/browser/` - Browser factory built on `zendriver`; also owns proxy/Tor rotation, FlareSolverr-based Cloudflare auto-solving, and ban detection. Use `find hbrowser/gallery/browser -name '*.py'` for the current file list rather than relying on this doc.

- **`hvbrowser`** - HentaiVerse game automation
  - `hv.py` - `HVDriver` extends `EHDriver` for HentaiVerse navigation (lottery, monster lab, market)
  - `hv_battle.py` - `BattleDriver` extends `HVDriver` for combat automation
  - `hv_battle_observer_pattern.py` - Observer pattern with `BattleDashboard` and `BattleSubject` for parsing battle state via `hv-bie` library
  - `hv_battle_*_manager.py` / `hv_battle_*_provider.py` - Managers for actions, buffs, skills, items

### Key Patterns

**Driver Inheritance**: `Driver` (ABC) → `EHDriver` → `HVDriver` → `BattleDriver`

**Observer Pattern**: `BattleSubject` notifies `Observer` instances (like `LogEntry`) when battle state updates. `BattleDashboard` coordinates parsing of page source via `hv-bie.parse_snapshot()`.

**Context Manager**: All drivers support `with` statement for automatic login and cleanup.

### External Dependencies

- `hv-bie` - Battle snapshot parsing library
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
  - Python formatting/lint/type-check: [.vscode/settings.json](.vscode/settings.json) (`[python]` block), the `[tool.ruff]` section of [pyproject.toml](pyproject.toml), [mypy.ini](mypy.ini), and [.claude/hooks/finalize-python.sh](.claude/hooks/finalize-python.sh).
  - Markdown formatting: [.vscode/settings.json](.vscode/settings.json) (`[markdown]` block) and [.claude/hooks/finalize-markdown.sh](.claude/hooks/finalize-markdown.sh).
  - Tool versions: the `dev` group of `[project.optional-dependencies]` in [pyproject.toml](pyproject.toml) pins `black`, `ruff`, `mypy`, and `pymarkdownlnt`. Both the IDE pipeline (when invoked via `uv run`) and the Stop hooks resolve to these venv-installed versions, so bumping any of them must be done here — not via Homebrew or any other system-wide install.
- Python version range: refer to `requires-python` in [pyproject.toml](pyproject.toml)
