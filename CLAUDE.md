# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HBrowser is a Python library for browser automation on E-Hentai/ExHentai websites and HentaiVerse game. It uses Selenium with undetected-chromedriver and integrates 2Captcha for solving Cloudflare challenges.

## Build & Development Commands

```bash
# Install dependencies (using uv or pip)
uv pip install -e .
# or
pip install -e .

# Type checking (strict mode configured in mypy.ini)
mypy hbrowser hvbrowser

# Linting with ruff (rules in pyproject.toml: E, F, I, UP)
ruff check .

# Formatting with black (88 char line length)
black .
```

## Environment Variables

- `APIKEY_2CAPTCHA` - Required for CAPTCHA solving via 2Captcha service
- `EH_USERNAME` / `EH_PASSWORD` - Login credentials for E-Hentai
- `HBROWSER_LOG_LEVEL` - Optional: DEBUG, INFO, WARNING, ERROR (default: INFO)

## Architecture

### Package Structure

- **`hbrowser`** - Gallery browsing automation
  - `gallery/driver_base.py` - Abstract `Driver` base class with Selenium integration, login flow, and CAPTCHA handling
  - `gallery/eh_driver.py` - `EHDriver` for E-Hentai
  - `gallery/exh_driver.py` - `ExHDriver` for ExHentai
  - `gallery/captcha/` - CAPTCHA detection and solving (supports Cloudflare Turnstile via 2Captcha)
  - `gallery/browser/` - Browser factory with undetected-chromedriver

- **`hvbrowser`** - HentaiVerse game automation
  - `hv.py` - `HVDriver` extends `EHDriver` for HentaiVerse navigation (lottery, monster lab, market)
  - `hv_battle.py` - `BattleDriver` extends `HVDriver` for combat automation
  - `hv_battle_observer_pattern.py` - Observer pattern with `BattleDashboard` and `BattleSubject` for parsing battle state via `hv-bie` library
  - `hv_battle_*_manager.py` - Managers for actions, buffs, skills, items

### Key Patterns

**Driver Inheritance**: `Driver` (ABC) → `EHDriver` → `HVDriver` → `BattleDriver`

**Observer Pattern**: `BattleSubject` notifies `Observer` instances (like `LogEntry`) when battle state updates. `BattleDashboard` coordinates parsing of page source via `hv-bie.parse_snapshot()`.

**Context Manager**: All drivers support `with` statement for automatic login and cleanup.

### External Dependencies

- `hv-bie` - Battle snapshot parsing library
- `h2h-galleryinfo-parser` - Gallery metadata parsing
- `undetected-chromedriver` - Evades bot detection
- `2captcha-python` - CAPTCHA solving API client

## Coding Guidelines

Follow SOLID principles when writing code:

- **Single Responsibility** - Each class/module should have one reason to change
- **Open/Closed** - Open for extension, closed for modification (use inheritance/composition)
- **Liskov Substitution** - Subtypes must be substitutable for their base types
- **Interface Segregation** - Prefer small, specific interfaces over large ones
- **Dependency Inversion** - Depend on abstractions (ABC), not concrete implementations
