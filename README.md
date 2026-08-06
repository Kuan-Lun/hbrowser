# HBrowser (hbrowser)

## Setup

### Tor Proxy (Optional)

HBrowser can route all traffic through the Tor network for IP privacy. If Tor Browser is installed, HBrowser will automatically detect and use it. To install:

1. Download and install from <https://www.torproject.org/download/>
2. HBrowser will automatically locate the `tor` binary from the default installation path:
   - **macOS**: `/Applications/Tor Browser.app/Contents/MacOS/Tor/tor`
   - **Linux**: `/usr/bin/tor`
   - **Windows**: Searches common installation paths (`Desktop`, `AppData`, `Program Files`)
3. If Tor Browser is not installed, HBrowser will use a direct connection instead.
4. To force disable Tor even when installed, set `USE_TOR=0`.
5. If your Tor Browser is installed in a non-standard location, set the `TOR_BINARY_PATH` environment variable.

### FlareSolverr (Optional)

HBrowser can use [FlareSolverr](https://github.com/FlareSolverr/FlareSolverr) to automatically
solve both Cloudflare's page-level managed challenge and the Turnstile widget embedded in the
Forums login form. Embedded Turnstile support requires FlareSolverr 3.5.0 or newer. Set
`FLARESOLVERR_URL` to the instance's `/v1` endpoint, for example
`http://127.0.0.1:8191/v1`. The endpoint must be a valid HTTP or HTTPS URL;
invalid configuration fails immediately with a sanitized configuration error.

HBrowser keeps one persistent FlareSolverr browser across the managed challenge and login
Turnstile so both steps use the same browser identity and clearance. Before that identity is
applied to the main browser, a failed FlareSolverr request causes the failed FlareSolverr
session to be discarded and a fresh session to be tried. The default budget is three session
attempts and can be changed with the `flaresolverr_session_attempts` driver argument.

Once a FlareSolverr identity has been applied, HBrowser does not silently replace its solver
session: a new session would no longer match the identity already installed in the main
browser. In a visible browser, a runtime solver failure falls back to manual resolution in that
same browser window. In headless mode, it fails immediately with a login error because no
manual interaction is possible.

Restarting Chrome is not a proxy or IP rotation, so challenge recovery never restarts the main
browser and never reports a route change. A direct connection has no route-rotation capability.
The FlareSolverr identity must share the main browser's public route, so HBrowser disables
FlareSolverr when Tor or a residential proxy is active; those configurations currently have no
shared sticky-route integration.

HBrowser 0.36 removes the `proxy_rotator` and `max_captcha_retries` driver
arguments. They coupled challenge handling to a Chrome restart that could not
verify any route change. Use `flaresolverr_session_attempts` to configure the
independent solver-session retry budget instead.

### Environment Variables

HBrowser requires the following environment variables:

- `EH_USERNAME`: Your E-Hentai account username
- `EH_PASSWORD`: Your E-Hentai account password
- `HBROWSER_LOG_LEVEL` (optional): Control logging verbosity (DEBUG, INFO, WARNING, ERROR). Default: INFO
- `HBROWSER_LOG_DIR` (optional): Store HTML failure diagnostics in this
  directory. Default: a `log` directory next to the main script
- `HBROWSER_PROCESS_LOG_FILE` (optional): Mirror all HBrowser-family logger
  records to this UTF-8 file. The file rotates at 10 MiB with five backups;
  applications should leave it unset when an external supervisor already
  captures stdout/stderr
- `USE_TOR` (optional): Set to `0` to disable Tor proxy even when Tor Browser is installed. Default: auto-detect
- `TOR_BINARY_PATH` (optional): Custom path to the `tor` binary if not installed in the default location
- `FLARESOLVERR_URL` (optional): FlareSolverr 3.5.0+ `/v1` endpoint used to auto-solve Cloudflare managed challenges and the Forums login Turnstile. Ignored when Tor or a residential proxy is active

Set the environment variables before running the script:

**Bash/Zsh:**

```bash
export EH_USERNAME=your_username
export EH_PASSWORD=your_password
export HBROWSER_LOG_LEVEL=INFO          # Optional
export HBROWSER_LOG_DIR=/path/to/log    # Optional: shared diagnostic directory
export USE_TOR=0                        # Optional: disable Tor proxy
export TOR_BINARY_PATH=/path/to/tor     # Optional: custom tor path
export FLARESOLVERR_URL=http://127.0.0.1:8191/v1  # Optional: auto-solve Cloudflare
```

**Fish:**

```fish
set -x EH_USERNAME your_username
set -x EH_PASSWORD your_password
set -x HBROWSER_LOG_LEVEL INFO          # Optional
set -x USE_TOR 0                        # Optional: disable Tor proxy
set -x TOR_BINARY_PATH /path/to/tor     # Optional: custom tor path
set -x FLARESOLVERR_URL http://127.0.0.1:8191/v1  # Optional: auto-solve Cloudflare
```

**Windows Command Prompt:**

```cmd
set EH_USERNAME=your_username
set EH_PASSWORD=your_password
set HBROWSER_LOG_LEVEL=INFO
set USE_TOR=0
set TOR_BINARY_PATH=C:\path\to\tor.exe
set FLARESOLVERR_URL=http://127.0.0.1:8191/v1
```

**Windows PowerShell:**

```powershell
$env:EH_USERNAME="your_username"
$env:EH_PASSWORD="your_password"
$env:HBROWSER_LOG_LEVEL="INFO"
$env:USE_TOR="0"
$env:TOR_BINARY_PATH="C:\path\to\tor.exe"
$env:FLARESOLVERR_URL="http://127.0.0.1:8191/v1"
```

When a challenge appears during login, HBrowser tries FlareSolverr for supported managed
challenges and Turnstile widgets when it is configured and route-compatible. With
`headless=False`, any unresolved challenge is left in the current browser window for manual
completion. With `headless=True`, an unresolved challenge raises a login error immediately;
HBrowser does not restart Chrome or claim that a proxy was rotated.

## Logging

HBrowser uses Python's built-in `logging` module. You can control the log level using the `HBROWSER_LOG_LEVEL` environment variable:

- **DEBUG**: Detailed information for diagnosing problems (most verbose)
- **INFO**: Confirmation that things are working as expected (default)
- **WARNING**: Something unexpected happened, but the software is still working
- **ERROR**: A serious problem that prevented a function from executing

Example:

```bash
# Set log level to DEBUG for detailed output
export HBROWSER_LOG_LEVEL=DEBUG
python your_script.py

# Set log level to WARNING to see only warnings and errors
export HBROWSER_LOG_LEVEL=WARNING
python your_script.py
```

Applications without an external process-log supervisor can also request a
rotating text file for HBrowser-family logger records:

```bash
export HBROWSER_PROCESS_LOG_FILE=/private/path/battle.log
```

One file handler is shared by all HBrowser, HVBrowser, and HVBattle loggers in
the process, so each record is written once. Existing targets must be regular,
single-link files rather than symbolic links. The handler verifies that the
configured path still names its open file before and after each record. Where
the platform provides them, opening also uses `O_NOFOLLOW` and `O_CLOEXEC`; on
POSIX, active and rotated files use owner-only permissions. Open, write, path
replacement, and rollover failures are raised to the logging caller instead of
being silently ignored.

This optional handler records logger output only. It does not capture arbitrary
stdout/stderr, subprocess output, or exceptions raised before logging is
configured, and it does not provide an `fsync`-based process transcript. Use an
external checked supervisor when those guarantees or terminal fail-closed
behavior are required. The parent directory remains part of the application's
trust boundary, and Windows file confidentiality depends on its inherited ACLs.

On browser failures, HBrowser saves uniquely named HTML diagnostics instead of
overwriting a single `error.txt`. Page diagnostics and search diagnostics each
retain at most 20 files and 20 MiB total, with each file capped at 2 MiB. On
POSIX systems these files are created with owner-only permissions. Because page
HTML can still contain account-specific data, keep `HBROWSER_LOG_DIR` in a
private location.

## Usage

Daily check-in returns an explicit outcome. A random encounter URL is accepted
only when it is the unique trusted HentaiVerse battle link in the E-Hentai
event pane:

```python
from hbrowser import PunchInComplete, RandomEncounterFound

result = await driver.punchin()
match result:
    case RandomEncounterFound(url=url):
        # The caller decides whether and when to navigate to this encounter.
        await driver.get(url)
    case PunchInComplete():
        pass
```

The returned encounter URL is sensitive, short-lived navigation state. Avoid
logging or persisting it. `hbrowser` never writes the raw value to its logs and
redacts the `encounter` query value from page diagnostics. If an event pane has
multiple encounter links or encounter-like markup that is not a trusted URL,
`punchin()` raises instead of reporting a completed check-in.

Gallery searches use an explicit request and return a bounded, ordered result.
An exact GID lookup only reports a missing gallery after two independent empty
searches:

```python
import asyncio

from hbrowser import (
    ConfirmedGalleryMissing,
    ExHDriver,
    GalleryFound,
    SearchRequest,
)


async def main() -> None:
    async with ExHDriver() as driver:
        result = await driver.search(
            SearchRequest(
                scope_url="https://exhentai.org/",
                query="artist:test",
            )
        )
        print(result.galleries, result.pages_visited)

        match await driver.lookup_gid(349189):
            case GalleryFound(gallery=gallery):
                print(gallery.url)
            case ConfirmedGalleryMissing(confirmations=confirmations):
                print(f"Missing after {confirmations} confirmations")


if __name__ == "__main__":
    asyncio.run(main())
```

HentaiVerse automation is provided separately by
[HVBrowser](https://github.com/Kuan-Lun/hvbrowser).
