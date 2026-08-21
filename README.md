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
- `HBROWSER_LOG_DIR` (optional): Store HTML failure diagnostics in this
  directory. Default: a `log` directory next to the main script
- `HBROWSER_CAPTURE_PUNCHIN_PAGES` (optional): Set to `1`, `true`, `yes`, or
  `on` to save the initial and, when needed, reloaded daily check-in documents
  in `HBROWSER_LOG_DIR`. Encounter query values are redacted, but the remaining
  HTML is account-specific and must be kept private
- `HBROWSER_PROCESS_LOG_FILE` (optional): Select the private UTF-8 application
  log owned by HBrowser's Python file handler. The default configuration writes
  DEBUG and above to this file and rotates it at 10 MiB with five backups
- `USE_TOR` (optional): Set to `0` to disable Tor proxy even when Tor Browser is installed. Default: auto-detect
- `TOR_BINARY_PATH` (optional): Custom path to the `tor` binary if not installed in the default location
- `FLARESOLVERR_URL` (optional): FlareSolverr 3.5.0+ `/v1` endpoint used to auto-solve Cloudflare managed challenges and the Forums login Turnstile. Ignored when Tor or a residential proxy is active

Set the environment variables before running the script:

**Bash/Zsh:**

```bash
export EH_USERNAME=your_username
export EH_PASSWORD=your_password
export HBROWSER_LOG_DIR=/path/to/log    # Optional: shared diagnostic directory
export USE_TOR=0                        # Optional: disable Tor proxy
export TOR_BINARY_PATH=/path/to/tor     # Optional: custom tor path
export FLARESOLVERR_URL=http://127.0.0.1:8191/v1  # Optional: auto-solve Cloudflare
```

**Fish:**

```fish
set -x EH_USERNAME your_username
set -x EH_PASSWORD your_password
set -x USE_TOR 0                        # Optional: disable Tor proxy
set -x TOR_BINARY_PATH /path/to/tor     # Optional: custom tor path
set -x FLARESOLVERR_URL http://127.0.0.1:8191/v1  # Optional: auto-solve Cloudflare
```

**Windows Command Prompt:**

```cmd
set EH_USERNAME=your_username
set EH_PASSWORD=your_password
set USE_TOR=0
set TOR_BINARY_PATH=C:\path\to\tor.exe
set FLARESOLVERR_URL=http://127.0.0.1:8191/v1
```

**Windows PowerShell:**

```powershell
$env:EH_USERNAME="your_username"
$env:EH_PASSWORD="your_password"
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

HBrowser uses Python's built-in `logging` module with independent console and
application-file thresholds. By default, the console displays INFO and above,
while a configured application file records DEBUG and above. Configure the
process explicitly with `configure_logging()`:

```python
from hbrowser import LogLevel, configure_logging

configure_logging(
    console_level=LogLevel.INFO,
    file_level=LogLevel.DEBUG,
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
)
```

`LogLevel` supports `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`.
`configure_logging()` immediately updates loggers that HBrowser, HVBrowser,
HVBattle, or the application already created. When no application file is
configured, only the console threshold determines the effective logger level.
When a logger name is first registered, `setup_logger()` rejects a pre-existing
unmanaged stream or file handler without modifying that logger. Handlers added
after registration by test instrumentation are preserved; process
reconfiguration only updates HBrowser-managed handlers.

Set the application-log path before importing HBrowser-family packages when
the earliest import-time records must also be retained:

```bash
export HBROWSER_PROCESS_LOG_FILE=/private/path/battle.log
```

One rotating file handler is shared by all managed loggers in the process, so
each record is written once. The active file plus the five default backups use
approximately 60 MiB at most, aside from a record that crosses a rotation
boundary. Existing targets must be regular, single-link files rather than
symbolic links. The handler verifies that the configured path still names its
open file before and after each record. Where the platform provides them,
opening also uses `O_NOFOLLOW` and `O_CLOEXEC`; on POSIX, active and rotated
files use owner-only permissions. Before and after rollover, every configured
backup slot is checked as a regular, single-link, non-symlink file and secured
to mode `0600` on POSIX. A successful rollover prunes the oldest backup in the
configured window. Lowering `backup_count` does not delete older suffixes that
are already outside the new window; application-level run retention should
remove those after the process no longer owns the log.

Each record is flushed from Python's stream before the logging call returns;
there is deliberately no multi-megabyte in-memory batch that could disappear
with the final diagnostics during a crash. This is not an `fsync` guarantee,
so the operating system may still hold data in its page cache. Configure file
rotation or application-level run retention to bound disk usage rather than
using a write buffer.

Open, write, path replacement, rollover, and reconfiguration-close failures
raise the public `LogPersistenceError`. Supervisors should classify this
exception as terminal instead of retrying uncertain work. When switching paths,
the new sink is attached and the old sink is detached before the old sink is
closed. If that close fails, the new sink remains usable and the old handler
remains strongly owned for a later cleanup attempt, but the failed
reconfiguration is still terminal for the current command. The Python handler
must be the only writer to its configured file; a supervisor may allocate and
export the path, but must not concurrently append or `tee` output into that same
file.

When a command boundary must print a machine-readable terminal result without
duplicating it through the console formatter, it can persist a separately
formatted file record explicitly:

```python
from hbrowser import LogLevel, log_to_process_file

log_to_process_file(logger, LogLevel.ERROR, terminal_json)
print(terminal_json)
```

`logger` must come from `setup_logger()`. The helper respects the configured
file threshold, never emits through the console handler, and is a no-op when no
process file is configured.

This optional handler records logger output only. It does not capture arbitrary
stdout/stderr, subprocess output, or exceptions raised before logging is
configured. Applications should route terminal status and uncaught exceptions
through a managed logger when those records belong in the application log. The
parent directory remains part of the application's trust boundary, and Windows
file confidentiality depends on its inherited ACLs.

On browser failures, HBrowser saves uniquely named HTML diagnostics instead of
overwriting a single `error.txt`. Page diagnostics and search diagnostics each
retain at most 20 files and 20 MiB total, with each file capped at 2 MiB. On
POSIX systems these files are created with owner-only permissions. Because page
HTML can still contain account-specific data, keep `HBROWSER_LOG_DIR` in a
private location.

Browser clients can persist the current page source at an application-defined
failure boundary with `await driver.save_page_diagnostic("failure_kind")`. The
same redaction, private-file, size, and retention rules apply.

Chrome and Tor are launched through a start-gated supervisor owned by HBrowser,
not through Zendriver's global process hook and not in the application's
terminal group. On POSIX, the supervisor owns a new session and the target owns
a distinct process group inside it. Shutdown signals that target group, proves
that it is empty, and only then releases its private files. On Windows, the
supervisor is assigned to a non-inheritable Job Object before it is allowed to
spawn the target; kill-on-close and active-process accounting cover every
descendant. Missing Job APIs or failed assignment abort startup instead of
falling back to an unowned process.

A terminal interrupt can therefore be converted into an application-level
cooperative stop without killing Chrome while a CDP mutation receipt is still
being confirmed. Each generation uses an HBrowser-owned Chrome profile, and
authenticated proxy material is removed only after process-tree termination is
proven. The application remains responsible for closing its Browser owner after
reaching a safe boundary. HBrowser pins Zendriver 0.16.0 because the
connect-existing lifecycle contract is verified against that exact release.

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

`punchin()` checks the initial news document before reloading it. A trusted
encounter is returned immediately so a second navigation cannot discard it.
When the initial document has no encounter, the historical reload fallback is
retained and the reloaded document is checked independently.

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
