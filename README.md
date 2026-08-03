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
`http://127.0.0.1:8191/v1`.

HBrowser keeps one persistent FlareSolverr browser across the managed challenge and login
Turnstile so both steps use the same browser identity and clearance. That identity must also
share the automated browser's public route. HBrowser therefore disables FlareSolverr when Tor
or a residential proxy is active unless a future integration can configure the same sticky
route on both browsers.

### Environment Variables

HBrowser requires the following environment variables:

- `EH_USERNAME`: Your E-Hentai account username
- `EH_PASSWORD`: Your E-Hentai account password
- `HBROWSER_LOG_LEVEL` (optional): Control logging verbosity (DEBUG, INFO, WARNING, ERROR). Default: INFO
- `HBROWSER_LOG_DIR` (optional): Store HTML failure diagnostics in this
  directory. Default: a `log` directory next to the main script
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

When a Cloudflare or CAPTCHA challenge appears during login, HBrowser will first try
FlareSolverr (if configured and applicable), then fall back to waiting for you to solve it
manually in the browser window. Set `headless=False` when initialising the driver to see the
browser window.

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

On browser failures, HBrowser saves uniquely named HTML diagnostics instead of
overwriting a single `error.txt`. Page diagnostics and search diagnostics each
retain at most 20 files and 20 MiB total, with each file capped at 2 MiB. On
POSIX systems these files are created with owner-only permissions. Because page
HTML can still contain account-specific data, keep `HBROWSER_LOG_DIR` in a
private location.

## Usage

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
