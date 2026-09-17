# HBrowser

HBrowser is a Python library for using E-Hentai and ExHentai through an automated
browser. It can sign in, search galleries, look up a gallery by its numeric ID,
submit archive downloads to H@H, and perform the daily check-in.

Use `EHDriver` for E-Hentai or `ExHDriver` for ExHentai. Both provide the same
async interface and open and close their browser with `async with`. HBrowser does
not provide a standalone command-line application.

## Requirements and installation

- Python 3.14 or newer on Windows, macOS, or Linux.
- An E-Hentai account, with ExHentai access if you use `ExHDriver`.
- Internet access. HBrowser installs Chrome for Testing automatically on first
  use unless you supply an existing Chrome executable.
- A graphical desktop when using `headless=False` to handle login challenges
  manually. H@H archive downloads also require an available H@H client and any
  funds required by the service.

Install into your Python environment:

```bash
python -m pip install hbrowser
```

To install this checkout instead, run `python -m pip install .` from its root.

## Set your account and connection

Set these variables in the environment that will run your Python script. Do not
put account credentials in the script or commit them to a repository.

Bash or Zsh:

```bash
export EH_USERNAME='your_username'
export EH_PASSWORD='your_password'
export USE_TOR=0
```

PowerShell:

```powershell
$env:EH_USERNAME = 'your_username'
$env:EH_PASSWORD = 'your_password'
$env:USE_TOR = '0'
```

`USE_TOR=0` selects a direct connection. If the variable is unset, HBrowser
uses Tor when it finds a local Tor executable. See [Connection options](#connection-options)
for proxy and challenge settings.

## Run your first search

Save this as `search_galleries.py`, replace the query, and run
`python search_galleries.py`:

```python
import asyncio

from hbrowser import ExHDriver, SearchRequest


async def main() -> None:
    async with ExHDriver(headless=False) as driver:
        result = await driver.search(
            SearchRequest(
                scope_url="https://exhentai.org/",
                query="language:chinese$",
            )
        )
        for gallery in result.galleries:
            print(gallery.gid, gallery.url)
        print(f"Read {result.pages_visited} search pages")


if __name__ == "__main__":
    asyncio.run(main())
```

The context manager signs in and opens the site's home page before running your
code, then closes the browser when the block exits. Leave the browser visible
for your first run so you can complete an unresolved challenge. For E-Hentai,
replace `ExHDriver` with `EHDriver` and use `https://e-hentai.org/` as the scope.

A search returns up to 5,000 galleries across at most 100 pages. You can lower
those limits with `SearchRequest(max_pages=..., max_results=...)`. A search that
cannot finish within its limits raises `SearchLimitExceededError`; narrow the
query instead of treating the result as a complete list.

## Common tasks

These examples run inside an active `async with` driver block.

### Find a gallery by ID

```python
from hbrowser import ConfirmedGalleryMissing, GalleryFound

match await driver.lookup_gid(349189):
    case GalleryFound(gallery=gallery):
        print(gallery.url)
    case ConfirmedGalleryMissing(confirmations=confirmations):
        print(f"Missing after {confirmations} independent searches")
```

A gallery is reported missing only after two independent empty searches.
Authentication, challenge, navigation, and malformed-page failures raise an
error instead; they do not mean that the gallery was removed.

### Submit an archive download

```python
from h2h_galleryinfo_parser import GalleryURLParser

# Replace this example URL with the gallery you want.
gallery = GalleryURLParser("https://exhentai.org/g/123/456/")
accepted = await driver.download(gallery)
print(f"H@H submission accepted: {accepted}")
```

`True` means H@H accepted the submission, not that the archive has finished
arriving. HBrowser does not return archive bytes or choose a local download
folder. If `ArchiveDownloadOutcomeUnknownError` is raised, check the H@H download
state before retrying: the service may already have accepted the request.

### Perform the daily check-in

```python
from hbrowser import PunchInComplete, RandomEncounterFound

match await driver.punchin():
    case RandomEncounterFound(url=url):
        await driver.get(url)
    case PunchInComplete():
        pass
```

Opening a returned encounter is optional. Encounter URLs are private and
short-lived; do not log or save them. HentaiVerse automation is provided by the
separate [HVBrowser project](https://github.com/Kuan-Lun/hvbrowser).

## Connection options

| Setting | Purpose |
| --- | --- |
| `USE_TOR` | Set `0` to disable Tor or `1` to require it. Unset means auto-detect. |
| `TOR_BINARY_PATH` | Path to your Tor executable when it is not found automatically. |
| `FLARESOLVERR_URL` | Optional FlareSolverr `/v1` endpoint, such as `http://127.0.0.1:8191/v1`. |
| `HBROWSER_CHROME_EXECUTABLE` | Absolute path to an installed, executable Chrome binary. Skips automatic Chrome installation. |

Tor is detected in common Tor Browser installation locations and, on Linux, at
`/usr/bin/tor`. If Tor is requested but unavailable, install it or set
`TOR_BINARY_PATH`; use `USE_TOR=0` if you want a direct connection.

An optional FlareSolverr 3.5.0 or newer instance can handle supported Cloudflare
managed challenges and the Forums login Turnstile widget. Its browser must use
the same public network route as HBrowser. HBrowser disables this integration
when Tor or a residential proxy is active.

With `headless=False`, unresolved challenges stay in the browser window for
manual completion. With `headless=True` (the driver default), they fail because
manual interaction is unavailable. `captcha_manual_timeout` controls the manual
wait in seconds (default: 180); `flaresolverr_session_attempts` controls initial
solver-session attempts (default: 3). Restarting Chrome does not rotate your IP.

## Logs and troubleshooting

To enable console and JSONL file logs, choose a private directory before the
first `configure_logging()` call. Close logging once when your application exits:

```python
import os

from hbrowser import LogLevel, close_logging, configure_logging

os.environ["HBROWSER_LOG_DIR"] = "/path/to/private/run-log"
configure_logging(console_level=LogLevel.INFO, file_level=LogLevel.DEBUG)
try:
    asyncio.run(main())
finally:
    close_logging()
```

This wraps the `main()` from the search example; replace its original
`asyncio.run(main())` call. The directory defaults to `log` beside the main
script. Use a separate directory for concurrent processes, and do not change it
after configuration. Log files are append-only `events-*.jsonl` segments;
HBrowser does not delete old segments, so archive or remove them after the
application has closed logging. A required log-file failure raises
`LogPersistenceError` during setup or when the failure is checked at shutdown.

Browser failures may also save HTML diagnostics. Keep these files private:
they can contain account-specific content. For a problem you detect yourself,
use `await driver.save_page_diagnostic("failure_kind")` while the session is
active.

| Symptom | What to check |
| --- | --- |
| Login or challenge fails | Confirm your credentials and site access, then retry with `headless=False`. For unattended use, check the optional solver endpoint and connection route. |
| Browser cannot start | Check the configured Chrome path, or allow the first-run Chrome download. Visible mode needs a graphical desktop. |
| Search limit exceeded | Narrow the query or choose a more specific search scope. |
| H@H client offline or insufficient funds | Restore the client or account balance before submitting again. |
| Archive outcome unknown | Inspect H@H state before retrying to avoid an accidental duplicate. |
| `ProcessOwnershipError` | Use a supported OS with normal process-management facilities. Minimal POSIX environments need `ps` and Python `os.waitid` support, including `WNOWAIT`. |
| Log setup fails | Use a writable private directory that no other running process owns. |

When reporting a problem, include the package version, OS, exception type, and
whether you used headless mode, Tor, or FlareSolverr. Remove credentials and
account-specific content before sharing logs. Report issues through the
[issue tracker](https://github.com/Kuan-Lun/hbrowser/issues).

## License

Licensed under GPL-3.0-only. See [LICENSE](LICENSE).
