# HBrowser (hbrowser)

## Supported platforms

HBrowser requires Python 3.14 or newer. The declared owned-process targets are
Windows, macOS, and Linux. A POSIX runtime must expose Python's non-reaping
child-exit receipt primitives (`os.waitid`, `P_PID`, `WEXITED`, `WNOHANG`, and
`WNOWAIT`) and a `ps -axo pid=,pgid=` process-group snapshot. A runtime without
either requirement is rejected before the browser target starts;
falling back to `Popen.poll()` would reap the child and make a cached process
group vulnerable to PID reuse.

On POSIX, the target and every descendant must remain in the process group that
HBrowser assigns until exit. `start_owned_process()` is process ownership, not
an OS containment sandbox, and does not support targets that call `setsid()` or
otherwise detach descendants. The owned browser/runtime must also stop creating
descendants after it receives termination or after its group leader exits. The
`ps` snapshot used for final proof is not an atomic containment primitive, so a
target that continuously or adversarially forks during shutdown is outside this
contract. Windows ownership uses a Job object instead.

## Owned-process shutdown

`OwnedProcess.terminate()` requests graceful termination only. It does not
silently escalate to a force kill after an internal timer; callers that need a
bounded TERM-to-KILL policy should use `OwnedProcess.shutdown()`, while an
explicit force request uses `OwnedProcess.kill()`. If a caller deadline expires,
the supervisor retains the target identity and continues proof-only settlement
instead of abandoning an unproven ownership set.

## Setup

### Tor Proxy (Optional)

HBrowser can route all traffic through the Tor network for IP privacy. If Tor
Browser is installed, HBrowser will automatically detect and use it. To install:

1. Download and install from <https://www.torproject.org/download/>
2. HBrowser will automatically locate the `tor` binary from the default installation path:
   - **macOS**: `/Applications/Tor Browser.app/Contents/MacOS/Tor/tor`
   - **Linux**: `/usr/bin/tor`
   - **Windows**: Searches common installation paths (`Desktop`, `AppData`, `Program Files`)
3. If Tor Browser is not installed, HBrowser will use a direct connection instead.
4. To force disable Tor even when installed, set `USE_TOR=0`.
5. If your Tor Browser is installed in a non-standard location, set the
   `TOR_BINARY_PATH` environment variable.

### FlareSolverr (Optional)

HBrowser can use
[FlareSolverr](https://github.com/FlareSolverr/FlareSolverr) to automatically
solve both Cloudflare's page-level managed challenge and the Turnstile widget
embedded in the Forums login form. Embedded Turnstile support requires
FlareSolverr 3.5.0 or newer. Set
`FLARESOLVERR_URL` to the instance's `/v1` endpoint, for example
`http://127.0.0.1:8191/v1`. The endpoint must be a valid HTTP or HTTPS URL;
invalid configuration fails immediately with a sanitized configuration error.

HBrowser keeps one persistent FlareSolverr browser across the managed challenge
and login Turnstile so both steps use the same browser identity and clearance.
Before that identity is applied to the main browser, a failed FlareSolverr
request causes the failed FlareSolverr session to be discarded and a fresh
session to be tried. The default budget is three session attempts and can be
changed with the `flaresolverr_session_attempts` driver argument.

Once a FlareSolverr identity has been applied, HBrowser does not silently
replace its solver session: a new session would no longer match the identity
already installed in the main browser. In a visible browser, a runtime solver
failure falls back to manual resolution in that same browser window. In
headless mode, it fails immediately with a login error because no manual
interaction is possible.

Restarting Chrome is not a proxy or IP rotation, so challenge recovery never
restarts the main browser and never reports a route change. A direct connection
has no route-rotation capability. The FlareSolverr identity must share the main
browser's public route, so HBrowser disables FlareSolverr when Tor or a
residential proxy is active; those configurations currently have no shared
sticky-route integration.

HBrowser 0.36 removes the `proxy_rotator` and `max_captcha_retries` driver
arguments. They coupled challenge handling to a Chrome restart that could not
verify any route change. Use `flaresolverr_session_attempts` to configure the
independent solver-session retry budget instead.

### Environment Variables

HBrowser requires the following environment variables:

- `EH_USERNAME`: Your E-Hentai account username
- `EH_PASSWORD`: Your E-Hentai account password
- `HBROWSER_LOG_DIR` (optional): Store private JSONL application-log segments
  and HTML failure diagnostics in this directory. Set it to the exact run
  directory before calling `configure_logging()`. Default: a `log` directory
  next to the main script
- `HBROWSER_CAPTURE_PUNCHIN_PAGES` (optional): Set to `1`, `true`, `yes`, or
  `on` to save the initial and, when needed, reloaded daily check-in documents
  in `HBROWSER_LOG_DIR`. Encounter query values are redacted, but the remaining
  HTML is account-specific and must be kept private
- `HBROWSER_CHROME_EXECUTABLE` (optional): Absolute path to a preinstalled Chrome
  executable managed by the application or container image. A package-manager
  symlink to an executable regular file is accepted. When set, HBrowser skips
  Chrome for Testing installation and metadata requests; it still owns the
  browser process, temporary profile, and cleanup. An empty, relative, missing,
  or non-executable path fails before browser or proxy startup. When unset,
  HBrowser keeps its automatic Chrome for Testing installation behavior
- `USE_TOR` (optional): Set to `0` to disable Tor proxy even when Tor
  Browser is installed. Default: auto-detect
- `TOR_BINARY_PATH` (optional): Custom path to the `tor` binary if not
  installed in the default location
- `FLARESOLVERR_URL` (optional): FlareSolverr 3.5.0+ `/v1` endpoint used
  to auto-solve Cloudflare managed challenges and the Forums login Turnstile.
  Ignored when Tor or a residential proxy is active

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

`Driver.login()` owns only the bounded Forums authentication phase and remains
on the verified Forums page. The driver context manager then performs exactly
one homepage navigation. Higher-level domain sessions that initialize the
browser themselves likewise own their first post-login destination, so they can
verify it before any later navigation.

## Logging

HBrowser uses Python's built-in `logging` module with independent console and
application-file thresholds. Library modules use ordinary
`logging.getLogger(__name__)`; importing them does not register a logger, add a
handler, create a directory, or open a sink. `setup_logger()` remains available
for composition roots and application-owned logger names and is likewise
side-effect-free until explicit configuration.
The application must select `HBROWSER_LOG_DIR` and explicitly configure its
single process-owned writer before expecting any records to be emitted:

```python
from hbrowser import LogLevel, configure_logging

configure_logging(
    console_level=LogLevel.INFO,
    file_level=LogLevel.DEBUG,
    segment_bytes=10 * 1024 * 1024,
    require_file_sink=True,
)
```

`LogLevel` supports `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`.
`configure_logging()` immediately attaches one console handler and the same
file handler to the `battle`, `hbrowser`, `hvbrowser`, and `hvbattle` namespace
loggers
and to application names registered with `setup_logger()`. Descendant library
loggers inherit exactly one namespace handler. A registered application logger
added later joins the configuration. Reconfiguration may change thresholds and
`segment_bytes`, but the process remains permanently pinned to its initially
selected directory and handler. Changing `HBROWSER_LOG_DIR` afterward is an
error. The directory must therefore be chosen before the first call, not merely
before package import.

`require_file_sink=True` makes directory selection and initial sink ownership a
startup requirement. With `False`, HBrowser still tries once to open the file
sink, but a failure is latched as degraded trace health while console logging
remains configured and usable. The process never retries or changes ownership
on a later reconfiguration. Use the optional mode only when losing diagnostic
trace output is acceptable; durable audit state needs a separate required
store.

The writer exclusively creates append-only files named
`events-000001.jsonl`, `events-000002.jsonl`, and so on. A new process validates
all matching existing entries, selects the maximum sequence plus one, and
claims that exact name with exclusive creation. A competing owner causes
configuration to fail; it does not silently share a file or skip to another
number. A lifecycle-long operating-system lock on the private
`.hbrowser-log-writer.lock` marker enforces one writer for the run directory;
the persistent marker can be reacquired by a later sequential process after
the prior writer closes or exits. Existing and active segments must remain
regular, single-link, non-reparse files, and the selected directory must remain
the same non-reparse directory. POSIX creation is pinned through a directory
descriptor and segment permissions are `0600`.

When the encoded next record would cross `segment_bytes`, the current segment
is flushed and closed and the next sequence is exclusively created. A record
that exactly fills a segment remains there; the following record starts the
next segment. A single oversized record is kept intact in one segment. HBrowser
never renames, reopens, overwrites, prunes, or deletes a segment. Run-level
archive and retention code may operate only after `close_logging()` succeeds.

Each JSONL record contains UTC timestamp, severity, logger, rendered message,
semantic label, the private `account`, `realm`, `tab_role`, `activity`, and
`scope` fields, process/thread identifiers, and exception or stack data when
present. Console output retains the concise timestamp/severity/semantic-label
format.

Ordinary logger calls encode the record and use a non-blocking put into a
bounded 512-record queue. A single dedicated daemon writer owns every segment
write, roll, flush, and close, and flushes each accepted record from Python's
stream before taking the next one. A full queue is a trace-persistence failure;
the logger call latches degraded health instead of waiting for filesystem I/O.
This is not an `fsync` guarantee, so the operating system may still hold data
in its page cache. Configure file segment sizing and application-level run
retention to bound disk usage rather than treating the queue as a durable
buffer.

Required configuration and initial-open failures raise `LogPersistenceError`
immediately; optional sink failures follow the degraded-health behavior above.
An ordinary logger call never performs file or socket I/O and never lets a
queue, write, flush, segment-open, or segment-close failure escape through
business logic. Instead, the first failure is latched, the file sink is
permanently disabled, and later calls to
`raise_for_log_persistence_failure()` raise that same error at an application
safe boundary. `logging_health()` provides the same path-free state without
raising: its `trace_degraded` flag and `operation`, `error_type`, numeric
`errno`/`winerror`, writer PID, segment sequence, and segment name fields are
safe for manifests and emergency channels. The rendered exception also omits
paths and operating-system messages. The original exception is retained as
`__cause__` for trusted in-process inspection.

The first latched failure also writes one compact JSON object directly to the
original stderr stream without passing through `logging`. It contains only the
event name, bounded persistence `stage`, and the safe diagnostic fields above;
it never contains the exception message or a path. Failure of this emergency
write is swallowed, never recurses, and never replaces the latched health
state.

Parent processes can call `start_log_forwarding_receiver()` after
`configure_logging()`. A healthy file sink starts an authenticated receiver on
an ephemeral `127.0.0.1` port and returns its lifecycle handle. The capability
uses a random 256-bit token and a strictly bounded, length-prefixed canonical
JSON protocol; it never accepts pickle, a serialized `LogRecord`, non-loopback
traffic, unknown fields, or unbounded strings. Validated child records go
directly to the parent's existing JSONL handler, so neither the receiver nor a
child opens another `events-*` writer.

Receiver bind/unavailability and optional trace-sink failures are diagnostics,
not business-startup gates: the start call returns `None`, latches
`forwarding_degraded`, and emits the same style of first-only path-free
emergency JSON. The additional `LoggingHealth` forwarding fields report its
first safe operation, error type, numeric error codes, and PID. Unauthenticated
port noise does not degrade health; an authenticated peer's protocol, receive,
drain, or sink failure does.

An explicitly selected Python child is started with
`start_owned_process(..., forward_logging=True)` and calls
`configure_forwarded_logging()` before ordinary logger use, followed by
`close_forwarded_logging()` at its lifecycle boundary. The child consumes and
removes the endpoint/token from its environment, installs only a bounded
asynchronous forwarding queue, and never opens a console or file sink. Its
sender thread alone owns socket connect, authentication, send, and close.
Missing/refused connections, queue exhaustion, and send/close failures latch
child forwarding health and never escape ordinary logging or change the
child's business result. Child close waits at most 0.5 seconds for its sender;
children still launch without the capability when the parent receiver is
unavailable. `close_logging()` first stops accepting and gives all receiver
clients one shared two-second drain deadline, then boundedly closes the segment
writer, preventing a late forwarded writer.

When a command boundary must print a machine-readable terminal result without
duplicating it through the console formatter, it can persist a separately
formatted file record explicitly:

```python
from hbrowser import (
    LogLevel,
    close_logging,
    log_to_process_file,
    logging_health,
    raise_for_log_persistence_failure,
)

log_to_process_file(logger, LogLevel.ERROR, terminal_json)
trace_health = logging_health()
raise_for_log_persistence_failure()
close_logging()
print(terminal_json)
```

`logger` must be the canonical logger returned by `logging.getLogger()` for a
configured namespace or one of its descendants, or come from `setup_logger()`.
The helper respects the configured file threshold, never emits through the
console handler, and is a no-op before explicit configuration. Unlike ordinary
logger calls, it waits up to the writer's two-second lifecycle budget for all
accepted records and immediately surfaces an unhealthy sink, so a terminal
record cannot be assumed persisted merely because it entered the queue.

`close_logging()` is the one-time process lifecycle boundary. It detaches the
sinks, boundedly drains them, and then reports any latched persistence failure.
If a receiver callback or writer I/O does not stop before its deadline, the
descriptors are deliberately left to process-exit reclamation; the launcher
cannot archive the run until that owner process has exited. A healthy repeated
close is a no-op; after a failure, repeated closes raise the same first error.
Once closed, logger registration, configuration, and direct file writes are
rejected for the rest of the process. Python's automatic `logging.shutdown()`
is not a substitute: it runs too late for the application to classify final
flush or close failures in its exit status.

The file handler records logger output only. It does not capture arbitrary
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

Every owned target receives a minimal environment rather than a copy of the
parent environment. Inherited values are allowlisted to necessary Windows
system/profile/temp values or POSIX path/home/temp/locale/display/XDG values,
plus explicit Python-runtime settings. Credentials, arbitrary sentinels,
`BATTLE_*`, `EH_*`, `HBROWSER_LOG_DIR`, the legacy process-log variable, and
forwarding capabilities are excluded. A caller may supply explicit application
values, but reserved keys are still stripped. Only `forward_logging=True`
injects a fresh endpoint/token after sanitization; the Chrome and Tor launch
paths cannot receive that capability. Short-lived external helpers such as
desktop notification, audio playback, `ditto`, and `xattr` also receive a copy
of their normal environment with the log directory, legacy sink, endpoint, and
token removed case-insensitively.

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

## License

This project is licensed under GPL-3.0-only. See [LICENSE](LICENSE).
