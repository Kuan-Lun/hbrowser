# HBrowser (hbrowser)

## Setup

### Prerequisites

HBrowser routes all traffic through the Tor network for IP privacy. You need to install **Tor Browser** before using HBrowser:

1. Download and install from https://www.torproject.org/download/
2. HBrowser will automatically locate the `tor` binary from the default installation path:
   - **macOS**: `/Applications/Tor Browser.app/Contents/MacOS/Tor/tor`
   - **Linux**: `/usr/bin/tor`
   - **Windows**: Searches common installation paths (`Desktop`, `AppData`, `Program Files`)
3. If your Tor Browser is installed in a non-standard location, set the `TOR_BINARY_PATH` environment variable.

### Environment Variables

HBrowser requires the following environment variables:

- `EH_USERNAME`: Your E-Hentai account username
- `EH_PASSWORD`: Your E-Hentai account password
- `APIKEY_2CAPTCHA`: Your 2Captcha API key for solving CAPTCHA challenges
- `HBROWSER_LOG_LEVEL` (optional): Control logging verbosity (DEBUG, INFO, WARNING, ERROR). Default: INFO
- `TOR_BINARY_PATH` (optional): Custom path to the `tor` binary if not installed in the default location

Set the environment variables before running the script:

**Bash/Zsh:**
```bash
export EH_USERNAME=your_username
export EH_PASSWORD=your_password
export APIKEY_2CAPTCHA=your_api_key_here
export HBROWSER_LOG_LEVEL=INFO          # Optional
export TOR_BINARY_PATH=/path/to/tor     # Optional
```

**Fish:**
```fish
set -x EH_USERNAME your_username
set -x EH_PASSWORD your_password
set -x APIKEY_2CAPTCHA your_api_key_here
set -x HBROWSER_LOG_LEVEL INFO          # Optional
set -x TOR_BINARY_PATH /path/to/tor     # Optional
```

**Windows Command Prompt:**
```cmd
set EH_USERNAME=your_username
set EH_PASSWORD=your_password
set APIKEY_2CAPTCHA=your_api_key_here
set HBROWSER_LOG_LEVEL=INFO
set TOR_BINARY_PATH=C:\path\to\tor.exe
```

**Windows PowerShell:**
```powershell
$env:EH_USERNAME="your_username"
$env:EH_PASSWORD="your_password"
$env:APIKEY_2CAPTCHA="your_api_key_here"
$env:HBROWSER_LOG_LEVEL="INFO"
$env:TOR_BINARY_PATH="C:\path\to\tor.exe"
```

HBrowser uses [2Captcha](https://2captcha.com/) service to automatically solve Cloudflare Turnstile and managed challenges that may appear during login. You need to register for a 2Captcha account and obtain an API key.

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

## Usage

Here's a quick example of how to use HBrowser:

```python
from hbrowser import EHDriver


if __name__ == "__main__":
    with EHDriver() as driver:
        driver.punchin()
```

Here's a quick example of how to use HVBrowser:

```python
from hvbrowser import HVDriver


if __name__ == "__main__":
    with HVDriver() as driver:
        driver.monstercheck()
```
