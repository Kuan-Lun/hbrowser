# HBrowser (hbrowser)

## Setup

### Environment Variables

HBrowser requires the following environment variables:

- `APIKEY_2CAPTCHA`: Your 2Captcha API key for solving CAPTCHA challenges
- `HBROWSER_LOG_LEVEL` (optional): Control logging verbosity (DEBUG, INFO, WARNING, ERROR). Default: INFO

Set the environment variables before running the script:

**Bash/Zsh:**
```bash
export APIKEY_2CAPTCHA=your_api_key_here
export HBROWSER_LOG_LEVEL=INFO  # Optional: DEBUG, INFO, WARNING, ERROR
```

**Fish:**
```fish
set -x APIKEY_2CAPTCHA your_api_key_here
set -x HBROWSER_LOG_LEVEL INFO  # Optional
```

**Windows Command Prompt:**
```cmd
set APIKEY_2CAPTCHA=your_api_key_here
set HBROWSER_LOG_LEVEL=INFO
```

**Windows PowerShell:**
```powershell
$env:APIKEY_2CAPTCHA="your_api_key_here"
$env:HBROWSER_LOG_LEVEL="INFO"
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
