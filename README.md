# HBrowser (hbrowser)

## Setup

### Environment Variables

HBrowser requires the following environment variable to handle Cloudflare verification challenges:

- `APIKEY_2CAPTCHA`: Your 2Captcha API key for solving CAPTCHA challenges

Set the environment variable before running the script:

**Bash/Zsh:**
```bash
export APIKEY_2CAPTCHA=your_api_key_here
```

**Fish:**
```fish
set -x APIKEY_2CAPTCHA your_api_key_here
```

**Windows Command Prompt:**
```cmd
set APIKEY_2CAPTCHA=your_api_key_here
```

**Windows PowerShell:**
```powershell
$env:APIKEY_2CAPTCHA="your_api_key_here"
```

HBrowser uses [2Captcha](https://2captcha.com/) service to automatically solve Cloudflare Turnstile and managed challenges that may appear during login. You need to register for a 2Captcha account and obtain an API key.

## Usage

Here's a quick example of how to use HBrowser:

```python
from hbrowser import EHDriver


if __name__ == "__main__":
    with EHDriver(**driverpass.getdict()) as driver:
        driver.punchin()
```

Here's a quick example of how to use HVBrowser:

```python
from hvbrowser import HVDriver


if __name__ == "__main__":
    with HVDriver() as driver:
        driver.monstercheck()
```
