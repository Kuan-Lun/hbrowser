"""驗證碼相關常量"""

import re

MAX_CAPTCHA_DETECTION_TIMEOUT_SECONDS = 5.0
MAX_MANUAL_CHALLENGE_TIMEOUT_SECONDS = 180.0

TURNSTILE_WIDGET_CSS = (
    "div.cf-turnstile[data-sitekey], "
    "[name='cf-turnstile-response'], "
    "iframe[src*='challenges.cloudflare.com'][src*='turnstile']"
)

SITEKEY_RE = re.compile(r"/(0x[a-zA-Z0-9_-]+)/")
RAY_RE = re.compile(r"Ray ID:\s*<code>\s*([0-9a-f]+)\s*</code>", re.IGNORECASE)
