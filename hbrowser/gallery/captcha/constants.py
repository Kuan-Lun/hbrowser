"""驗證碼相關常量"""

import re

TURNSTILE_WIDGET_CSS = (
    "div.cf-turnstile[data-sitekey], "
    "[name='cf-turnstile-response'], "
    "iframe[src*='challenges.cloudflare.com'][src*='turnstile']"
)

SITEKEY_RE = re.compile(r"/(0x[a-zA-Z0-9_-]+)/")
RAY_RE = re.compile(r"Ray ID:\s*<code>\s*([0-9a-f]+)\s*</code>", re.IGNORECASE)
