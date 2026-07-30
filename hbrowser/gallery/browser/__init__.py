"""瀏覽器相關模組"""

from .ban_handler import handle_ban_decorator, parse_ban_time
from .factory import create_browser, stop_browser
from .flaresolverr import (
    FlareSolverrClient,
    FlareSolverrResult,
    FlareSolverrSession,
    get_flaresolverr_url,
    should_use_flaresolverr,
)
from .proxy_rotator import DriverRestartRotator, ProxyRotator

__all__ = [
    "DriverRestartRotator",
    "FlareSolverrClient",
    "FlareSolverrResult",
    "FlareSolverrSession",
    "ProxyRotator",
    "create_browser",
    "stop_browser",
    "get_flaresolverr_url",
    "handle_ban_decorator",
    "parse_ban_time",
    "should_use_flaresolverr",
]
