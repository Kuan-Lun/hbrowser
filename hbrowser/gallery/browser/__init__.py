"""瀏覽器相關模組"""

from .ban_handler import handle_ban_decorator, parse_ban_time
from .factory import create_browser, stop_browser
from .flaresolverr import (
    FlareSolverrClient,
    FlareSolverrConfigurationError,
    FlareSolverrError,
    FlareSolverrProtocolError,
    FlareSolverrRequestError,
    FlareSolverrResult,
    FlareSolverrSession,
    FlareSolverrSessionScope,
    FlareSolverrSessionUnavailable,
    FlareSolverrSolveReceipt,
    get_flaresolverr_url,
    should_use_flaresolverr,
)

__all__ = [
    "FlareSolverrClient",
    "FlareSolverrConfigurationError",
    "FlareSolverrError",
    "FlareSolverrProtocolError",
    "FlareSolverrRequestError",
    "FlareSolverrResult",
    "FlareSolverrSession",
    "FlareSolverrSessionScope",
    "FlareSolverrSessionUnavailable",
    "FlareSolverrSolveReceipt",
    "create_browser",
    "stop_browser",
    "get_flaresolverr_url",
    "handle_ban_decorator",
    "parse_ban_time",
    "should_use_flaresolverr",
]
