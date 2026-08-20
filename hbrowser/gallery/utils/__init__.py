"""工具函數模組"""

from .browser_generation import is_browser_generation_error
from .diagnostic import write_page_diagnostic
from .log import get_log_dir, log_context, setup_logger
from .platform import (
    get_chrome_executable_name,
    get_platform,
)
from .protocol import (
    ZendriverOperationTimeout,
    wait_for_zendriver,
)
from .url import matchurl
from .window import wait_for_new_tab

__all__ = [
    "get_chrome_executable_name",
    "get_log_dir",
    "get_platform",
    "is_browser_generation_error",
    "log_context",
    "setup_logger",
    "ZendriverOperationTimeout",
    "wait_for_zendriver",
    "write_page_diagnostic",
    "matchurl",
    "wait_for_new_tab",
]
