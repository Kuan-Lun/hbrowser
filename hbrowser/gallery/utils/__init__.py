"""工具函數模組"""

from .connection import is_connection_error
from .log import get_log_dir, setup_logger
from .platform import (
    get_chrome_executable_name,
    get_platform,
)
from .url import matchurl
from .window import wait_for_new_tab

__all__ = [
    "get_chrome_executable_name",
    "get_log_dir",
    "get_platform",
    "is_connection_error",
    "setup_logger",
    "matchurl",
    "wait_for_new_tab",
]
