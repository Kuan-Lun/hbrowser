"""工具函數模組"""

from .log import get_log_dir, setup_logger
from .platform import (
    get_chrome_executable_name,
    get_chromedriver_executable_name,
    get_platform,
)
from .url import matchurl
from .window import find_new_window

__all__ = [
    "get_chrome_executable_name",
    "get_chromedriver_executable_name",
    "get_log_dir",
    "get_platform",
    "setup_logger",
    "matchurl",
    "find_new_window",
]
