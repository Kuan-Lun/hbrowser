"""工具函數模組"""

from .browser_generation import is_browser_generation_error
from .deadline import Deadline
from .log import (
    LogLevel,
    LogPersistenceError,
    configure_logging,
    get_log_dir,
    log_context,
    log_to_process_file,
    setup_logger,
)
from .page_state import (
    NavigationReceipt,
    PageStateTimeout,
    mutate_and_wait_for_navigation,
    navigate_and_wait,
    open_tab_and_wait,
    reload_and_wait,
    wait_for_selector,
    wait_for_xpath,
)
from .platform import (
    get_chrome_executable_name,
    get_platform,
)
from .protocol import (
    ZendriverOperationTimeout,
    wait_for_zendriver,
)
from .url import matchurl
from .window import mutate_and_wait_for_new_tab

__all__ = [
    "get_chrome_executable_name",
    "configure_logging",
    "get_log_dir",
    "get_platform",
    "is_browser_generation_error",
    "LogLevel",
    "LogPersistenceError",
    "log_to_process_file",
    "log_context",
    "setup_logger",
    "ZendriverOperationTimeout",
    "Deadline",
    "NavigationReceipt",
    "PageStateTimeout",
    "wait_for_zendriver",
    "mutate_and_wait_for_navigation",
    "mutate_and_wait_for_new_tab",
    "navigate_and_wait",
    "open_tab_and_wait",
    "reload_and_wait",
    "wait_for_selector",
    "wait_for_xpath",
    "matchurl",
]
