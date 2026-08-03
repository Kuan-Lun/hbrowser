"""日誌相關工具函數"""

import logging
import os
import sys
from pathlib import Path

_LOG_DIR_ENVIRONMENT_VARIABLE = "HBROWSER_LOG_DIR"


def setup_logger(name: str) -> logging.Logger:
    """
    創建或獲取 logger

    日誌級別可通過環境變量 HBROWSER_LOG_LEVEL 控制：
    - DEBUG: 詳細的調試信息
    - INFO: 一般信息（默認）
    - WARNING: 警告信息
    - ERROR: 錯誤信息

    Args:
        name: logger 名稱（通常使用 __name__）

    Returns:
        配置好的 Logger 實例

    Example:
        >>> import os
        >>> os.environ["HBROWSER_LOG_LEVEL"] = "DEBUG"
        >>> logger = setup_logger(__name__)
        >>> logger.debug("這是調試信息")
    """
    logger = logging.getLogger(name)

    # 避免重複配置
    if logger.handlers:
        return logger

    # 從環境變量獲取日誌級別
    level_str = os.getenv("HBROWSER_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    # 創建控制台處理器
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # 設置格式化器
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    # 添加處理器到 logger
    logger.addHandler(handler)

    # 防止日誌向上傳播到 root logger（避免重複輸出）
    logger.propagate = False

    return logger


def get_log_dir() -> Path:
    """
    獲取並建立診斷資料夾。

    ``HBROWSER_LOG_DIR`` 可指定整合應用的共同日誌目錄。未指定時，維持
    原有行為，使用主腳本所在目錄下的 ``log`` 資料夾；若沒有可用的主腳本
    路徑則使用目前工作目錄。

    Returns:
        log 資料夾的絕對路徑
    """
    configured_directory = os.getenv(_LOG_DIR_ENVIRONMENT_VARIABLE)
    if configured_directory:
        log_dir = Path(configured_directory).expanduser()
        if not log_dir.is_absolute():
            log_dir = Path.cwd() / log_dir
        log_dir = log_dir.resolve()
    else:
        script_name = sys.argv[0] if sys.argv and sys.argv[0] else ""
        if script_name and not script_name.startswith("-"):
            script_dir = Path(script_name).expanduser().resolve().parent
        else:
            script_dir = Path.cwd().resolve()
        log_dir = script_dir / "log"

    log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    return log_dir
