"""日誌相關工具函數"""

import logging
import os
import sys


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


def get_log_dir() -> str:
    """
    獲取主腳本所在目錄下的 log 資料夾路徑，如果不存在則建立

    Returns:
        log 資料夾的絕對路徑
    """
    # 獲取主腳本的路徑
    if hasattr(sys, "argv") and len(sys.argv) > 0:
        main_script = sys.argv[0]
        if main_script:
            # 獲取主腳本所在的目錄
            script_dir = os.path.dirname(os.path.abspath(main_script))
        else:
            # 如果無法獲取主腳本路徑，使用當前工作目錄
            script_dir = os.getcwd()
    else:
        script_dir = os.getcwd()

    # 建立 log 資料夾路徑
    log_dir = os.path.join(script_dir, "log")

    # 如果 log 資料夾不存在，則建立
    os.makedirs(log_dir, exist_ok=True)

    return log_dir
