"""日誌相關工具函數"""
import os
import sys


def get_log_dir() -> str:
    """
    獲取主腳本所在目錄下的 log 資料夾路徑，如果不存在則建立

    Returns:
        log 資料夾的絕對路徑
    """
    # 獲取主腳本的路徑
    if hasattr(sys, 'argv') and len(sys.argv) > 0:
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
