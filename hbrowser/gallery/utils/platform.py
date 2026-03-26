"""平台偵測工具函數"""

import platform


def get_platform() -> str:
    """
    根據作業系統和架構取得對應的平台代碼

    Returns:
        平台代碼 (mac-arm64, mac-x64, linux64, win64, win32)
    """
    system = platform.system()
    machine = platform.machine().lower()

    match system:
        case "Darwin":
            # macOS
            if machine == "arm64":
                return "mac-arm64"
            else:
                return "mac-x64"
        case "Linux":
            return "linux64"
        case "Windows":
            # 檢查是否為 64 位元
            if platform.machine().endswith("64"):
                return "win64"
            else:
                return "win32"
        case _:
            raise RuntimeError(f"Unsupported platform: {system} {machine}")


def get_chrome_executable_name(plat: str) -> str:
    """取得 Chrome 執行檔名稱"""
    if plat.startswith("win"):
        return "chrome.exe"
    elif plat.startswith("mac"):
        return "Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
    else:
        return "chrome"


def get_chromedriver_executable_name(plat: str) -> str:
    """取得 ChromeDriver 執行檔名稱"""
    if plat.startswith("win"):
        return "chromedriver.exe"
    else:
        return "chromedriver"
