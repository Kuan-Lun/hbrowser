"""代理設定"""

import os
import socket
import tempfile
import zipfile
from typing import Any
from urllib.request import urlopen

from ..utils import setup_logger

logger = setup_logger(__name__)


def _create_proxy_extension(
    proxy_host: str, proxy_port: int, proxy_user: str, proxy_pass: str
) -> str:
    """
    創建一個 Chrome 擴充功能來處理代理認證

    Returns:
        擴充功能 zip 檔案的路徑
    """
    manifest_json = """
{
    "version": "1.0.0",
    "manifest_version": 2,
    "name": "Chrome Proxy",
    "permissions": [
        "proxy",
        "tabs",
        "unlimitedStorage",
        "storage",
        "<all_urls>",
        "webRequest",
        "webRequestBlocking"
    ],
    "background": {
        "scripts": ["background.js"]
    },
    "minimum_chrome_version":"22.0.0"
}
"""

    background_js = f"""
var config = {{
        mode: "fixed_servers",
        rules: {{
          singleProxy: {{
            scheme: "http",
            host: "{proxy_host}",
            port: parseInt({proxy_port})
          }},
          bypassList: ["localhost"]
        }}
      }};

chrome.proxy.settings.set({{value: config, scope: "regular"}}, function() {{}});

function callbackFn(details) {{
    return {{
        authCredentials: {{
            username: "{proxy_user}",
            password: "{proxy_pass}"
        }}
    }};
}}

chrome.webRequest.onAuthRequired.addListener(
            callbackFn,
            {{urls: ["<all_urls>"]}},
            ['blocking']
);
"""

    # 創建臨時目錄
    plugin_dir = tempfile.mkdtemp()

    # 寫入 manifest.json
    with open(os.path.join(plugin_dir, "manifest.json"), "w") as f:
        f.write(manifest_json)

    # 寫入 background.js
    with open(os.path.join(plugin_dir, "background.js"), "w") as f:
        f.write(background_js)

    # 創建 zip 檔案
    plugin_file = os.path.join(tempfile.gettempdir(), "proxy_auth_plugin.zip")
    with zipfile.ZipFile(plugin_file, "w") as zp:
        zp.write(os.path.join(plugin_dir, "manifest.json"), "manifest.json")
        zp.write(os.path.join(plugin_dir, "background.js"), "background.js")

    return plugin_file


def configure_proxy() -> str | None:
    """建立住宅代理擴充功能（如果有設定）。

    Returns:
        proxy extension 路徑，或 None
    """
    rp_username = os.getenv("RP_USERNAME")
    rp_password = os.getenv("RP_PASSWORD")
    rp_dns = os.getenv("RP_DNS")

    if not (rp_username and rp_password and rp_dns):
        return None

    if ":" in rp_dns:
        proxy_host, proxy_port = rp_dns.split(":", 1)
    else:
        proxy_host = rp_dns
        proxy_port = "8080"

    logger.info(f"Using residential proxy: {rp_username}@{proxy_host}:{proxy_port}")

    proxy_extension = _create_proxy_extension(
        proxy_host=proxy_host,
        proxy_port=int(proxy_port),
        proxy_user=rp_username,
        proxy_pass=rp_password,
    )
    logger.debug(f"Proxy extension created at: {proxy_extension}")
    return proxy_extension


def has_residential_proxy() -> bool:
    """檢查是否有設定住宅代理環境變數。"""
    return all(os.getenv(k) for k in ("RP_USERNAME", "RP_PASSWORD", "RP_DNS"))


def find_available_port(start: int = 9150) -> int:
    """找到一個可用的端口"""
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No available port found in range {start}-{start + 99}")


async def verify_proxy_ip(browser: Any, page: Any) -> None:
    """驗證代理連線的 IP 與本機 IP 不同"""
    try:
        with urlopen("https://api.ipify.org", timeout=10) as response:
            local_ip: str = response.read().decode("utf-8").strip()

        await page.get("https://api.ipify.org")
        body = await page.select("body")
        proxy_ip: str = body.text.strip()

        if local_ip == proxy_ip:
            raise RuntimeError(
                f"Proxy IP safety check failed: proxy IP ({proxy_ip}) is the same "
                f"as local IP ({local_ip}). Proxy may not be working properly."
            )

        logger.info(f"Proxy IP verified: {proxy_ip} (local: {local_ip})")
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning(f"Could not verify proxy IP (non-fatal): {e}")
