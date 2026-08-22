"""代理設定"""

import asyncio
import ipaddress
import json
import os
import shutil
import socket
import tempfile
from pathlib import Path
from typing import Any

import httpx

from ..utils import (
    Deadline,
    is_browser_generation_error,
    navigate_and_wait,
    setup_logger,
    wait_for_selector,
)
from .process import ProcessOwnershipError

logger = setup_logger(__name__)
_PROXY_PAGE_DEADLINE_SECONDS = 10.0
_DIRECT_IP_REQUEST_TIMEOUT_SECONDS = 5.0


class ProxyVerificationError(RuntimeError):
    """Proxy availability or isolation could not be proven safely."""


def _require_proxy_deadline(deadline: Deadline, phase: str) -> None:
    if deadline.expired:
        raise ProxyVerificationError(
            f"Proxy verification deadline expired during {phase}"
        )


def _parse_public_ip(value: str) -> str:
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError:
        raise ProxyVerificationError(
            "Proxy verification service returned an invalid public address"
        ) from None


async def _read_direct_public_ip(*, deadline: Deadline) -> str:
    """Read the direct public address with async, cancellable transport ownership."""

    remaining = deadline.remaining()
    if remaining <= 0:
        raise ProxyVerificationError(
            "Proxy verification deadline expired before direct-IP preflight"
        )
    phase_timeout = min(_DIRECT_IP_REQUEST_TIMEOUT_SECONDS, remaining)
    transport_timeout = httpx.Timeout(
        remaining,
        connect=phase_timeout,
        write=phase_timeout,
        pool=phase_timeout,
        read=phase_timeout,
    )

    async def request() -> str:
        async with httpx.AsyncClient(
            timeout=transport_timeout,
            trust_env=False,
        ) as client:
            response = await client.get("https://api.ipify.org")
            _require_proxy_deadline(deadline, "direct-IP response")
            response.raise_for_status()
            _require_proxy_deadline(deadline, "direct-IP status validation")
            public_ip = _parse_public_ip(response.text)
            _require_proxy_deadline(deadline, "direct-IP response parsing")
            return public_ip

    try:
        public_ip = await asyncio.wait_for(request(), timeout=remaining)
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        raise ProxyVerificationError(
            "Direct-IP preflight exceeded the shared proxy deadline"
        ) from None
    except ProxyVerificationError:
        raise
    except httpx.HTTPError:
        raise ProxyVerificationError("Direct-IP preflight transport failed") from None
    _require_proxy_deadline(deadline, "direct-IP preflight completion")
    return public_ip


def _create_proxy_extension(
    proxy_host: str, proxy_port: int, proxy_user: str, proxy_pass: str
) -> str:
    """
    創建一個 Chrome 擴充功能來處理代理認證

    Returns:
        只屬於本次瀏覽器 generation 的擴充功能目錄
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

    proxy_host_literal = json.dumps(proxy_host)
    proxy_user_literal = json.dumps(proxy_user)
    proxy_pass_literal = json.dumps(proxy_pass)
    background_js = f"""
var config = {{
        mode: "fixed_servers",
        rules: {{
          singleProxy: {{
            scheme: "http",
            host: {proxy_host_literal},
            port: {proxy_port}
          }},
          bypassList: ["localhost"]
        }}
      }};

chrome.proxy.settings.set({{value: config, scope: "regular"}}, function() {{}});

function callbackFn(details) {{
    return {{
        authCredentials: {{
            username: {proxy_user_literal},
            password: {proxy_pass_literal}
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
    plugin_dir = Path(tempfile.mkdtemp())

    try:
        manifest_path = plugin_dir / "manifest.json"
        manifest_path.write_text(manifest_json, encoding="utf-8")

        background_path = plugin_dir / "background.js"
        background_path.write_text(background_js, encoding="utf-8")
        return str(plugin_dir)
    except BaseException as creation_error:
        try:
            shutil.rmtree(plugin_dir)
        except BaseException as cleanup_error:
            private_error = ProcessOwnershipError(
                "Proxy extension creation failed and private material "
                "could not be removed"
            )
            private_error.add_note(
                "Creation failure type: " f"{type(creation_error).__name__}"
            )
            raise private_error from cleanup_error
        raise


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

    try:
        parsed_proxy_port = int(proxy_port)
    except ValueError:
        raise ValueError("Residential proxy port is invalid") from None
    if not 1 <= parsed_proxy_port <= 65535:
        raise ValueError("Residential proxy port is invalid")

    logger.debug("Using authenticated residential proxy")

    proxy_extension = _create_proxy_extension(
        proxy_host=proxy_host,
        proxy_port=parsed_proxy_port,
        proxy_user=rp_username,
        proxy_pass=rp_password,
    )
    logger.debug("Proxy extension created for the current browser generation")
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


async def verify_proxy_ip(
    browser: Any,
    page: Any,
    *,
    deadline: Deadline | None = None,
) -> None:
    """驗證代理連線的 IP 與本機 IP 不同"""

    verification_deadline = (
        Deadline.after(_PROXY_PAGE_DEADLINE_SECONDS)
        if deadline is None
        else deadline.bounded(_PROXY_PAGE_DEADLINE_SECONDS)
    )

    local_ip = await _read_direct_public_ip(deadline=verification_deadline)

    await navigate_and_wait(
        page,
        "https://api.ipify.org",
        deadline=verification_deadline,
    )

    try:
        body = await wait_for_selector(
            page,
            "body",
            deadline=verification_deadline,
        )
        _require_proxy_deadline(verification_deadline, "proxy-page DOM receipt")
        proxy_ip = _parse_public_ip(body.text)

        if local_ip == proxy_ip:
            raise ProxyVerificationError(
                "Proxy IP safety check failed: proxy resolved to the local "
                "public address"
            )

        logger.info("Proxy IP verification succeeded")
    except ProxyVerificationError:
        raise
    except Exception as error:
        if is_browser_generation_error(error):
            raise
        raise ProxyVerificationError(
            "Proxy page did not provide a trustworthy public address receipt"
        ) from None
