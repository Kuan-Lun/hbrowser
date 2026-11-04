"""代理設定"""

import asyncio
import ipaddress
import json
import logging
import os
import secrets
import socket
import tempfile
from pathlib import Path
from typing import Any

import httpx
from zendriver import cdp

from ..utils import (
    Deadline,
    is_browser_generation_error,
    navigate_and_wait,
    wait_for_selector,
    wait_for_zendriver,
)
from .process import ProcessOwnershipError, _PrivateDirectory

logger = logging.getLogger(__name__)
_PROXY_PAGE_DEADLINE_SECONDS = 10.0
_DIRECT_IP_REQUEST_TIMEOUT_SECONDS = 5.0
_PROXY_EXTENSION_CLEANUP_TIMEOUT_SECONDS = 5.0


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
    proxy_host: str,
    proxy_port: int,
    proxy_user: str,
    proxy_pass: str,
    *,
    cleanup_deadline: Deadline | None = None,
) -> str:
    """
    創建一個 Chrome 擴充功能來處理代理認證

    Returns:
        只屬於本次瀏覽器 generation 的擴充功能目錄
    """
    manifest_json = json.dumps(
        {
            "version": "1.0.0",
            "manifest_version": 3,
            "name": "HBrowser Proxy",
            "permissions": ["proxy", "webRequest", "webRequestAuthProvider"],
            "host_permissions": ["<all_urls>"],
            "background": {"service_worker": "background.js"},
        }
    )
    verification_token = secrets.token_hex(32)

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

function callbackFn(details, callback) {{
    const challenger = details.challenger;
    if (details.isProxy !== true || !challenger ||
        typeof challenger.host !== "string" ||
        challenger.host.toLowerCase() !== {proxy_host_literal}.toLowerCase() ||
        challenger.port !== {proxy_port}) {{
        callback({{}});
        return;
    }}
    callback({{
        authCredentials: {{
            username: {proxy_user_literal},
            password: {proxy_pass_literal}
        }}
    }});
}}

chrome.webRequest.onAuthRequired.addListener(
            callbackFn,
            {{urls: ["<all_urls>"]}},
            ['asyncBlocking']
);
const configurationApplied = new Promise(resolve => {{
    chrome.proxy.settings.set({{value: config, scope: "regular"}}, () => {{
        resolve(!chrome.runtime.lastError);
    }});
}});
// A private generation nonce prevents accepting another extension's receipt.
// Re-read Chrome's effective setting; successful set() alone is insufficient.
globalThis.hbrowserProxyReady = async token => {{
    if (token !== {json.dumps(verification_token)} ||
        !(await configurationApplied) ||
        !chrome.webRequest.onAuthRequired.hasListener(callbackFn)) {{
        return false;
    }}
    return new Promise(resolve => chrome.proxy.settings.get(
        {{incognito: false}}, details => {{
            const value = details && details.value;
            const rules = value && value.rules;
            const server = rules && rules.singleProxy;
            resolve(!chrome.runtime.lastError &&
                details.levelOfControl === "controlled_by_this_extension" &&
                value.mode === "fixed_servers" && !!server &&
                server.scheme === "http" &&
                server.host.toLowerCase() === {proxy_host_literal}.toLowerCase() &&
                server.port === {proxy_port} &&
                JSON.stringify(rules.bypassList) === JSON.stringify(["localhost"]));
        }}
    ));
}};
"""

    # 創建臨時目錄
    plugin_dir = Path(tempfile.mkdtemp())
    private_directory = _PrivateDirectory.capture(plugin_dir)

    try:
        manifest_path = plugin_dir / "manifest.json"
        manifest_path.write_text(manifest_json, encoding="utf-8")

        background_path = plugin_dir / "background.js"
        background_path.write_text(background_js, encoding="utf-8")
        (plugin_dir / "verification-token").write_text(
            verification_token, encoding="ascii"
        )
        return str(plugin_dir)
    except BaseException as creation_error:
        try:
            deadline = (
                Deadline.after(_PROXY_EXTENSION_CLEANUP_TIMEOUT_SECONDS)
                if cleanup_deadline is None
                else cleanup_deadline.bounded(_PROXY_EXTENSION_CLEANUP_TIMEOUT_SECONDS)
            )
            private_directory.remove(deadline=deadline.expires_at)
        except BaseException as cleanup_error:
            private_error = ProcessOwnershipError(
                "Proxy extension creation failed and private material "
                "could not be removed"
            )
            private_error.add_note(
                f"Creation failure type: {type(creation_error).__name__}"
            )
            raise private_error from cleanup_error
        raise


def configure_proxy(*, cleanup_deadline: Deadline | None = None) -> str | None:
    """建立住宅代理擴充功能（如果有設定）。

    Returns:
        proxy extension 路徑，或 None
    """
    credentials = _residential_proxy_credentials()
    if credentials is None:
        return None
    rp_username, rp_password, rp_dns = credentials

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
        cleanup_deadline=cleanup_deadline,
    )
    logger.debug("Proxy extension created for the current browser generation")
    return proxy_extension


def _residential_proxy_credentials() -> tuple[str, str, str] | None:
    """Treat an entirely empty configuration as disabled; reject partial input."""

    username = os.getenv("RP_USERNAME")
    password = os.getenv("RP_PASSWORD")
    endpoint = os.getenv("RP_DNS")
    if not any((username, password, endpoint)):
        return None
    if not username or not password or not endpoint:
        raise ValueError(
            "RP_USERNAME, RP_PASSWORD and RP_DNS must be provided together"
        )
    return username, password, endpoint


def has_residential_proxy() -> bool:
    """Return whether complete credentials are configured, rejecting partial input."""
    return _residential_proxy_credentials() is not None


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


async def verify_residential_proxy(
    browser: Any,
    extension_directory: str,
    *,
    deadline: Deadline,
) -> None:
    """Verify the private extension and Chrome's effective proxy without navigation."""

    token = (Path(extension_directory) / "verification-token").read_text(
        encoding="ascii"
    )
    if len(token) != 64 or any(
        character not in "0123456789abcdef" for character in token
    ):
        raise ProxyVerificationError(
            "Residential proxy verification identity is invalid"
        )
    verification_deadline = deadline.bounded(_PROXY_PAGE_DEADLINE_SECONDS)
    expression = (
        "typeof globalThis.hbrowserProxyReady === 'function' && "
        f"globalThis.hbrowserProxyReady({json.dumps(token)})"
    )
    while not verification_deadline.expired:
        for target in tuple(browser.targets):
            if target.type_ != "service_worker" or not target.url.startswith(
                "chrome-extension://"
            ):
                continue
            _require_proxy_deadline(verification_deadline, "extension verification")
            result, error = await wait_for_zendriver(
                target.send(
                    cdp.runtime.evaluate(
                        expression,
                        await_promise=True,
                        return_by_value=True,
                    )
                ),
                timeout=min(5.0, verification_deadline.remaining()),
                owner=target,
            )
            _require_proxy_deadline(verification_deadline, "extension receipt")
            if error is not None:
                raise ProxyVerificationError(
                    "Residential proxy extension failed verification"
                )
            if result.value is True:
                return
        await asyncio.sleep(min(0.1, verification_deadline.remaining()))
    raise ProxyVerificationError(
        "Residential proxy extension did not establish its configuration"
    )


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
