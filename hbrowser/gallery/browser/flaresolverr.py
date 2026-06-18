"""FlareSolverr 整合 - 自動解決 Cloudflare managed challenge"""

import os
from dataclasses import dataclass
from typing import Any

import httpx
from zendriver import cdp

from ..utils import setup_logger
from .tor import should_use_tor

logger = setup_logger(__name__)

_SAME_SITE_MAP = {
    "Strict": cdp.network.CookieSameSite.STRICT,
    "Lax": cdp.network.CookieSameSite.LAX,
    "None": cdp.network.CookieSameSite.NONE,
}


@dataclass
class FlareSolverrResult:
    cookies: list[dict[str, Any]]
    user_agent: str

    def to_cdp_cookie_params(self) -> list[cdp.network.CookieParam]:
        """將 FlareSolverr 回傳的 cookie 轉換成可直接送進 CDP 的格式。"""
        return [self._cookie_to_cdp_param(c) for c in self.cookies]

    @staticmethod
    def _cookie_to_cdp_param(cookie: dict[str, Any]) -> cdp.network.CookieParam:
        expiry = cookie.get("expiry")
        return cdp.network.CookieParam(
            name=cookie["name"],
            value=cookie["value"],
            domain=cookie.get("domain"),
            path=cookie.get("path"),
            secure=cookie.get("secure"),
            http_only=cookie.get("httpOnly"),
            same_site=_SAME_SITE_MAP.get(cookie.get("sameSite", "")),
            expires=(
                cdp.network.TimeSinceEpoch(expiry) if expiry and expiry > 0 else None
            ),
        )


def get_flaresolverr_url() -> str | None:
    """讀取 FLARESOLVERR_URL 環境變數（例如 http://127.0.0.1:8191/v1）"""
    url = os.getenv("FLARESOLVERR_URL")
    return url.rstrip("/") if url else None


def should_use_flaresolverr() -> bool:
    """判斷是否啟用 FlareSolverr。

    FlareSolverr 解出的 cf_clearance cookie 與求解時的來源 IP 綁定，
    若同時使用 Tor，求解瀏覽器與正式瀏覽器走的 IP 不一致，cookie 也無法套用，
    因此 Tor 啟用時直接忽略 FlareSolverr。
    """
    url = get_flaresolverr_url()
    if not url:
        return False
    if should_use_tor():
        logger.warning(
            "FLARESOLVERR_URL is set but Tor is enabled; ignoring FlareSolverr "
            "because the clearance cookie would not match the Tor exit IP."
        )
        return False
    return True


async def solve_with_flaresolverr(
    url: str, flaresolverr_url: str, timeout_ms: int = 60000
) -> FlareSolverrResult:
    """呼叫 FlareSolverr API 解決 Cloudflare 挑戰，回傳 cookies 與 user agent。

    Raises:
        httpx.HTTPError: 連線失敗
        RuntimeError: FlareSolverr 回報求解失敗
    """
    async with httpx.AsyncClient(timeout=timeout_ms / 1000 + 10) as client:
        response = await client.post(
            flaresolverr_url,
            json={
                "cmd": "request.get",
                "url": url,
                "maxTimeout": timeout_ms,
            },
        )
        response.raise_for_status()
        data = response.json()

    if data.get("status") != "ok":
        raise RuntimeError(f"FlareSolverr failed to solve challenge: {data}")

    solution = data["solution"]
    return FlareSolverrResult(
        cookies=solution.get("cookies", []),
        user_agent=solution.get("userAgent", ""),
    )
