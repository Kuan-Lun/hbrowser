"""驗證碼相關數據模型"""

from dataclasses import dataclass
from typing import Literal

Kind = Literal["none", "turnstile_widget", "cf_managed_challenge", "recaptcha_v2"]


@dataclass(frozen=True)
class ChallengeDetection:
    """驗證碼檢測結果"""

    url: str
    kind: Kind
    sitekey: str | None = None
    iframe_src: str | None = None
    ray_id: str | None = None


@dataclass
class SolveResult:
    """驗證碼解決結果"""

    success: bool
    token: str | None = None
    error_message: str | None = None
    solver_name: str = "unknown"
