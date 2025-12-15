"""驗證碼相關數據模型"""

from dataclasses import dataclass
from typing import Optional, Literal


Kind = Literal["none", "turnstile_widget", "cf_managed_challenge", "recaptcha_v2"]


@dataclass(frozen=True)
class ChallengeDetection:
    """驗證碼檢測結果"""

    url: str
    kind: Kind
    sitekey: Optional[str] = None
    iframe_src: Optional[str] = None
    ray_id: Optional[str] = None


@dataclass
class SolveResult:
    """驗證碼解決結果"""

    success: bool
    token: Optional[str] = None
    error_message: Optional[str] = None
    solver_name: str = "unknown"
