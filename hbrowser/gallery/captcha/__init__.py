"""
驗證碼處理模組

提供驗證碼檢測功能
"""

from .detector import CaptchaDetector
from .models import ChallengeDetection, Kind

__all__ = [
    "ChallengeDetection",
    "Kind",
    "CaptchaDetector",
]
