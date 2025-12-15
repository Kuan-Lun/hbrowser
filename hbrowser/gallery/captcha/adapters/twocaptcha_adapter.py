"""
TwoCaptcha 適配器
將 TwoCaptcha 第三方服務適配到 CaptchaSolver 接口

第三方依賴：此文件依賴 2captcha-python 套件
移除 TwoCaptcha 依賴時，刪除此文件即可
"""

import os
import re
import time
from typing import Any

from ..solver_interface import CaptchaSolver
from ..models import ChallengeDetection, SolveResult
from ....exceptions import CaptchaAPIKeyNotSetException, CaptchaSolveException
from ...utils.log import get_log_dir
from ..detector import CaptchaDetector

from twocaptcha import TwoCaptcha  # type: ignore


class TwoCaptchaAdapter(CaptchaSolver):
    """適配 TwoCaptcha 服務到統一接口"""

    def __init__(self, api_key: str | None = None, max_wait: int = 120) -> None:
        """
        初始化 TwoCaptcha 適配器

        Args:
            api_key: 2Captcha API key（如果未提供，從環境變數 APIKEY_2CAPTCHA 讀取）
            max_wait: 驗證碼解決的最大等待時間（秒），默認 120 秒
        """
        self.api_key = api_key or os.getenv("APIKEY_2CAPTCHA")
        if not self.api_key:
            raise CaptchaAPIKeyNotSetException(
                "APIKEY_2CAPTCHA environment variable is not set. "
                "Please set it using: export APIKEY_2CAPTCHA=your_api_key"
            )

        self.max_wait = max_wait
        # 被適配的對象
        self._twocaptcha = TwoCaptcha(self.api_key)

    def solve(self, challenge: ChallengeDetection, driver: Any) -> SolveResult:
        """
        使用 TwoCaptcha 服務解決驗證碼

        Args:
            challenge: 檢測到的驗證信息
            driver: Selenium WebDriver 實例

        Returns:
            SolveResult: 解決結果
        """
        print(f"Detected {challenge.kind} challenge, attempting to solve...")

        try:
            match challenge.kind:
                case "cf_managed_challenge":
                    solveresult = self._solve_managed_challenge(
                        challenge, driver, max_wait=self.max_wait
                    )
                case "turnstile_widget":
                    solveresult = self._solve_turnstile_widget(challenge, driver)
                case "recaptcha_v2":
                    solveresult = self._solve_recaptcha_v2(challenge, driver)
                case _:
                    solveresult = SolveResult(
                        success=False,
                        error_message=f"Unsupported challenge type: {challenge.kind}",
                        solver_name="TwoCaptcha",
                    )
            return solveresult

        except (CaptchaAPIKeyNotSetException, CaptchaSolveException):
            raise
        except Exception as e:
            # 保存錯誤時的頁面狀態
            with open(
                os.path.join(get_log_dir(), "challenge_error.html"),
                "w",
                errors="ignore",
            ) as f:
                f.write(driver.page_source)

            raise CaptchaSolveException(
                f"Failed to solve Cloudflare challenge: {str(e)}"
            ) from e

    def _solve_managed_challenge(
        self, challenge: ChallengeDetection, driver: Any, max_wait: int
    ) -> SolveResult:
        """解決 Cloudflare managed challenge"""
        # 保存當前頁面以供調試
        with open(
            os.path.join(get_log_dir(), "challenge_page.html"),
            "w",
            errors="ignore",
        ) as f:
            f.write(driver.page_source)

        print(f"Cloudflare managed challenge detected (Ray ID: {challenge.ray_id})")

        # 嘗試提取 sitekey
        html = driver.page_source
        sitekey_match = re.search(r'sitekey["\s:]+([0-9a-zA-Z_-]+)', html)

        if sitekey_match:
            sitekey = sitekey_match.group(1)
            print(f"Found sitekey in managed challenge: {sitekey}")

            # 嘗試使用 2Captcha 解決 Turnstile
            try:
                print("Attempting to solve with 2Captcha Turnstile API...")
                result = self._twocaptcha.turnstile(
                    sitekey=sitekey,
                    url=challenge.url,
                )

                token = result.get("code")
                if token:
                    print(f"Got token from 2Captcha: {token[:50]}...")

                    # 嘗試注入 token
                    driver.execute_script(
                        """
                        // 方法1: 尋找並設置 turnstile response input
                        var inputs = document.querySelectorAll('input[name*="turnstile"], input[name*="cf-turnstile"]');
                        for (var i = 0; i < inputs.length; i++) {
                            inputs[i].value = arguments[0];
                        }

                        // 方法2: 如果有 callback
                        if (window.turnstile && typeof window.turnstile.reset === 'function') {
                            try {
                                // 嘗試觸發驗證完成
                                if (window.cfCallback) window.cfCallback(arguments[0]);
                                if (window.tsCallback) window.tsCallback(arguments[0]);
                            } catch(e) {
                                console.log('Callback error:', e);
                            }
                        }

                        // 方法3: 提交表單（如果存在）
                        var form = document.querySelector('form');
                        if (form) {
                            try {
                                form.submit();
                            } catch(e) {
                                console.log('Form submit error:', e);
                            }
                        }
                        """,
                        token,
                    )

                    print("Token injected, waiting for page to respond...")
                    time.sleep(5)
            except Exception as e:
                print(f"2Captcha solve attempt failed: {str(e)}")
                print("Falling back to passive waiting...")

        # 輪詢檢查頁面是否已經通過驗證
        print("Monitoring page for challenge resolution...")
        if not sitekey_match:
            print(
                "NOTE: No sitekey found. If running in non-headless mode, please solve the captcha manually."
            )
            print(
                "      The script will automatically continue once the challenge is resolved."
            )

        detector = CaptchaDetector()

        start_time = time.time()
        check_interval = 5
        last_url = driver.current_url
        last_title = driver.title

        while time.time() - start_time < max_wait:
            time.sleep(check_interval)

            current_url = driver.current_url
            current_title = driver.title

            # 檢查 URL 是否變化
            if current_url != last_url:
                print(f"URL changed from {last_url} to {current_url}")
                last_url = current_url

            # 檢查標題是否變化（可能表示頁面狀態變更）
            if current_title != last_title:
                print(f"Page title changed: {last_title} -> {current_title}")
                last_title = current_title

            # 重新檢測是否還有驗證
            current_det = detector.detect(driver, timeout=1.0)
            if current_det.kind == "none":
                print("Challenge resolved successfully!")
                return SolveResult(success=True, solver_name="TwoCaptcha")

            elapsed = int(time.time() - start_time)
            print(
                f"Still waiting... ({elapsed}s/{max_wait}s) - Current URL: {current_url[:50]}..."
            )

        # 超時仍未解決
        raise CaptchaSolveException(
            f"Cloudflare managed challenge not resolved after {max_wait} seconds. "
            f"Ray ID: {challenge.ray_id}. "
            f"\n\nPossible solutions:"
            f"\n1. Disable headless mode by setting headless=False"
            f"\n2. Try running the script with a real browser window"
            f"\n3. Use a different IP address or wait before retrying"
            f"\n4. Cloudflare may be blocking automated access to this site"
        )

    def _solve_turnstile_widget(
        self, challenge: ChallengeDetection, driver: Any
    ) -> SolveResult:
        """解決 Turnstile widget"""
        if not challenge.sitekey:
            raise CaptchaSolveException(
                "Turnstile widget detected but sitekey not found"
            )

        print(f"Solving Turnstile with sitekey: {challenge.sitekey}")

        # 提交驗證任務到 2Captcha
        result = self._twocaptcha.turnstile(
            sitekey=challenge.sitekey,
            url=challenge.url,
        )

        # 獲取解決的 token
        token = result.get("code")
        if not token:
            raise CaptchaSolveException("Failed to get token from 2Captcha response")

        print(f"Got token from 2Captcha: {token[:50]}...")

        # 將 token 注入到頁面
        success = driver.execute_script(
            """
            if (window.turnstile && window.turnstile.reset) {
                // 如果有 callback，直接調用
                if (window.cfCallback || window.tsCallback) {
                    const callback = window.cfCallback || window.tsCallback;
                    callback(arguments[0]);
                    return true;
                }
            }

            // 方法2: 設置到隱藏的表單欄位
            const input = document.querySelector('input[name="cf-turnstile-response"]');
            if (input) {
                input.value = arguments[0];
                return true;
            }

            return false;
            """,
            token,
        )

        if success:
            print("Token injected successfully, waiting for page to respond...")
            time.sleep(3)

            # 檢查是否通過驗證

            detector = CaptchaDetector()
            current_det = detector.detect(driver, timeout=1.0)
            if current_det.kind == "none":
                print("Turnstile challenge resolved successfully!")
                return SolveResult(success=True, solver_name="TwoCaptcha")
        else:
            print("Warning: Could not inject token using standard methods")

        return SolveResult(
            success=False,
            error_message="Token injection failed",
            solver_name="TwoCaptcha",
        )

    def _solve_recaptcha_v2(self, challenge: ChallengeDetection, driver: Any) -> SolveResult:
        """解決 reCAPTCHA v2"""
        if not challenge.sitekey:
            raise CaptchaSolveException("reCAPTCHA v2 detected but sitekey not found")

        print(f"Solving reCAPTCHA v2 with sitekey: {challenge.sitekey}")

        # 提交驗證任務到 2Captcha
        result = self._twocaptcha.recaptcha(
            sitekey=challenge.sitekey,
            url=challenge.url,
        )

        # 獲取解決的 token
        token = result.get("code")
        if not token:
            raise CaptchaSolveException("Failed to get token from 2Captcha response")

        print(f"Got token from 2Captcha: {token[:50]}...")

        # 將 token 注入到頁面
        success = driver.execute_script(
            """
            // 嘗試設置 g-recaptcha-response 欄位
            var recaptchaResponse = document.getElementById('g-recaptcha-response');
            if (recaptchaResponse) {
                recaptchaResponse.style.display = '';
                recaptchaResponse.value = arguments[0];
                return true;
            }
            return false;
            """,
            token,
        )
        if success:
            print("Token injected successfully, waiting for page to respond...")
            time.sleep(3)

            # 檢查是否通過驗證
            detector = CaptchaDetector()
            current_det = detector.detect(driver, timeout=1.0)
            if current_det.kind == "none":
                print("reCAPTCHA v2 challenge resolved successfully!")
                return SolveResult(success=True, solver_name="TwoCaptcha")
        else:
            print("Warning: Could not inject token using standard methods")

        return SolveResult(
            success=False,
            error_message="Token injection failed",
            solver_name="TwoCaptcha",
        )
