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

from twocaptcha import TwoCaptcha  # type: ignore

from ....exceptions import CaptchaAPIKeyNotSetException, CaptchaSolveException
from ...utils.log import get_log_dir, setup_logger
from ..detector import CaptchaDetector
from ..models import ChallengeDetection, SolveResult
from ..solver_interface import CaptchaSolver

logger = setup_logger(__name__)


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
        logger.info(f"Detected {challenge.kind} challenge, attempting to solve...")

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
            error_file = os.path.join(get_log_dir(), "challenge_error.html")
            with open(error_file, "w", errors="ignore") as f:
                f.write(driver.page_source)
            logger.error(f"Failed to solve challenge, page saved to {error_file}")

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

        logger.info(
            f"Cloudflare managed challenge detected (Ray ID: {challenge.ray_id})"
        )

        # 等待 Turnstile iframe 載入（最多等待 10 秒）
        logger.debug("Waiting for Cloudflare Turnstile to load...")
        iframe_wait_start = time.time()
        iframe_loaded = False
        while time.time() - iframe_wait_start < 10:
            try:
                iframes = driver.find_elements("tag name", "iframe")
                if iframes:
                    wait_time = int(time.time() - iframe_wait_start)
                    logger.debug(f"Found {len(iframes)} iframe(s) after {wait_time}s")
                    iframe_loaded = True
                    break
            except Exception:
                pass
            time.sleep(0.5)

        if not iframe_loaded:
            logger.warning("No iframes loaded after 10 seconds")

        # 嘗試多種方式提取 sitekey
        html = driver.page_source
        sitekey = None

        # 方法 1: 從 sitekey 屬性提取
        patterns = [
            r'sitekey["\s:]+([0-9a-zA-Z_-]+)',
            r'data-sitekey["\s]*=["\s]*([0-9a-zA-Z_-]+)',
            r'"siteKey"["\s]*:["\s]*"([0-9a-zA-Z_-]+)"',
            r'turnstile\.render\([^,]+,[^{]*\{[^}]*sitekey["\s]*:["\s]*["\']([0-9a-zA-Z_-]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                sitekey = match.group(1)
                logger.debug(f"Found sitekey using pattern '{pattern}': {sitekey}")
                break

        # 方法 2: 從 JavaScript 變數提取
        if not sitekey:
            try:
                sitekey = driver.execute_script(
                    """
                    // 嘗試從各種可能的全域變數獲取 sitekey
                    if (window.turnstile && window.turnstile.sitekey) {
                        return window.turnstile.sitekey;
                    }
                    if (window._cf_chl_opt && window._cf_chl_opt.cKey) {
                        return window._cf_chl_opt.cKey;
                    }
                    // 搜尋所有 iframe 的 src
                    const iframes = document.querySelectorAll('iframe');
                    for (let iframe of iframes) {
                        const src = iframe.src || '';
                        const match = src.match(/sitekey=([0-9a-zA-Z_-]+)/);
                        if (match) return match[1];
                    }
                    return null;
                """
                )
                if sitekey:
                    logger.debug(f"Found sitekey from JavaScript: {sitekey}")
            except Exception as e:
                logger.warning(f"Failed to extract sitekey from JavaScript: {e}")

        # 方法 3: 檢查是否有 Turnstile iframe
        if not sitekey:
            try:
                iframes = driver.find_elements(
                    "css selector", "iframe[src*='challenges.cloudflare.com']"
                )
                if iframes:
                    iframe_src = iframes[0].get_attribute("src")
                    logger.debug(f"Found Cloudflare challenge iframe: {iframe_src}")
                    match = re.search(r"sitekey=([0-9a-zA-Z_-]+)", iframe_src)
                    if match:
                        sitekey = match.group(1)
                        logger.debug(f"Extracted sitekey from iframe src: {sitekey}")
            except Exception as e:
                logger.warning(f"Failed to check iframes: {e}")

        if sitekey:
            logger.info(f"Final sitekey to use: {sitekey}")

            # 嘗試使用 2Captcha 解決 Turnstile
            try:
                logger.info("Attempting to solve with 2Captcha Turnstile API...")
                result = self._twocaptcha.turnstile(
                    sitekey=sitekey,
                    url=challenge.url,
                )

                token = result.get("code")
                if token:
                    logger.info(f"Got token from 2Captcha: {token[:50]}...")

                    # 嘗試注入 token
                    driver.execute_script(
                        """
                        // 方法1: 尋找並設置 turnstile response input
                        var inputs = document.querySelectorAll(
                            'input[name*="turnstile"], input[name*="cf-turnstile"]'
                        );
                        for (var i = 0; i < inputs.length; i++) {
                            inputs[i].value = arguments[0];
                        }

                        // 方法2: 如果有 callback
                        if (
                            window.turnstile &&
                            typeof window.turnstile.reset === 'function'
                        ) {
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

                    logger.info("Token injected, waiting for page to respond...")
                    time.sleep(5)
            except Exception as e:
                logger.warning(f"2Captcha solve attempt failed: {str(e)}")
                logger.info("Falling back to passive waiting...")

        # 輪詢檢查頁面是否已經通過驗證
        logger.info("Monitoring page for challenge resolution...")
        if not sitekey:
            logger.warning(
                "No sitekey found. If running in non-headless mode, "
                "please solve the captcha manually"
            )
            logger.info(
                "The script will automatically continue once the challenge is resolved"
            )
            logger.debug("Saving additional debug information...")

            # 保存頁面截圖（如果可能）
            try:
                screenshot_path = os.path.join(
                    get_log_dir(), "challenge_screenshot.png"
                )
                driver.save_screenshot(screenshot_path)
                logger.debug(f"Screenshot saved to {screenshot_path}")
            except Exception as e:
                logger.debug(f"Failed to save screenshot: {e}")

            # 打印當前 URL 和標題
            logger.debug(f"Current URL: {driver.current_url}")
            logger.debug(f"Page title: {driver.title}")

            # 檢查頁面中的關鍵元素
            try:
                cf_elements = driver.find_elements(
                    "css selector", "[class*='cf'], [id*='cf']"
                )
                logger.debug(f"Found {len(cf_elements)} Cloudflare-related elements")

                challenge_forms = driver.find_elements("css selector", "form")
                logger.debug(f"Found {len(challenge_forms)} forms on page")

                all_iframes = driver.find_elements("tag name", "iframe")
                logger.debug(f"Found {len(all_iframes)} iframes on page")
                for idx, iframe in enumerate(all_iframes):
                    src = iframe.get_attribute("src") or "(no src)"
                    logger.debug(f"  iframe {idx}: {src[:100]}")
            except Exception as e:
                logger.debug(f"Failed to inspect page elements: {e}")

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
                logger.info(f"URL changed from {last_url} to {current_url}")
                last_url = current_url

            # 檢查標題是否變化（可能表示頁面狀態變更）
            if current_title != last_title:
                logger.info(f"Page title changed: {last_title} -> {current_title}")
                last_title = current_title

            # 重新檢測是否還有驗證
            current_det = detector.detect(driver, timeout=1.0)
            if current_det.kind == "none":
                logger.info("Challenge resolved successfully!")
                return SolveResult(success=True, solver_name="TwoCaptcha")

            elapsed = int(time.time() - start_time)
            url_preview = current_url[:50]
            logger.debug(
                f"Still waiting... ({elapsed}s/{max_wait}s) - "
                f"Current URL: {url_preview}..."
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

        logger.info(f"Solving Turnstile with sitekey: {challenge.sitekey}")

        # 提交驗證任務到 2Captcha
        result = self._twocaptcha.turnstile(
            sitekey=challenge.sitekey,
            url=challenge.url,
        )

        # 獲取解決的 token
        token = result.get("code")
        if not token:
            raise CaptchaSolveException("Failed to get token from 2Captcha response")

        logger.info(f"Got token from 2Captcha: {token[:50]}...")

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
            logger.info("Token injected successfully, waiting for page to respond...")
            time.sleep(3)

            # 檢查是否通過驗證

            detector = CaptchaDetector()
            current_det = detector.detect(driver, timeout=1.0)
            if current_det.kind == "none":
                logger.info("Turnstile challenge resolved successfully!")
                return SolveResult(success=True, solver_name="TwoCaptcha")
        else:
            logger.warning("Could not inject token using standard methods")

        return SolveResult(
            success=False,
            error_message="Token injection failed",
            solver_name="TwoCaptcha",
        )

    def _solve_recaptcha_v2(
        self, challenge: ChallengeDetection, driver: Any
    ) -> SolveResult:
        """解決 reCAPTCHA v2"""
        if not challenge.sitekey:
            raise CaptchaSolveException("reCAPTCHA v2 detected but sitekey not found")

        logger.info(f"Solving reCAPTCHA v2 with sitekey: {challenge.sitekey}")

        # 提交驗證任務到 2Captcha
        result = self._twocaptcha.recaptcha(
            sitekey=challenge.sitekey,
            url=challenge.url,
        )

        # 獲取解決的 token
        token = result.get("code")
        if not token:
            raise CaptchaSolveException("Failed to get token from 2Captcha response")

        logger.info(f"Got token from 2Captcha: {token[:50]}...")

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
            logger.info("Token injected successfully, waiting for page to respond...")
            time.sleep(3)

            # 檢查是否通過驗證
            detector = CaptchaDetector()
            current_det = detector.detect(driver, timeout=1.0)
            if current_det.kind == "none":
                logger.info("reCAPTCHA v2 challenge resolved successfully!")
                return SolveResult(success=True, solver_name="TwoCaptcha")
        else:
            logger.warning("Could not inject token using standard methods")

        return SolveResult(
            success=False,
            error_message="Token injection failed",
            solver_name="TwoCaptcha",
        )
