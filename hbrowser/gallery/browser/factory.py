"""瀏覽器工廠"""

import asyncio
import json
import math
import os
import platform
import secrets
import subprocess
import sys
import tempfile
from collections.abc import Awaitable, Callable, MutableMapping
from pathlib import Path
from typing import Any

import asyncio_atexit  # type: ignore[import-untyped]
import zendriver as zd

from ..utils import (
    Deadline,
    is_browser_generation_error,
    log_context,
    setup_logger,
)
from ..utils.mutation import wait_for_zendriver_mutation
from ..utils.protocol import (
    _begin_zendriver_retirement,
    _validate_zendriver_operation,
    _ZendriverQuiescenceChanged,
    _ZendriverRetirement,
)
from .chrome_manager import ChromePaths, create_chrome_install_staging_root
from .mapper import (
    start_zendriver_mapper_janitor,
    stop_zendriver_mapper_janitor,
)
from .process import (
    OwnedProcess,
    ProcessOwnershipError,
    _PrivateDirectory,
    start_owned_browser_process,
    start_owned_process,
)
from .proxy import (
    configure_proxy,
    find_available_port,
    has_residential_proxy,
    verify_proxy_ip,
)
from .tor import (
    TOR_SOCKS_PORT,
    should_use_tor,
    start_tor_with_retry,
    terminate_tor_process,
)

logger = setup_logger(__name__)

MAIN_TAB_WAIT_TIMEOUT = 5.0
MAIN_TAB_POLL_INTERVAL = 0.1
_BROWSER_STOP_TIMEOUT_SECONDS = 5.0
_BROWSER_STARTUP_DEADLINE_SECONDS = 300.0
_BROWSER_STARTUP_CLEANUP_RESERVE_SECONDS = 15.0
_BROWSER_ATTACH_DEADLINE_SECONDS = 30.0
_BROWSER_CONSTRUCTION_DEADLINE_SECONDS = 25.0
_BROWSER_CONSTRUCTION_CLEANUP_RESERVE_SECONDS = 5.0
_PROCESS_OWNERSHIP_START_TIMEOUT_SECONDS = 10.0
_CONNECTION_CLOSE_TIMEOUT_SECONDS = 5.0
_CONNECTION_LISTENER_STOP_TIMEOUT_SECONDS = 2.0
_STOP_TASK_CANCEL_TIMEOUT_SECONDS = 2.0
_JANITOR_STOP_TIMEOUT_SECONDS = 2.0
_BROWSER_PROCESS_NATURAL_EXIT_SECONDS = 3.0
_BROWSER_PROCESS_TERMINATE_WAIT_SECONDS = 3.0
_BROWSER_PROCESS_KILL_WAIT_SECONDS = 3.0
_BROWSER_PRIVATE_RELEASE_TIMEOUT_SECONDS = 5.0
_PROFILE_CLEANUP_TIMEOUT_SECONDS = 3.0
_CHROME_INSTALL_WORKER_READY_SECONDS = 5.0
_CHROME_INSTALL_RECEIPT_MAX_BYTES = 16 * 1024
_PROTOCOL_RETIREMENT_MAX_PASSES = 8
_BROWSER_ATEXIT_CLEANUP_ATTEMPTS = 3
_PAGE_SETUP_MUTATION_TIMEOUT_SECONDS = 5.0
_DRAINING_SHUTDOWN_TASKS: set[asyncio.Future[Any]] = set()
_CONNECTION_CLOSE_TASK_ATTRIBUTE = "_hbrowser_close_task"
_BROWSER_ATEXIT_CLEANUP_ATTRIBUTE = "_hbrowser_asyncio_atexit_cleanup"
_BROWSER_PROCESS_OWNER_ATTRIBUTE = "_hbrowser_process_owner"
_DEVTOOLS_ACTIVE_PORT_TIMEOUT_SECONDS = 15.0
_DEVTOOLS_ACTIVE_PORT_POLL_SECONDS = 0.02
# This bounds the complete local cleanup state machine, not one CDP command.
_BROWSER_SHUTDOWN_DEADLINE_SECONDS = 15.0


class _OwnedZendriverBrowser(zd.Browser):
    """Zendriver connection whose OS process tree is owned by hbrowser."""

    @property
    def stopped(self) -> bool:
        owner = getattr(self, _BROWSER_PROCESS_OWNER_ATTRIBUTE, None)
        if isinstance(owner, OwnedProcess):
            return owner.poll() is not None
        return super().stopped


def _build_config(
    headless: bool,
    proxy_extension: str | None,
    user_data_directory: str,
    use_tor: bool = False,
    socks_port: int | None = None,
    chrome_path: str | None = None,
) -> zd.Config:
    config = zd.Config(user_data_dir=user_data_directory)

    if chrome_path:
        config.browser_executable_path = chrome_path

    config.headless = headless
    config.disable_webrtc = True

    if proxy_extension:
        logger.debug("Using residential proxy extension")
        extension_directory = Path(proxy_extension)
        if not extension_directory.is_dir():
            raise RuntimeError("Proxy extension directory is unavailable")
        config.add_argument(f"--load-extension={extension_directory}")
    elif use_tor and socks_port is not None:
        config.add_argument(f"--proxy-server=socks5://127.0.0.1:{socks_port}")
        logger.debug("Using Tor SOCKS proxy")
        logger.debug("Tor SOCKS proxy endpoint: host=127.0.0.1 port=%d", socks_port)
    else:
        logger.debug("Using a direct connection")

    is_xvfb_env = (
        platform.system() == "Linux"
        and os.environ.get("DISPLAY")
        and ":" in os.environ.get("DISPLAY", "")
    )

    if not proxy_extension:
        config.add_argument("--disable-extensions")
    config.sandbox = False
    config.add_argument("--window-size=1600,900")
    config.add_argument("--disable-dev-shm-usage")

    if headless:
        is_linux_server = platform.system() == "Linux" and (
            not os.environ.get("DISPLAY") or "Xvfb" in os.environ.get("DISPLAY", "")
        )
        if is_linux_server:
            config.add_argument("--disable-gpu")
            config.add_argument("--disable-software-rasterizer")

    if is_xvfb_env and not headless:
        # Xvfb 環境讓 Chrome 使用 SwiftShader 軟體渲染，刻意不加 --disable-gpu，
        # 明確禁用 GPU 反而容易被 Cloudflare 偵測。
        logger.debug("Detected Xvfb environment; retaining default GPU settings")

    config.add_argument("--disable-blink-features=AutomationControlled")
    config.add_argument("--disable-infobars")
    config.add_argument("--disable-notifications")
    config.add_argument("--disable-popup-blocking")

    config.add_argument("--disable-save-password-bubble")
    config.add_argument("--disable-translate")
    config.add_argument("--password-store=basic")

    if platform.system() == "Darwin":
        # 避免 Chrome for Testing 存取系統鑰匙圈時彈出授權提示
        config.add_argument("--use-mock-keychain")

    config.add_argument("--disable-features=WebRtcHideLocalIpsWithMdns")
    config.add_argument("--enforce-webrtc-ip-permission-check")
    config.add_argument("--webrtc-ip-handling-policy=disable_non_proxied_udp")

    return config


def _parse_devtools_active_port(contents: str) -> int:
    lines = contents.splitlines()
    if len(lines) < 2:
        raise RuntimeError("Chrome DevToolsActivePort record is incomplete")
    try:
        port = int(lines[0])
    except ValueError as error:
        raise RuntimeError(
            "Chrome DevToolsActivePort contains an invalid port"
        ) from error
    if not 1 <= port <= 65535:
        raise RuntimeError("Chrome DevToolsActivePort contains an invalid port")
    if not lines[1].startswith("/devtools/browser/"):
        raise RuntimeError("Chrome DevToolsActivePort contains an invalid endpoint")
    return port


async def _wait_for_devtools_active_port_async(
    owner: OwnedProcess,
    user_data_directory: Path,
    *,
    deadline: Deadline,
) -> int:
    """Observe Chrome's local readiness without blocking the event loop."""

    active_port_deadline = deadline.bounded(_DEVTOOLS_ACTIVE_PORT_TIMEOUT_SECONDS)
    active_port_file = user_data_directory / "DevToolsActivePort"
    last_parse_error: RuntimeError | None = None
    while not active_port_deadline.expired:
        if active_port_file.is_file():
            try:
                port = _parse_devtools_active_port(
                    active_port_file.read_text(encoding="utf-8")
                )
                if active_port_deadline.expired:
                    break
                return port
            except (OSError, RuntimeError) as error:
                last_parse_error = (
                    error
                    if isinstance(error, RuntimeError)
                    else RuntimeError("Chrome DevToolsActivePort could not be read")
                )
        returncode = owner.poll()
        if returncode is not None:
            raise RuntimeError(
                "Chrome exited before publishing DevToolsActivePort "
                f"(exit_code={returncode})"
            )
        await asyncio.sleep(
            min(_DEVTOOLS_ACTIVE_PORT_POLL_SECONDS, active_port_deadline.remaining())
        )
    timeout_error = TimeoutError(
        "Chrome did not publish DevToolsActivePort before its shared deadline"
    )
    if last_parse_error is not None:
        timeout_error.add_note(
            "Last DevToolsActivePort error: "
            f"{type(last_parse_error).__name__}: {last_parse_error}"
        )
    raise timeout_error


def _terminate_unbound_owner(owner: OwnedProcess, *, deadline: Deadline) -> None:
    owner.shutdown(
        graceful_timeout=0,
        terminate_timeout=_BROWSER_PROCESS_TERMINATE_WAIT_SECONDS,
        kill_timeout=_BROWSER_PROCESS_KILL_WAIT_SECONDS,
        cleanup_timeout=_BROWSER_PRIVATE_RELEASE_TIMEOUT_SECONDS,
        deadline=deadline.expires_at,
    )


def _prepare_owned_browser_launch(
    config: zd.Config,
) -> tuple[Path, list[str]]:
    if config.host is not None or config.port is not None:
        raise RuntimeError("Owned Chrome launch requires an unbound debugging endpoint")
    if config.lang is not None:
        raise RuntimeError(
            "Owned Chrome launch does not accept implicit language mutation"
        )
    if getattr(config, "_extensions", None):
        raise RuntimeError("Owned Chrome launch requires explicit extension arguments")

    executable = Path(config.browser_executable_path)
    if not executable.is_file():
        raise FileNotFoundError(f"Browser executable is unavailable: {executable}")
    config.add_argument("--remote-debugging-host=127.0.0.1")
    config.add_argument("--remote-debugging-port=0")
    parameters = config()
    parameters.append("about:blank")
    return executable, parameters


async def _construct_owned_browser_async(
    config: zd.Config,
    *,
    cleanup_paths: tuple[str, ...],
    startup_deadline: Deadline,
) -> _OwnedZendriverBrowser:
    """Launch in a bounded owner phase, then await local readiness asynchronously."""

    construction_deadline = startup_deadline.bounded(
        _BROWSER_CONSTRUCTION_DEADLINE_SECONDS
    )
    cleanup_reserve = min(
        _BROWSER_CONSTRUCTION_CLEANUP_RESERVE_SECONDS,
        construction_deadline.remaining() / 2.0,
    )
    construction_work_deadline = Deadline(
        construction_deadline.expires_at - cleanup_reserve
    )

    def require_construction_budget(phase: str) -> None:
        if construction_work_deadline.expired:
            raise TimeoutError(f"Chrome construction deadline expired before {phase}")

    require_construction_budget("configuration validation")
    executable, parameters = _prepare_owned_browser_launch(config)
    require_construction_budget("owned process launch")
    owner: OwnedProcess | None = None
    try:
        launch_task = asyncio.create_task(
            asyncio.to_thread(
                start_owned_browser_process,
                executable,
                parameters,
                cleanup_paths=cleanup_paths,
                startup_timeout=min(
                    _PROCESS_OWNERSHIP_START_TIMEOUT_SECONDS,
                    construction_work_deadline.remaining(),
                ),
                deadline=construction_deadline.expires_at,
            )
        )
        owner, launch_cancellation = await _settle_owned_startup_task(launch_task)
        if launch_cancellation is not None:
            raise launch_cancellation
        assert owner is not None

        require_construction_budget("DevToolsActivePort discovery")
        port = await _wait_for_devtools_active_port_async(
            owner,
            Path(config.user_data_dir),
            deadline=construction_work_deadline,
        )
        require_construction_budget("Zendriver client construction")
        config.host = "127.0.0.1"
        config.port = port
        browser = _OwnedZendriverBrowser(config)
        setattr(browser, _BROWSER_PROCESS_OWNER_ATTRIBUTE, owner)
        require_construction_budget("browser ownership publication")
        owner = None
        return browser
    except BaseException as startup_error:
        if owner is not None:
            cleanup_task = asyncio.create_task(
                asyncio.to_thread(
                    _terminate_unbound_owner,
                    owner,
                    deadline=construction_deadline,
                )
            )
            try:
                _, cleanup_cancellation = await _settle_owned_startup_task(cleanup_task)
            except BaseException as cleanup_error:
                ownership_error = ProcessOwnershipError(
                    "Chrome startup failed and its process ownership remains unresolved"
                )
                ownership_error.add_note(
                    f"Startup failure type: {type(startup_error).__name__}"
                )
                raise ownership_error from cleanup_error
            if cleanup_cancellation is not None and not isinstance(
                startup_error,
                asyncio.CancelledError,
            ):
                cleanup_cancellation.add_note(
                    "Chrome construction was already failing with: "
                    f"{type(startup_error).__name__}"
                )
                raise cleanup_cancellation
        raise startup_error.with_traceback(startup_error.__traceback__)


def _attach_tor_process(browser: zd.Browser, tor_process: OwnedProcess | None) -> None:
    """把 Tor 子程序掛在 browser 物件上，讓 stop_browser 之後能找到它清理。

    一拿到 browser 就立刻呼叫（在任何後續可能失敗的步驟之前），確保只要
    browser 物件存在，stop_browser 就一定能找到並終止對應的 Tor 程序，
    不會因為中途某一步驟拋例外而洩漏。
    """
    if tor_process is not None:
        browser._tor_process = tor_process  # type: ignore[attr-defined]


async def _post_create_setup(
    browser: zd.Browser,
    page: zd.Tab,
    use_tor: bool,
    *,
    deadline: Deadline,
) -> None:
    from zendriver import cdp

    command_timeout = min(
        _PAGE_SETUP_MUTATION_TIMEOUT_SECONDS,
        deadline.remaining(),
    )
    if command_timeout <= 0:
        raise TimeoutError("Browser startup expired before page setup")
    await wait_for_zendriver_mutation(
        page.send(cdp.emulation.set_geolocation_override()),
        timeout=command_timeout,
        owner=page,
        operation="Browser geolocation reset",
    )

    if use_tor and not has_residential_proxy():
        await verify_proxy_ip(browser, page, deadline=deadline)


def _select_main_tab(browser: zd.Browser) -> zd.Tab | None:
    """取得可用的主分頁，並容忍 zendriver 啟動時的 target 同步競態。

    zendriver 的 ``update_targets`` 可能先把初始 page 記成一般 Connection，
    稍後的 TargetCreated 事件才會加入真正的 Tab。這時 ``main_tab`` 可能持續
    回傳 None，即使 ``targets`` 裡已經有可用的 Tab，因此需要額外掃描一次。
    """
    page = browser.main_tab
    if isinstance(page, zd.Tab) and page.type_ == "page":
        return page
    return next(
        (
            target
            for target in browser.targets
            if isinstance(target, zd.Tab) and target.type_ == "page"
        ),
        None,
    )


def _describe_browser_startup_state(browser: zd.Browser) -> str:
    process = getattr(browser, "_process", None)
    returncode = process.poll() if process is not None else None
    targets = [
        f"{type(target).__name__}(type={target.type_!r}, url={target.url!r})"
        for target in browser.targets
    ]
    return (
        f"stopped={browser.stopped}, "
        f"process_returncode={returncode!r}, "
        f"targets=[{', '.join(targets)}]"
    )


async def _wait_for_main_tab(
    browser: zd.Browser,
    *,
    timeout: float = MAIN_TAB_WAIT_TIMEOUT,
    poll_interval: float = MAIN_TAB_POLL_INTERVAL,
) -> zd.Tab:
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int | float)
        or not math.isfinite(float(timeout))
        or not 0 <= timeout <= MAIN_TAB_WAIT_TIMEOUT
    ):
        raise ValueError(
            f"main-tab timeout must be finite and in [0, {MAIN_TAB_WAIT_TIMEOUT:g}]"
        )
    if (
        isinstance(poll_interval, bool)
        or not isinstance(poll_interval, int | float)
        or not math.isfinite(float(poll_interval))
        or poll_interval <= 0
    ):
        raise ValueError("main-tab poll interval must be finite and positive")
    if not browser.stopped:
        page = _select_main_tab(browser)
        if page is not None and not browser.stopped:
            return page

        logger.warning(
            "Browser did not expose a main tab immediately; "
            f"waiting up to {timeout:.1f} seconds"
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        while not browser.stopped:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(poll_interval, remaining))
            if browser.stopped:
                break
            if loop.time() >= deadline:
                break
            page = _select_main_tab(browser)
            if page is not None and not browser.stopped:
                logger.debug("Browser main tab became available after startup delay")
                return page

    state = _describe_browser_startup_state(browser)
    raise RuntimeError(
        f"Browser failed to expose a main tab within {timeout:.1f} seconds "
        f"({state})"
    )


async def create_browser(
    headless: bool = True,
) -> tuple[zd.Browser, zd.Tab]:
    """創建 zendriver Browser 實例。

    Returns:
        (browser, page) tuple
    """
    with log_context(scope="Browser"):
        return await _create_browser(headless)


async def _settle_owned_startup_task[T](
    task: asyncio.Task[T],
    caller_cancellation: asyncio.CancelledError | None = None,
) -> tuple[T, asyncio.CancelledError | None]:
    """Keep awaiting startup after cancellation so its resource is recoverable."""

    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            current_task = asyncio.current_task()
            if task.done() and (current_task is None or current_task.cancelling() == 0):
                break
            if current_task is None or current_task.cancelling() == 0:
                raise
            if caller_cancellation is None:
                caller_cancellation = error
    try:
        result = task.result()
    except BaseException as startup_error:
        if caller_cancellation is not None:
            caller_cancellation.add_note(
                "Cancelled startup also failed: " f"{type(startup_error).__name__}"
            )
            raise caller_cancellation
        raise
    return result, caller_cancellation


async def _await_browser_start(
    browser: zd.Browser,
    *,
    deadline: Deadline,
) -> zd.Browser:
    """Await the composite cold-start lifecycle without a long CDP watchdog."""

    task = asyncio.create_task(browser.start())
    _, _, connection, lifecycle = _validate_zendriver_operation(
        task,
        timeout=5.0,
        owner=browser,
    )
    lifecycle.register(task, connection, browser)
    while not task.done():
        remaining = deadline.remaining()
        if remaining <= 0:
            lifecycle.begin_retirement_for_operation(
                task,
                owner=browser,
                connection=connection,
            )
            timeout_error = TimeoutError(
                "Browser cold-start lifecycle exceeded its shared deadline"
            )
            raise timeout_error
        try:
            await asyncio.wait((task,), timeout=remaining)
        except asyncio.CancelledError:
            lifecycle.begin_retirement_for_operation(
                task,
                owner=browser,
                connection=connection,
            )
            raise
    if deadline.expired:
        lifecycle.begin_retirement_for_operation(
            task,
            owner=browser,
            connection=connection,
        )
        raise TimeoutError(
            "Browser cold-start lifecycle completed after its shared deadline"
        )
    return task.result()


async def _terminate_owned_tor_after_cancellation(
    tor_process: Any,
    cancellation: asyncio.CancelledError,
    *,
    deadline: Deadline,
) -> None:
    cleanup_task = asyncio.create_task(
        asyncio.to_thread(
            terminate_tor_process,
            tor_process,
            deadline=deadline,
        )
    )
    try:
        _, settled_cancellation = await _settle_owned_startup_task(
            cleanup_task,
            cancellation,
        )
    except asyncio.CancelledError:
        raise
    assert settled_cancellation is not None
    raise settled_cancellation


async def _cleanup_browser_after_startup_failure(
    browser: Any,
    startup_error: BaseException,
    *,
    phase: str,
    deadline: Deadline,
) -> None:
    try:
        await stop_browser(browser, deadline)
    except asyncio.CancelledError as cleanup_cancellation:
        if isinstance(startup_error, asyncio.CancelledError):
            startup_error.add_note(
                f"Browser cleanup received another cancellation during {phase}"
            )
            return
        cleanup_cancellation.add_note(
            f"Browser {phase} was already failing with: "
            f"{type(startup_error).__name__}"
        )
        raise
    except BaseException as cleanup_error:
        ownership_error = ProcessOwnershipError(
            f"Browser {phase} failed and generation cleanup remains unresolved"
        )
        ownership_error.add_note(
            "Startup failure type: " f"{type(startup_error).__name__}"
        )
        raise ownership_error from cleanup_error


async def _cleanup_unowned_private_paths(
    paths: tuple[str, ...],
    *,
    deadline: Deadline,
) -> list[BaseException]:
    """Remove pre-process private material through settled owned workers."""

    cleanup_deadline = deadline.bounded(_PROFILE_CLEANUP_TIMEOUT_SECONDS)
    guards: list[_PrivateDirectory] = []
    errors: list[BaseException] = []
    for path in paths:
        try:
            guards.append(_PrivateDirectory.capture(Path(path)))
        except FileNotFoundError:
            pass
        except BaseException as error:
            errors.append(error)

    tasks = tuple(
        asyncio.create_task(
            asyncio.to_thread(
                guard.remove,
                deadline=cleanup_deadline.expires_at,
            )
        )
        for guard in guards
    )
    if not tasks:
        return errors

    aggregate = asyncio.gather(*tasks, return_exceptions=True)
    caller_cancellation: asyncio.CancelledError | None = None
    while not aggregate.done():
        try:
            await asyncio.shield(aggregate)
        except asyncio.CancelledError as error:
            current_task = asyncio.current_task()
            if aggregate.done() and (
                current_task is None or current_task.cancelling() == 0
            ):
                break
            if current_task is None or current_task.cancelling() == 0:
                raise
            if caller_cancellation is None:
                caller_cancellation = error

    errors.extend(
        result for result in aggregate.result() if isinstance(result, BaseException)
    )
    if caller_cancellation is not None:
        if errors:
            errors[0].add_note(
                "Private-path cleanup was cancelled but settled every owned worker"
            )
            raise errors[0]
        raise caller_cancellation
    return errors


def _chrome_install_worker_command(
    receipt_path: Path,
    nonce: str,
    *,
    work_deadline: Deadline,
    staging_directory: Path,
) -> tuple[str, tuple[str, ...]]:
    return (
        sys.executable,
        (
            "-m",
            "hbrowser.gallery.browser._chrome_install_worker",
            str(receipt_path),
            nonce,
            repr(work_deadline.expires_at),
            str(staging_directory),
        ),
    )


async def _install_chrome_in_owned_worker(
    *,
    work_deadline: Deadline,
    cleanup_deadline: Deadline,
) -> ChromePaths:
    """Run shared-cache installation in a killable, fully reaped process tree."""

    receipt_directory = Path(tempfile.mkdtemp(prefix="hbrowser-chrome-install-"))
    receipt_path = receipt_directory / "receipt.json"
    nonce = secrets.token_hex(16)
    staging_directory: Path | None = None
    owner: OwnedProcess | None = None
    result: ChromePaths | None = None
    primary_error: BaseException | None = None
    cleanup_error: BaseException | None = None

    try:
        remaining = work_deadline.remaining()
        if remaining <= 0:
            raise TimeoutError("Chrome install deadline expired before worker startup")
        staging_directory = create_chrome_install_staging_root()
        if work_deadline.expired:
            raise TimeoutError(
                "Chrome install deadline expired while creating staging ownership"
            )
        worker_executable, worker_parameters = _chrome_install_worker_command(
            receipt_path,
            nonce,
            work_deadline=work_deadline,
            staging_directory=staging_directory,
        )
        startup_task = asyncio.create_task(
            asyncio.to_thread(
                start_owned_process,
                worker_executable,
                worker_parameters,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                startup_timeout=min(
                    _CHROME_INSTALL_WORKER_READY_SECONDS,
                    remaining,
                ),
                cleanup_paths=(staging_directory,),
                deadline=cleanup_deadline.expires_at,
            )
        )
        owner, startup_cancellation = await _settle_owned_startup_task(startup_task)
        if startup_cancellation is not None:
            raise startup_cancellation
        assert owner is not None

        while owner.poll() is None:
            remaining = work_deadline.remaining()
            if remaining <= 0:
                raise TimeoutError("Chrome installation worker exceeded its deadline")
            await asyncio.sleep(min(0.05, remaining))

        remaining = cleanup_deadline.remaining()
        if remaining <= 0:
            raise ProcessOwnershipError(
                "Chrome installer exited after its ownership deadline"
            )
        wait_task = asyncio.create_task(
            asyncio.to_thread(
                owner.wait,
                timeout=min(15.0, remaining),
            )
        )
        _, wait_cancellation = await _settle_owned_startup_task(wait_task)
        if wait_cancellation is not None:
            raise wait_cancellation
        owner = None

        if work_deadline.expired:
            raise TimeoutError(
                "Chrome installation receipt arrived after its work deadline"
            )
        encoded_receipt = receipt_path.read_bytes()
        if len(encoded_receipt) > _CHROME_INSTALL_RECEIPT_MAX_BYTES:
            raise RuntimeError("Chrome installation receipt exceeded its size limit")
        if work_deadline.expired:
            raise TimeoutError(
                "Chrome installation receipt read completed after its deadline"
            )
        try:
            receipt: object = json.loads(encoded_receipt)
        except json.JSONDecodeError:
            raise RuntimeError(
                "Chrome installation worker returned invalid JSON"
            ) from None
        if work_deadline.expired:
            raise TimeoutError(
                "Chrome installation receipt parsing completed after its deadline"
            )
        if not isinstance(receipt, dict) or receipt.get("schema") != 1:
            raise RuntimeError("Chrome installation worker returned an invalid receipt")
        if receipt.get("nonce") != nonce:
            raise RuntimeError("Chrome installation receipt identity did not match")
        chrome = receipt.get("chrome")
        version = receipt.get("version")
        if (
            not isinstance(chrome, str)
            or not Path(chrome).is_absolute()
            or not Path(chrome).is_file()
            or not isinstance(version, str)
            or not version
        ):
            raise RuntimeError("Chrome installation receipt was not trustworthy")
        if work_deadline.expired:
            raise TimeoutError(
                "Chrome installation validation completed after its deadline"
            )
        result = ChromePaths(chrome=chrome, version=version)
    except BaseException as error:
        primary_error = error

    if owner is not None:
        remaining = cleanup_deadline.remaining()
        try:
            if remaining <= 0:
                raise ProcessOwnershipError(
                    "Chrome installation worker ownership deadline expired"
                )
            shutdown_task = asyncio.create_task(
                asyncio.to_thread(
                    owner.shutdown,
                    graceful_timeout=0,
                    terminate_timeout=min(3.0, remaining),
                    kill_timeout=min(3.0, remaining),
                    cleanup_timeout=min(5.0, remaining),
                    deadline=cleanup_deadline.expires_at,
                )
            )
            _, shutdown_cancellation = await _settle_owned_startup_task(shutdown_task)
            if shutdown_cancellation is not None and primary_error is None:
                primary_error = shutdown_cancellation
            owner = None
        except BaseException as error:
            cleanup_error = error

    unowned_private_paths = [str(receipt_directory)]
    if (
        staging_directory is not None
        and cleanup_error is None
        and not isinstance(primary_error, ProcessOwnershipError)
    ):
        # A successful wait/shutdown or an ordinary startup failure proves no
        # process can still mutate staging. In the unresolved-ownership case,
        # OwnedProcess retains its identity guard for durable atexit cleanup.
        unowned_private_paths.append(str(staging_directory))
    try:
        private_errors = await _cleanup_unowned_private_paths(
            tuple(unowned_private_paths),
            deadline=cleanup_deadline,
        )
    except BaseException as error:
        private_errors = [error]
    if private_errors and cleanup_error is None:
        cleanup_error = private_errors[0]

    if cleanup_error is not None:
        ownership_error = ProcessOwnershipError(
            "Chrome installation ended with unresolved worker or receipt ownership"
        )
        if primary_error is not None:
            ownership_error.add_note(
                f"Install failure type: {type(primary_error).__name__}"
            )
        raise ownership_error from cleanup_error
    if primary_error is not None:
        raise primary_error.with_traceback(primary_error.__traceback__)
    if result is None:
        raise RuntimeError("Chrome installation produced no result")
    return result


async def _cleanup_failed_construction(
    tor_process: OwnedProcess | None,
    private_paths: tuple[str, ...],
    *,
    deadline: Deadline,
) -> tuple[list[BaseException], BaseException | None]:
    """Clean independent startup resources concurrently under one deadline."""

    private_task = asyncio.create_task(
        _cleanup_unowned_private_paths(private_paths, deadline=deadline)
    )
    tor_task = (
        None
        if tor_process is None
        else asyncio.create_task(
            asyncio.to_thread(
                terminate_tor_process,
                tor_process,
                deadline=deadline,
            )
        )
    )
    if tor_task is None:
        return await private_task, None
    private_result, tor_result = await asyncio.gather(
        private_task,
        tor_task,
        return_exceptions=True,
    )
    private_errors = (
        [private_result]
        if isinstance(private_result, BaseException)
        else private_result
    )
    tor_error = tor_result if isinstance(tor_result, BaseException) else None
    return private_errors, tor_error


type _BlockingCleanupRunner = Callable[[Callable[[], None]], Awaitable[None]]


async def _run_blocking_cleanup_in_thread(operation: Callable[[], None]) -> None:
    await asyncio.to_thread(operation)


async def _run_blocking_cleanup_inline(operation: Callable[[], None]) -> None:
    """Run bounded cleanup after asyncio has shut down its default executor."""

    operation()


class _BrowserAtexitCleanup:
    """Retain a browser only until its complete shutdown has been proven."""

    def __init__(self, browser: zd.Browser, loop: asyncio.AbstractEventLoop) -> None:
        self.browser: zd.Browser | None = browser
        self.loop = loop
        self.running = False

    async def __call__(self) -> None:
        browser = self.browser
        if browser is None:
            return
        self.running = True
        try:
            errors: list[Exception] = []
            for _ in range(_BROWSER_ATEXIT_CLEANUP_ATTEMPTS):
                try:
                    await _stop_browser(
                        browser,
                        run_blocking_cleanup=_run_blocking_cleanup_inline,
                        renew_expired_atexit_attempt=True,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    errors.append(error)
                else:
                    return
            final_error = errors[-1]
            for earlier_error in errors[:-1]:
                final_error.add_note(
                    "Earlier asyncio-atexit browser cleanup failure: "
                    f"{type(earlier_error).__name__}: {earlier_error}"
                )
            raise final_error
        finally:
            self.running = False

    def release(self) -> None:
        if self.browser is None:
            return
        if not self.running:
            asyncio_atexit.unregister(self, loop=self.loop)
        self.browser = None


def _register_browser_atexit(browser: zd.Browser) -> None:
    if getattr(browser, _BROWSER_ATEXIT_CLEANUP_ATTRIBUTE, None) is not None:
        raise RuntimeError("Browser asyncio-atexit cleanup is already registered")
    loop = asyncio.get_running_loop()
    cleanup = _BrowserAtexitCleanup(browser, loop)
    setattr(browser, _BROWSER_ATEXIT_CLEANUP_ATTRIBUTE, cleanup)
    asyncio_atexit.register(cleanup, loop=loop)


def _release_browser_atexit(browser: Any) -> None:
    registered_instances = zd.util.get_registered_instances()
    cleanup = getattr(browser, _BROWSER_ATEXIT_CLEANUP_ATTRIBUTE, None)
    if cleanup is None:
        try:
            registered_instances.discard(browser)
        except TypeError:
            # An unhashable test double cannot have entered Zendriver's set.
            pass
        return
    if not isinstance(cleanup, _BrowserAtexitCleanup):
        raise RuntimeError("Browser asyncio-atexit cleanup state is corrupted")
    cleanup.release()
    setattr(browser, _BROWSER_ATEXIT_CLEANUP_ATTRIBUTE, None)
    registered_instances.discard(browser)


async def _create_browser(headless: bool) -> tuple[zd.Browser, zd.Tab]:
    mode = "headless" if headless else "windowed"
    logger.info("Starting browser")
    # This semantic deadline covers external Tor bootstrap, Chrome metadata /
    # artifact installation, local process launch, and target discovery.
    browser_startup_deadline = Deadline.after(_BROWSER_STARTUP_DEADLINE_SECONDS)
    browser_startup_work_deadline = Deadline(
        browser_startup_deadline.expires_at - _BROWSER_STARTUP_CLEANUP_RESERVE_SECONDS
    )

    use_tor = should_use_tor()
    tor_process: OwnedProcess | None = None
    socks_port: int | None = None
    proxy_extension: str | None = None
    profile_directory: str | None = None
    private_paths_are_owned = False
    if use_tor:
        socks_port = find_available_port(TOR_SOCKS_PORT)
        tor_start_task = asyncio.create_task(
            asyncio.to_thread(
                start_tor_with_retry,
                socks_port,
                deadline=browser_startup_work_deadline,
            )
        )
        tor_process, tor_start_cancellation = await _settle_owned_startup_task(
            tor_start_task
        )
        if tor_start_cancellation is not None:
            await _terminate_owned_tor_after_cancellation(
                tor_process,
                tor_start_cancellation,
                deadline=browser_startup_deadline,
            )

    try:
        proxy_extension = configure_proxy()
        if proxy_extension is not None:
            connection = "residential proxy"
        elif use_tor:
            connection = "Tor"
        else:
            connection = "direct"
        chrome_paths = await _install_chrome_in_owned_worker(
            work_deadline=browser_startup_work_deadline,
            cleanup_deadline=browser_startup_deadline,
        )
        profile_directory = tempfile.mkdtemp(prefix="hbrowser-profile-")
        config = _build_config(
            headless,
            proxy_extension,
            profile_directory,
            use_tor,
            socks_port,
            chrome_paths.chrome,
        )

        logger.debug("Initializing browser")
        browser = await _construct_owned_browser_async(
            config,
            cleanup_paths=tuple(
                path
                for path in (profile_directory, proxy_extension)
                if path is not None
            ),
            startup_deadline=browser_startup_work_deadline,
        )
        private_paths_are_owned = True
    except BaseException as construction_error:
        private_paths = (
            ()
            if private_paths_are_owned
            or isinstance(construction_error, ProcessOwnershipError)
            else tuple(
                path
                for path in (profile_directory, proxy_extension)
                if path is not None
            )
        )
        cleanup_task = asyncio.create_task(
            _cleanup_failed_construction(
                tor_process,
                private_paths,
                deadline=browser_startup_deadline,
            )
        )
        cleanup_result, cleanup_cancellation = await _settle_owned_startup_task(
            cleanup_task,
            (
                construction_error
                if isinstance(construction_error, asyncio.CancelledError)
                else None
            ),
        )
        private_cleanup_errors, tor_cleanup_error = cleanup_result
        if private_cleanup_errors:
            ownership_error = ProcessOwnershipError(
                "Browser construction failed and private generation material "
                "could not be removed"
            )
            ownership_error.add_note(
                "Construction failure type: " f"{type(construction_error).__name__}"
            )
            for additional_error in private_cleanup_errors[1:]:
                private_cleanup_errors[0].add_note(
                    "Additional private cleanup failure: "
                    f"{type(additional_error).__name__}"
                )
            raise ownership_error from private_cleanup_errors[0]
        if tor_cleanup_error is not None:
            ownership_error = ProcessOwnershipError(
                "Browser construction failed and Tor process ownership "
                "remains unresolved"
            )
            ownership_error.add_note(
                "Construction failure type: " f"{type(construction_error).__name__}"
            )
            raise ownership_error from tor_cleanup_error
        if cleanup_cancellation is not None:
            raise cleanup_cancellation
        raise construction_error.with_traceback(construction_error.__traceback__)

    _attach_tor_process(browser, tor_process)

    browser_attach_deadline = browser_startup_work_deadline.bounded(
        _BROWSER_ATTACH_DEADLINE_SECONDS
    )
    try:
        _register_browser_atexit(browser)
        started_browser = await _await_browser_start(
            browser,
            deadline=browser_attach_deadline,
        )
        if started_browser is not browser:
            raise RuntimeError("Zendriver Browser.start returned another instance")
    except BaseException as startup_error:
        await _cleanup_browser_after_startup_failure(
            browser,
            startup_error,
            phase="launch",
            deadline=browser_startup_deadline,
        )
        raise startup_error.with_traceback(startup_error.__traceback__)

    try:
        page = await _wait_for_main_tab(
            browser,
            timeout=min(MAIN_TAB_WAIT_TIMEOUT, browser_attach_deadline.remaining()),
        )
        await _post_create_setup(
            browser,
            page,
            use_tor,
            deadline=browser_attach_deadline,
        )
        start_zendriver_mapper_janitor(browser)
        logger.info("Browser ready (%s, %s)", mode, connection)
    except BaseException as startup_error:
        await _cleanup_browser_after_startup_failure(
            browser,
            startup_error,
            phase="setup",
            deadline=browser_startup_deadline,
        )
        raise startup_error.with_traceback(startup_error.__traceback__)

    return browser, page


def _browser_connection_snapshot(browser: Any) -> tuple[Any, ...]:
    """Return the unique root and target connections currently owned by browser."""

    try:
        attributes = vars(browser)
    except TypeError:
        return ()
    candidates: list[Any] = [attributes.get("connection")]
    targets = attributes.get("targets", ())
    if isinstance(targets, (list, tuple, set, frozenset)):
        candidates.extend(targets)

    seen: set[int] = set()
    connections: list[Any] = []
    for connection in candidates:
        if connection is None or id(connection) in seen:
            continue
        seen.add(id(connection))
        connections.append(connection)
    return tuple(connections)


def _merge_connection_snapshots(*snapshots: tuple[Any, ...]) -> tuple[Any, ...]:
    seen: set[int] = set()
    merged: list[Any] = []
    for snapshot in snapshots:
        for connection in snapshot:
            if id(connection) in seen:
                continue
            seen.add(id(connection))
            merged.append(connection)
    return tuple(merged)


def _connection_transport_closed(connection: Any) -> bool:
    try:
        attributes = vars(connection)
    except TypeError:
        return False
    return "websocket" in attributes and attributes["websocket"] is None


def _connection_listener_task(connection: Any) -> asyncio.Future[Any] | None:
    try:
        listener = vars(connection).get("listener")
        task = None if listener is None else vars(listener).get("task")
    except TypeError:
        return None
    return task if isinstance(task, asyncio.Future) else None


def _observe_shutdown_task(task: asyncio.Future[Any]) -> None:
    _DRAINING_SHUTDOWN_TASKS.discard(task)
    if task.cancelled():
        return
    try:
        task.exception()
    except BaseException:
        pass


def _detach_shutdown_task(task: asyncio.Future[Any]) -> None:
    _DRAINING_SHUTDOWN_TASKS.add(task)
    task.add_done_callback(_observe_shutdown_task)


async def _bounded_shutdown_awaitable(
    awaitable: Any,
    *,
    timeout: float,
    description: str,
    deadline: Deadline | None = None,
) -> tuple[bool, list[BaseException]]:
    """Wait for cleanup without letting a cancellation-resistant task hang."""

    effective_timeout = (
        timeout if deadline is None else min(timeout, deadline.remaining())
    )
    if effective_timeout <= 0:
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        return False, [TimeoutError(f"{description} had no shutdown budget left")]
    task = asyncio.ensure_future(awaitable)
    done, _ = await asyncio.wait((task,), timeout=effective_timeout)
    if not done:
        task.cancel()
        _detach_shutdown_task(task)
        return False, [
            TimeoutError(
                f"{description} exceeded its {effective_timeout:g}-second "
                "shutdown phase"
            )
        ]
    try:
        task.result()
    except BaseException as error:
        return True, [error]
    return True, []


async def _close_and_wait_for_connection(
    connection: Any,
    *,
    deadline: Deadline,
) -> tuple[bool, list[BaseException]]:
    """Close one transport and await its listener's terminal state."""

    errors: list[BaseException] = []
    if not _connection_transport_closed(connection):
        aclose = getattr(connection, "aclose", None)
        if not callable(aclose):
            errors.append(RuntimeError("Zendriver connection has no close operation"))
        else:
            try:
                close_task = vars(connection).get(_CONNECTION_CLOSE_TASK_ATTRIBUTE)
            except TypeError:
                close_task = None
            if not isinstance(close_task, asyncio.Future) or close_task.done():
                if deadline.expired:
                    errors.append(
                        ProcessOwnershipError(
                            "Browser shutdown expired before connection close"
                        )
                    )
                    close_task = None
                else:
                    close_task = asyncio.ensure_future(aclose())
                    try:
                        setattr(
                            connection,
                            _CONNECTION_CLOSE_TASK_ATTRIBUTE,
                            close_task,
                        )
                    except (AttributeError, TypeError) as error:
                        close_task.cancel()
                        errors.append(error)
            if isinstance(close_task, asyncio.Future):
                if deadline.expired and not close_task.done():
                    _detach_shutdown_task(close_task)
                    errors.append(
                        ProcessOwnershipError(
                            "Connection close remained pending at shutdown deadline"
                        )
                    )
                else:
                    _, close_errors = await _bounded_shutdown_awaitable(
                        close_task,
                        timeout=_CONNECTION_CLOSE_TIMEOUT_SECONDS,
                        description="Zendriver connection close",
                        deadline=deadline,
                    )
                    errors.extend(close_errors)

    listener_task = _connection_listener_task(connection)
    if listener_task is not None:
        if not listener_task.done():
            try:
                listener = vars(connection).get("listener")
                cancel = getattr(listener, "cancel", None)
                if callable(cancel):
                    cancel()
                else:
                    listener_task.cancel()
            except BaseException as error:
                errors.append(error)
                listener_task.cancel()
        listener_done, _ = await asyncio.wait(
            (listener_task,),
            timeout=min(
                _CONNECTION_LISTENER_STOP_TIMEOUT_SECONDS,
                deadline.remaining(),
            ),
        )
        if listener_done:
            try:
                listener_task.result()
            except asyncio.CancelledError:
                pass
            except BaseException as error:
                if not is_browser_generation_error(error):
                    errors.append(error)
        else:
            listener_task.cancel()
            _detach_shutdown_task(listener_task)
            errors.append(
                TimeoutError(
                    "Zendriver connection listener did not stop within "
                    f"{_CONNECTION_LISTENER_STOP_TIMEOUT_SECONDS:g} seconds"
                )
            )

    listener_stopped = listener_task is None or listener_task.done()
    return _connection_transport_closed(connection) and listener_stopped, errors


async def _close_and_wait_for_browser_connections(
    connections: tuple[Any, ...],
    *,
    deadline: Deadline,
) -> tuple[bool, list[BaseException]]:
    """Quiesce every captured connection before protocol tasks are cancelled."""

    all_closed = True
    errors: list[BaseException] = []
    results = await asyncio.gather(
        *(
            _close_and_wait_for_connection(connection, deadline=deadline)
            for connection in connections
        ),
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, BaseException):
            all_closed = False
            errors.append(result)
            continue
        connection_closed, connection_errors = result
        all_closed = all_closed and connection_closed
        errors.extend(connection_errors)
    return all_closed, errors


def _clear_closed_connection_mappers(connections: tuple[Any, ...]) -> None:
    """Drop response records only after every captured listener is dead."""

    for connection in connections:
        try:
            mapper = vars(connection).get("mapper")
        except TypeError:
            continue
        if isinstance(mapper, MutableMapping):
            mapper.clear()


def _raise_browser_shutdown_errors(
    errors: list[BaseException],
    *,
    retirement: _ZendriverRetirement,
) -> None:
    if not errors:
        return
    primary, *secondary = errors
    for error in secondary:
        primary.add_note(f"Additional shutdown error: {type(error).__name__}: {error}")
    if retirement.is_complete():
        raise primary
    if isinstance(primary, ProcessOwnershipError):
        raise primary
    ownership_error = ProcessOwnershipError(
        "Browser generation cleanup remains unresolved"
    )
    ownership_error.add_note(
        f"Primary shutdown failure: {type(primary).__name__}: {primary}"
    )
    raise ownership_error from primary


async def _stop_mapper_janitor(
    browser: Any,
    retirement: _ZendriverRetirement,
    *,
    deadline: Deadline,
) -> list[BaseException]:
    if retirement.janitor_cleanup_is_complete():
        return []
    completed, errors = await _bounded_shutdown_awaitable(
        stop_zendriver_mapper_janitor(browser),
        timeout=_JANITOR_STOP_TIMEOUT_SECONDS,
        description="Zendriver mapper janitor stop",
        deadline=deadline,
    )
    if completed and not errors:
        retirement.mark_janitor_cleanup_complete()
    return errors


async def _await_browser_stop_task(
    task: asyncio.Future[Any],
    *,
    deadline: Deadline,
) -> BaseException | None:
    timeout = min(_BROWSER_STOP_TIMEOUT_SECONDS, deadline.remaining())
    done, _ = await asyncio.wait((task,), timeout=timeout)
    if not done:
        _detach_shutdown_task(task)
        return TimeoutError("Zendriver browser stop exceeded " f"{timeout:g} seconds")
    try:
        task.result()
    except BaseException as error:
        return error
    return None


def _terminate_browser_process(browser: Any, *, deadline: Deadline) -> None:
    """Synchronously retire every process owner and its private material."""

    owner = getattr(browser, _BROWSER_PROCESS_OWNER_ATTRIBUTE, None)
    process = getattr(browser, "_process", None)
    if owner is None and process is None:
        return
    if deadline.expired:
        raise ProcessOwnershipError(
            "Browser process cleanup was not started after its deadline"
        )
    if isinstance(owner, OwnedProcess):
        try:
            owner.shutdown(
                graceful_timeout=_BROWSER_PROCESS_NATURAL_EXIT_SECONDS,
                terminate_timeout=_BROWSER_PROCESS_TERMINATE_WAIT_SECONDS,
                kill_timeout=_BROWSER_PROCESS_KILL_WAIT_SECONDS,
                cleanup_timeout=_BROWSER_PRIVATE_RELEASE_TIMEOUT_SECONDS,
                deadline=deadline.expires_at,
            )
        except BaseException as cleanup_error:
            raise ProcessOwnershipError(
                "Browser process/private-material cleanup remains unresolved"
            ) from cleanup_error
        setattr(browser, _BROWSER_PROCESS_OWNER_ATTRIBUTE, None)

    if process is None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(
            timeout=min(
                _BROWSER_PROCESS_TERMINATE_WAIT_SECONDS,
                deadline.remaining(),
            )
        )
    except subprocess.TimeoutExpired:
        if deadline.expired:
            raise ProcessOwnershipError(
                "Browser process termination exhausted the shutdown deadline"
            )
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait(
            timeout=min(
                _BROWSER_PROCESS_KILL_WAIT_SECONDS,
                deadline.remaining(),
            )
        )
    browser._process = None
    if hasattr(browser, "_process_pid"):
        browser._process_pid = None


async def _explicit_browser_cleanup(
    browser: Any,
    *,
    run_blocking_cleanup: _BlockingCleanupRunner,
    deadline: Deadline,
) -> list[BaseException]:
    """Finish process/profile cleanup after a repeatedly failed Browser.stop."""

    errors: list[BaseException] = []
    if deadline.expired:
        errors.append(
            ProcessOwnershipError(
                "Browser shutdown expired before process cleanup was started"
            )
        )
    else:
        try:
            await run_blocking_cleanup(
                lambda: _terminate_browser_process(browser, deadline=deadline)
            )
        except BaseException as error:
            errors.append(error)

    cleanup_profile = getattr(browser, "_cleanup_temporary_profile", None)
    if callable(cleanup_profile):
        if deadline.expired:
            errors.append(
                ProcessOwnershipError(
                    "Browser shutdown expired before profile cleanup was started"
                )
            )
        else:
            _, profile_errors = await _bounded_shutdown_awaitable(
                cleanup_profile(),
                timeout=_PROFILE_CLEANUP_TIMEOUT_SECONDS,
                description="Zendriver temporary profile cleanup",
                deadline=deadline,
            )
            errors.extend(profile_errors)
    return errors


async def _ensure_browser_cleanup(
    browser: Any,
    retirement: _ZendriverRetirement,
    *,
    allow_fallback: bool,
    run_blocking_cleanup: _BlockingCleanupRunner,
    deadline: Deadline,
) -> list[BaseException]:
    """Start, resume, or safely recover the one Browser.stop sequence."""

    if retirement.browser_cleanup_is_complete():
        return []

    task = retirement.existing_browser_stop_task()
    prior_stop_is_still_pending = task is not None and not task.done()
    stop_error: BaseException | None = None
    if task is None:
        if deadline.expired:
            return [
                ProcessOwnershipError(
                    "Browser shutdown expired before Zendriver stop was started"
                )
            ]
        task = asyncio.ensure_future(browser.stop())
        retirement.bind_browser_stop_task(task)
    elif task.done():
        try:
            task.result()
        except BaseException as previous_error:
            if not allow_fallback:
                return [previous_error]
            stop_error = previous_error
        else:
            if deadline.expired:
                return [
                    ProcessOwnershipError(
                        "Browser shutdown expired before process reconciliation"
                    )
                ]
            try:
                await run_blocking_cleanup(
                    lambda: _terminate_browser_process(browser, deadline=deadline)
                )
            except BaseException as process_error:
                return [process_error]
            retirement.mark_browser_cleanup_complete()
            return []

    if stop_error is None:
        if prior_stop_is_still_pending and allow_fallback:
            # The initial shutdown pass already spent the full stop watchdog on
            # this exact task. Do not stack another wait during fallback.
            stop_error = TimeoutError("Prior Zendriver browser stop is still pending")
        else:
            stop_error = await _await_browser_stop_task(task, deadline=deadline)
    if stop_error is None:
        try:
            await run_blocking_cleanup(
                lambda: _terminate_browser_process(browser, deadline=deadline)
            )
        except BaseException as process_error:
            return [process_error]
        retirement.mark_browser_cleanup_complete()
        return []
    if not allow_fallback or not retirement.protocol_is_retired():
        return [stop_error]

    cancellation_errors: list[BaseException] = []
    if not task.done():
        task.cancel()
        cancelled, _ = await asyncio.wait(
            (task,),
            timeout=min(_STOP_TASK_CANCEL_TIMEOUT_SECONDS, deadline.remaining()),
        )
        if not cancelled:
            _detach_shutdown_task(task)
            cancellation_errors.append(
                TimeoutError(
                    "Repeatedly timed-out Zendriver Browser.stop task did not "
                    "accept cancellation"
                )
            )

    fallback_errors = await _explicit_browser_cleanup(
        browser,
        run_blocking_cleanup=run_blocking_cleanup,
        deadline=deadline,
    )
    if fallback_errors:
        return [stop_error, *cancellation_errors, *fallback_errors]
    if cancellation_errors:
        logger.warning(
            "Zendriver Browser.stop did not settle, but explicit process and "
            "profile ownership cleanup completed"
        )
    retirement.mark_browser_cleanup_complete()
    return []


async def _ensure_tor_cleanup(
    retirement: _ZendriverRetirement,
    tor_process: Any,
    *,
    run_blocking_cleanup: _BlockingCleanupRunner,
    deadline: Deadline,
) -> list[BaseException]:
    if retirement.tor_cleanup_is_complete():
        return []
    if tor_process is None:
        retirement.mark_tor_cleanup_complete()
        return []

    task = retirement.existing_tor_stop_task()
    if task is None:
        if deadline.expired:
            return [
                ProcessOwnershipError(
                    "Browser shutdown expired before Tor ownership was released"
                )
            ]
        task = asyncio.ensure_future(
            run_blocking_cleanup(
                lambda: terminate_tor_process(tor_process, deadline=deadline)
            )
        )
        retirement.bind_tor_stop_task(task)
    elif task.done():
        try:
            task.result()
        except BaseException:
            if deadline.expired:
                return [
                    ProcessOwnershipError(
                        "Browser shutdown expired before Tor cleanup retry"
                    )
                ]
            task = asyncio.ensure_future(
                run_blocking_cleanup(
                    lambda: terminate_tor_process(tor_process, deadline=deadline)
                )
            )
            retirement.replace_tor_stop_task(task)
        else:
            retirement.mark_tor_cleanup_complete()
            return []

    done, _ = await asyncio.wait((task,), timeout=deadline.remaining())
    if not done:
        _detach_shutdown_task(task)
        return [
            ProcessOwnershipError(
                "Tor process ownership was not released before the shared "
                "browser shutdown deadline"
            )
        ]
    try:
        task.result()
    except BaseException as error:
        return [error]
    retirement.mark_tor_cleanup_complete()
    return []


async def _retire_protocol_operations(
    browser: Any,
    retirement: _ZendriverRetirement,
    *,
    deadline: Deadline,
) -> list[BaseException]:
    if retirement.protocol_is_retired():
        return []
    errors: list[BaseException] = []
    for _ in range(_PROTOCOL_RETIREMENT_MAX_PASSES):
        retirement.capture_connections(
            _merge_connection_snapshots(
                _browser_connection_snapshot(browser),
                retirement.owned_connections(),
            )
        )
        connections = retirement.captured_connections()
        all_connections_closed, close_errors = (
            await _close_and_wait_for_browser_connections(
                connections,
                deadline=deadline,
            )
        )
        errors.extend(close_errors)
        if not all_connections_closed:
            errors.append(RuntimeError("Zendriver browser transports are still live"))
            return errors

        # Browser.start and target discovery can publish a connection while an
        # earlier snapshot is being closed. Capture again before authorizing
        # operation cancellation and repeat until no transport was added.
        retirement.capture_connections(
            _merge_connection_snapshots(
                _browser_connection_snapshot(browser),
                retirement.owned_connections(),
            )
        )
        refreshed_connections = retirement.captured_connections()
        if len(refreshed_connections) != len(connections):
            _clear_closed_connection_mappers(connections)
            continue

        try:
            await retirement.retire_operations(
                quiescent_connections=connections,
            )
        except _ZendriverQuiescenceChanged as error:
            _clear_closed_connection_mappers(connections)
            retirement.capture_connections(
                _merge_connection_snapshots(
                    _browser_connection_snapshot(browser),
                    retirement.owned_connections(),
                )
            )
            if len(retirement.captured_connections()) == len(connections):
                errors.append(error)
                return errors
            continue
        except BaseException as error:
            errors.append(error)
        finally:
            _clear_closed_connection_mappers(connections)
        return errors

    errors.append(
        RuntimeError("Zendriver transports did not reach a stable shutdown snapshot")
    )
    return errors


def _mark_shutdown_complete_if_ready(
    browser: Any,
    retirement: _ZendriverRetirement,
) -> None:
    if (
        retirement.protocol_is_retired()
        and retirement.browser_cleanup_is_complete()
        and retirement.tor_cleanup_is_complete()
        and retirement.janitor_cleanup_is_complete()
    ):
        _release_browser_atexit(browser)
        retirement.mark_complete()


async def _stop_browser_cleanup(
    browser: Any,
    retirement: _ZendriverRetirement,
    initial_connections: tuple[Any, ...],
    tor_process: Any,
    run_blocking_cleanup: _BlockingCleanupRunner,
    deadline: Deadline,
) -> None:
    """Run the first bounded shutdown attempt for one browser generation."""

    errors = await _stop_mapper_janitor(
        browser,
        retirement,
        deadline=deadline,
    )
    initial_browser_errors = await _ensure_browser_cleanup(
        browser,
        retirement,
        allow_fallback=False,
        run_blocking_cleanup=run_blocking_cleanup,
        deadline=deadline,
    )
    retirement.capture_connections(
        _merge_connection_snapshots(
            initial_connections,
            _browser_connection_snapshot(browser),
            retirement.owned_connections(),
        )
    )
    protocol_errors = await _retire_protocol_operations(
        browser,
        retirement,
        deadline=deadline,
    )
    errors.extend(protocol_errors)

    # Browser.stop gets only a short command/composite watchdog. Once every
    # protocol connection is quiescent, use the remainder of this *same*
    # shutdown deadline to release process/profile ownership. A late protocol
    # response must not require a second public stop call before fallback runs.
    if retirement.protocol_is_retired():
        fallback_errors = await _ensure_browser_cleanup(
            browser,
            retirement,
            allow_fallback=True,
            run_blocking_cleanup=run_blocking_cleanup,
            deadline=deadline,
        )
        if fallback_errors:
            errors.extend(initial_browser_errors)
            errors.extend(fallback_errors)
    else:
        errors.extend(initial_browser_errors)
    errors.extend(
        await _ensure_tor_cleanup(
            retirement,
            tor_process,
            run_blocking_cleanup=run_blocking_cleanup,
            deadline=deadline,
        )
    )
    _mark_shutdown_complete_if_ready(browser, retirement)
    _raise_browser_shutdown_errors(errors, retirement=retirement)


async def _retry_browser_retirement(
    browser: Any,
    retirement: _ZendriverRetirement,
    connections: tuple[Any, ...],
    tor_process: Any,
    run_blocking_cleanup: _BlockingCleanupRunner,
    deadline: Deadline,
) -> None:
    """Retry incomplete resource cleanup without reviving the generation."""

    retirement.capture_connections(connections)
    errors = await _stop_mapper_janitor(
        browser,
        retirement,
        deadline=deadline,
    )
    errors.extend(
        await _retire_protocol_operations(
            browser,
            retirement,
            deadline=deadline,
        )
    )
    errors.extend(
        await _ensure_browser_cleanup(
            browser,
            retirement,
            allow_fallback=True,
            run_blocking_cleanup=run_blocking_cleanup,
            deadline=deadline,
        )
    )
    errors.extend(
        await _ensure_tor_cleanup(
            retirement,
            tor_process,
            run_blocking_cleanup=run_blocking_cleanup,
            deadline=deadline,
        )
    )
    _mark_shutdown_complete_if_ready(browser, retirement)
    _raise_browser_shutdown_errors(errors, retirement=retirement)


async def _await_cleanup_deferring_caller_cancellation(
    cleanup_task: asyncio.Task[None],
    deadline: Deadline,
) -> None:
    """Bound caller latency while allowing one shared cleanup task to finish."""

    caller_cancellation: asyncio.CancelledError | None = None
    while not cleanup_task.done():
        remaining = deadline.remaining()
        if remaining <= 0:
            _detach_shutdown_task(cleanup_task)
            timeout_error = ProcessOwnershipError(
                "Browser shutdown exceeded its overall deadline; durable "
                "atexit ownership remains registered"
            )
            if caller_cancellation is not None:
                caller_cancellation.add_note(str(timeout_error))
                raise caller_cancellation
            raise timeout_error
        try:
            await asyncio.wait((cleanup_task,), timeout=remaining)
        except asyncio.CancelledError as error:
            current_task = asyncio.current_task()
            if current_task is None or current_task.cancelling() == 0:
                raise
            if caller_cancellation is None:
                caller_cancellation = error

    if caller_cancellation is not None:
        try:
            cleanup_task.result()
        except BaseException as cleanup_error:
            caller_cancellation.add_note(
                "Browser cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise caller_cancellation
    cleanup_task.result()


async def _stop_browser(
    browser: Any,
    *,
    run_blocking_cleanup: _BlockingCleanupRunner,
    renew_expired_atexit_attempt: bool = False,
    deadline: Deadline | None = None,
) -> None:
    """Stop one generation using the supplied blocking-cleanup execution mode."""

    # Tombstone synchronously before the first shutdown await. This closes the
    # race where another task could register work on a detached target while
    # connection snapshots are being closed.
    retirement = _begin_zendriver_retirement(browser)
    cleanup_task = retirement.existing_shutdown_task()
    proposed_deadline = (
        Deadline.after(_BROWSER_SHUTDOWN_DEADLINE_SECONDS)
        if deadline is None
        else deadline.bounded(_BROWSER_SHUTDOWN_DEADLINE_SECONDS)
    )
    bound_expires_at = retirement.bind_shutdown_deadline(proposed_deadline.expires_at)
    if (
        renew_expired_atexit_attempt
        and cleanup_task is not None
        and cleanup_task.done()
        and Deadline(bound_expires_at).expired
        and not retirement.is_complete()
    ):
        # Normal/concurrent close callers cannot reset a generation's budget.
        # asyncio-atexit owns one explicit new terminal attempt only after the
        # previous cleanup task has definitively stopped.
        bound_expires_at = retirement.replace_shutdown_deadline(
            proposed_deadline.expires_at
        )
    shutdown_deadline = Deadline(bound_expires_at)
    if retirement.is_complete():
        return
    if shutdown_deadline.expired and (cleanup_task is None or cleanup_task.done()):
        raise ProcessOwnershipError(
            "Browser shutdown deadline expired; durable atexit ownership remains"
        )
    if cleanup_task is None:
        initial_connections = _merge_connection_snapshots(
            _browser_connection_snapshot(browser),
            retirement.owned_connections(),
        )
        retirement.capture_connections(initial_connections)
        cleanup_task = asyncio.create_task(
            _stop_browser_cleanup(
                browser,
                retirement,
                initial_connections,
                getattr(browser, "_tor_process", None),
                run_blocking_cleanup,
                shutdown_deadline,
            )
        )
        retirement.bind_shutdown_task(cleanup_task)
    elif cleanup_task.done():
        retry_connections = _merge_connection_snapshots(
            retirement.captured_connections(),
            _browser_connection_snapshot(browser),
            retirement.owned_connections(),
        )
        retirement.capture_connections(retry_connections)
        cleanup_task = asyncio.create_task(
            _retry_browser_retirement(
                browser,
                retirement,
                retirement.captured_connections(),
                getattr(browser, "_tor_process", None),
                run_blocking_cleanup,
                shutdown_deadline,
            )
        )
        retirement.replace_completed_shutdown_task(cleanup_task)
    await _await_cleanup_deferring_caller_cancellation(
        cleanup_task,
        shutdown_deadline,
    )


async def stop_browser(
    browser: Any,
    deadline: Deadline | None = None,
) -> None:
    """Stop one browser generation, deferring caller cancellation until clean."""

    await _stop_browser(
        browser,
        run_blocking_cleanup=_run_blocking_cleanup_in_thread,
        deadline=deadline,
    )
