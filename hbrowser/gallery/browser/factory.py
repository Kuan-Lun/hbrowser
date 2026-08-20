"""瀏覽器工廠"""

import asyncio
import os
import platform
import subprocess
from collections.abc import MutableMapping
from typing import Any

import asyncio_atexit  # type: ignore[import-untyped]
import zendriver as zd

from ..utils import (
    is_browser_generation_error,
    log_context,
    setup_logger,
    wait_for_zendriver,
)
from ..utils.mutation import wait_for_zendriver_mutation
from ..utils.protocol import (
    _begin_zendriver_retirement,
    _ZendriverQuiescenceChanged,
    _ZendriverRetirement,
)
from .chrome_manager import ensure_chrome_installed
from .mapper import (
    start_zendriver_mapper_janitor,
    stop_zendriver_mapper_janitor,
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
_BROWSER_STOP_TIMEOUT_SECONDS = 15.0
_BROWSER_START_TIMEOUT_SECONDS = 30.0
_CONNECTION_CLOSE_TIMEOUT_SECONDS = 5.0
_CONNECTION_LISTENER_STOP_TIMEOUT_SECONDS = 2.0
_STOP_TASK_CANCEL_TIMEOUT_SECONDS = 2.0
_JANITOR_STOP_TIMEOUT_SECONDS = 2.0
_TOR_STOP_TIMEOUT_SECONDS = 12.0
_BROWSER_PROCESS_GRACE_SECONDS = 3.0
_BROWSER_PROCESS_KILL_WAIT_SECONDS = 3.0
_PROFILE_CLEANUP_TIMEOUT_SECONDS = 3.0
_PROTOCOL_RETIREMENT_MAX_PASSES = 8
_BROWSER_ATEXIT_CLEANUP_ATTEMPTS = 3
_PAGE_SETUP_MUTATION_TIMEOUT_SECONDS = 15.0
_DRAINING_SHUTDOWN_TASKS: set[asyncio.Future[Any]] = set()
_CONNECTION_CLOSE_TASK_ATTRIBUTE = "_hbrowser_close_task"
_BROWSER_ATEXIT_CLEANUP_ATTRIBUTE = "_hbrowser_asyncio_atexit_cleanup"


def _build_config(
    headless: bool,
    proxy_extension: str | None,
    use_tor: bool = False,
    socks_port: int | None = None,
    chrome_path: str | None = None,
) -> zd.Config:
    config = zd.Config()

    if chrome_path:
        config.browser_executable_path = chrome_path

    config.headless = headless
    config.disable_webrtc = True

    if proxy_extension:
        logger.debug("Using residential proxy extension")
        config.add_extension(proxy_extension)
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


def _attach_tor_process(
    browser: zd.Browser, tor_process: subprocess.Popen[bytes] | None
) -> None:
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
) -> None:
    from zendriver import cdp

    await wait_for_zendriver_mutation(
        page.send(cdp.emulation.set_geolocation_override()),
        timeout=_PAGE_SETUP_MUTATION_TIMEOUT_SECONDS,
        owner=page,
        operation="Browser geolocation reset",
    )

    if use_tor and not has_residential_proxy():
        await verify_proxy_ip(browser, page)


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


async def _terminate_owned_tor_after_cancellation(
    tor_process: Any,
    cancellation: asyncio.CancelledError,
) -> None:
    cleanup_task = asyncio.create_task(
        asyncio.to_thread(terminate_tor_process, tor_process)
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
) -> None:
    try:
        await stop_browser(browser)
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
        startup_error.add_note(
            f"Browser cleanup after {phase} failure also failed: "
            f"{type(cleanup_error).__name__}"
        )


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
                    await stop_browser(browser)
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

    use_tor = should_use_tor()
    tor_process: subprocess.Popen[bytes] | None = None
    socks_port: int | None = None
    if use_tor:
        socks_port = find_available_port(TOR_SOCKS_PORT)
        tor_start_task = asyncio.create_task(
            asyncio.to_thread(start_tor_with_retry, socks_port)
        )
        tor_process, tor_start_cancellation = await _settle_owned_startup_task(
            tor_start_task
        )
        if tor_start_cancellation is not None:
            await _terminate_owned_tor_after_cancellation(
                tor_process,
                tor_start_cancellation,
            )

    try:
        proxy_extension = configure_proxy()
        if proxy_extension is not None:
            connection = "residential proxy"
        elif use_tor:
            connection = "Tor"
        else:
            connection = "direct"
        chrome_paths = ensure_chrome_installed()
        config = _build_config(
            headless, proxy_extension, use_tor, socks_port, chrome_paths.chrome
        )

        logger.debug("Initializing browser")
        browser = zd.Browser(config)
    except BaseException as construction_error:
        if tor_process is not None:
            try:
                await asyncio.to_thread(terminate_tor_process, tor_process)
            except BaseException as cleanup_error:
                construction_error.add_note(
                    "Tor cleanup after browser startup failure also failed: "
                    f"{type(cleanup_error).__name__}"
                )
        raise construction_error.with_traceback(construction_error.__traceback__)

    _attach_tor_process(browser, tor_process)
    try:
        _register_browser_atexit(browser)
        started_browser = await wait_for_zendriver(
            browser.start(),
            timeout=_BROWSER_START_TIMEOUT_SECONDS,
            owner=browser,
        )
        if started_browser is not browser:
            raise RuntimeError("Zendriver Browser.start returned another instance")
    except BaseException as startup_error:
        await _cleanup_browser_after_startup_failure(
            browser,
            startup_error,
            phase="launch",
        )
        raise startup_error.with_traceback(startup_error.__traceback__)

    try:
        page = await _wait_for_main_tab(browser)
        await _post_create_setup(browser, page, use_tor)
        start_zendriver_mapper_janitor(browser)
        logger.info("Browser ready (%s, %s)", mode, connection)
    except BaseException as startup_error:
        await _cleanup_browser_after_startup_failure(
            browser,
            startup_error,
            phase="setup",
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
) -> tuple[bool, list[BaseException]]:
    """Wait for cleanup without letting a cancellation-resistant task hang."""

    task = asyncio.ensure_future(awaitable)
    done, _ = await asyncio.wait((task,), timeout=timeout)
    if not done:
        task.cancel()
        _detach_shutdown_task(task)
        return False, [TimeoutError(f"{description} exceeded {timeout:g} seconds")]
    try:
        task.result()
    except BaseException as error:
        return True, [error]
    return True, []


async def _close_and_wait_for_connection(
    connection: Any,
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
                close_task = asyncio.ensure_future(aclose())
                try:
                    setattr(connection, _CONNECTION_CLOSE_TASK_ATTRIBUTE, close_task)
                except (AttributeError, TypeError) as error:
                    close_task.cancel()
                    errors.append(error)
            _, close_errors = await _bounded_shutdown_awaitable(
                close_task,
                timeout=_CONNECTION_CLOSE_TIMEOUT_SECONDS,
                description="Zendriver connection close",
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
            timeout=_CONNECTION_LISTENER_STOP_TIMEOUT_SECONDS,
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
) -> tuple[bool, list[BaseException]]:
    """Quiesce every captured connection before protocol tasks are cancelled."""

    all_closed = True
    errors: list[BaseException] = []
    for connection in connections:
        connection_closed, connection_errors = await _close_and_wait_for_connection(
            connection
        )
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


def _raise_browser_shutdown_errors(errors: list[BaseException]) -> None:
    if not errors:
        return
    primary, *secondary = errors
    for error in secondary:
        primary.add_note(f"Additional shutdown error: {type(error).__name__}: {error}")
    raise primary


async def _stop_mapper_janitor(
    browser: Any,
    retirement: _ZendriverRetirement,
) -> list[BaseException]:
    if retirement.janitor_cleanup_is_complete():
        return []
    completed, errors = await _bounded_shutdown_awaitable(
        stop_zendriver_mapper_janitor(browser),
        timeout=_JANITOR_STOP_TIMEOUT_SECONDS,
        description="Zendriver mapper janitor stop",
    )
    if completed and not errors:
        retirement.mark_janitor_cleanup_complete()
    return errors


async def _await_browser_stop_task(
    task: asyncio.Future[Any],
) -> BaseException | None:
    done, _ = await asyncio.wait((task,), timeout=_BROWSER_STOP_TIMEOUT_SECONDS)
    if not done:
        _detach_shutdown_task(task)
        return TimeoutError(
            "Zendriver browser stop exceeded "
            f"{_BROWSER_STOP_TIMEOUT_SECONDS:g} seconds"
        )
    try:
        task.result()
    except BaseException as error:
        return error
    return None


def _terminate_browser_process(browser: Any) -> None:
    """Synchronously terminate, kill if needed, and reap Browser._process."""

    process = getattr(browser, "_process", None)
    if process is None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=_BROWSER_PROCESS_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait(timeout=_BROWSER_PROCESS_KILL_WAIT_SECONDS)
    browser._process = None
    if hasattr(browser, "_process_pid"):
        browser._process_pid = None


async def _explicit_browser_cleanup(browser: Any) -> list[BaseException]:
    """Finish process/profile cleanup after a repeatedly failed Browser.stop."""

    errors: list[BaseException] = []
    try:
        await asyncio.to_thread(_terminate_browser_process, browser)
    except BaseException as error:
        errors.append(error)

    cleanup_profile = getattr(browser, "_cleanup_temporary_profile", None)
    if callable(cleanup_profile):
        _, profile_errors = await _bounded_shutdown_awaitable(
            cleanup_profile(),
            timeout=_PROFILE_CLEANUP_TIMEOUT_SECONDS,
            description="Zendriver temporary profile cleanup",
        )
        errors.extend(profile_errors)
    return errors


async def _ensure_browser_cleanup(
    browser: Any,
    retirement: _ZendriverRetirement,
    *,
    allow_fallback: bool,
) -> list[BaseException]:
    """Start, resume, or safely recover the one Browser.stop sequence."""

    if retirement.browser_cleanup_is_complete():
        return []

    task = retirement.existing_browser_stop_task()
    task_started_now = False
    if task is None:
        task = asyncio.ensure_future(browser.stop())
        retirement.bind_browser_stop_task(task)
        task_started_now = True
    elif task.done():
        try:
            task.result()
        except BaseException as previous_error:
            if not allow_fallback:
                return [previous_error]
            task = asyncio.ensure_future(browser.stop())
            retirement.replace_browser_stop_task(task)
            task_started_now = True
        else:
            retirement.mark_browser_cleanup_complete()
            return []

    stop_error = await _await_browser_stop_task(task)
    if stop_error is None:
        retirement.mark_browser_cleanup_complete()
        return []
    if not allow_fallback or not retirement.protocol_is_retired():
        return [stop_error]

    # If a prior attempt completed with an error while this retry awaited it,
    # give Browser.stop one fresh idempotent cleanup attempt before falling back
    # to its explicit process/profile equivalent.
    if not task_started_now and not isinstance(stop_error, TimeoutError):
        replacement = asyncio.ensure_future(browser.stop())
        retirement.replace_browser_stop_task(replacement)
        task = replacement
        stop_error = await _await_browser_stop_task(task)
        if stop_error is None:
            retirement.mark_browser_cleanup_complete()
            return []

    if not task.done():
        task.cancel()
        cancelled, _ = await asyncio.wait(
            (task,),
            timeout=_STOP_TASK_CANCEL_TIMEOUT_SECONDS,
        )
        if not cancelled:
            _detach_shutdown_task(task)
            return [
                stop_error,
                TimeoutError(
                    "Repeatedly timed-out Zendriver Browser.stop task did not "
                    "accept cancellation"
                ),
            ]

    fallback_errors = await _explicit_browser_cleanup(browser)
    if fallback_errors:
        return [stop_error, *fallback_errors]
    retirement.mark_browser_cleanup_complete()
    return []


async def _ensure_tor_cleanup(
    retirement: _ZendriverRetirement,
    tor_process: Any,
) -> list[BaseException]:
    if retirement.tor_cleanup_is_complete():
        return []
    if tor_process is None:
        retirement.mark_tor_cleanup_complete()
        return []

    task = retirement.existing_tor_stop_task()
    if task is None:
        task = asyncio.ensure_future(
            asyncio.to_thread(terminate_tor_process, tor_process)
        )
        retirement.bind_tor_stop_task(task)
    elif task.done():
        try:
            task.result()
        except BaseException:
            task = asyncio.ensure_future(
                asyncio.to_thread(terminate_tor_process, tor_process)
            )
            retirement.replace_tor_stop_task(task)
        else:
            retirement.mark_tor_cleanup_complete()
            return []

    done, _ = await asyncio.wait((task,), timeout=_TOR_STOP_TIMEOUT_SECONDS)
    if not done:
        _detach_shutdown_task(task)
        return [
            TimeoutError(
                f"Tor process termination exceeded {_TOR_STOP_TIMEOUT_SECONDS:g} "
                "seconds"
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
            await _close_and_wait_for_browser_connections(connections)
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
) -> None:
    """Run the first bounded shutdown attempt for one browser generation."""

    errors = await _stop_mapper_janitor(browser, retirement)
    errors.extend(
        await _ensure_browser_cleanup(
            browser,
            retirement,
            allow_fallback=False,
        )
    )
    retirement.capture_connections(
        _merge_connection_snapshots(
            initial_connections,
            _browser_connection_snapshot(browser),
            retirement.owned_connections(),
        )
    )
    errors.extend(await _retire_protocol_operations(browser, retirement))
    errors.extend(await _ensure_tor_cleanup(retirement, tor_process))
    _mark_shutdown_complete_if_ready(browser, retirement)
    _raise_browser_shutdown_errors(errors)


async def _retry_browser_retirement(
    browser: Any,
    retirement: _ZendriverRetirement,
    connections: tuple[Any, ...],
    tor_process: Any,
) -> None:
    """Retry incomplete resource cleanup without reviving the generation."""

    retirement.capture_connections(connections)
    errors = await _stop_mapper_janitor(browser, retirement)
    errors.extend(await _retire_protocol_operations(browser, retirement))
    errors.extend(
        await _ensure_browser_cleanup(
            browser,
            retirement,
            allow_fallback=True,
        )
    )
    errors.extend(await _ensure_tor_cleanup(retirement, tor_process))
    _mark_shutdown_complete_if_ready(browser, retirement)
    _raise_browser_shutdown_errors(errors)


async def _await_cleanup_deferring_caller_cancellation(
    cleanup_task: asyncio.Task[None],
) -> None:
    """Propagate caller cancellation only after the cleanup task terminates."""

    caller_cancellation: asyncio.CancelledError | None = None
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
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


async def stop_browser(browser: Any) -> None:
    """Stop one browser generation, deferring caller cancellation until clean."""

    # Tombstone synchronously before the first shutdown await. This closes the
    # race where another task could register work on a detached target while
    # connection snapshots are being closed.
    retirement = _begin_zendriver_retirement(browser)
    if retirement.is_complete():
        return
    cleanup_task = retirement.existing_shutdown_task()
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
            )
        )
        retirement.replace_completed_shutdown_task(cleanup_task)
    await _await_cleanup_deferring_caller_cancellation(cleanup_task)
