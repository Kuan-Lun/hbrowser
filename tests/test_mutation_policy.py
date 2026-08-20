import ast
import asyncio
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from hbrowser import BrowserMutationOutcomeUnknownError
from hbrowser.gallery import driver_base, eh_driver
from hbrowser.gallery.browser import ban_handler, factory, owner, proxy
from hbrowser.gallery.captcha import login_challenge
from hbrowser.gallery.utils.mutation import wait_for_zendriver_mutation
from hbrowser.gallery.utils.protocol import (
    _LIFECYCLE_ATTRIBUTE,
    ZendriverOwnerRetiredError,
    _begin_zendriver_retirement,
    wait_for_zendriver,
)

_PROJECT_ROOT = Path(__file__).parents[1]
_STATE_CHANGING_METHODS = frozenset(
    {"activate", "apply", "click", "close", "get", "reload", "send_keys"}
)
_STATE_CHANGING_CDP_COMMANDS = frozenset(
    {"set_cookies", "set_geolocation_override", "set_user_agent_override"}
)
_OUTCOME_UNKNOWN_MARKERS = frozenset(
    {
        "ArchiveDownloadOutcomeUnknownError",
        "BrowserIdentityApplyException",
        "BrowserMutationOutcomeUnknownError",
        "LoginTokenInjectionOutcomeUnknownError",
    }
)


def _called_name(call: ast.Call) -> str | None:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


class BrowserMutationBudgetTests(unittest.TestCase):
    def test_every_fixed_browser_mutation_watchdog_is_fifteen_seconds(self) -> None:
        budgets = {
            "archive": eh_driver._PAGE_MUTATION_TIMEOUT_SECONDS,
            "ban-navigation": ban_handler._PAGE_MUTATION_TIMEOUT_SECONDS,
            "browser-geolocation": factory._PAGE_SETUP_MUTATION_TIMEOUT_SECONDS,
            "browser-tab-open": owner._TAB_OPEN_TIMEOUT_SECONDS,
            "browser-tab-navigation": owner._TAB_NAVIGATION_TIMEOUT_SECONDS,
            "driver-identity": driver_base._IDENTITY_MUTATION_TIMEOUT_SECONDS,
            "driver-page": driver_base._PAGE_MUTATION_TIMEOUT_SECONDS,
            "login-token": login_challenge._TOKEN_MUTATION_TIMEOUT_SECONDS,
            "proxy-navigation": proxy._PROXY_NAVIGATION_TIMEOUT_SECONDS,
        }

        self.assertEqual(budgets, dict.fromkeys(budgets, 15.0))


class BrowserMutationArchitectureTests(unittest.TestCase):
    def test_state_changing_calls_never_use_the_read_watchdog(self) -> None:
        violations: list[str] = []
        for path in sorted((_PROJECT_ROOT / "hbrowser").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
                if _called_name(call) != "wait_for_zendriver" or not call.args:
                    continue
                operation = call.args[0]
                nested_calls = [
                    node for node in ast.walk(operation) if isinstance(node, ast.Call)
                ]
                called_names = {_called_name(node) for node in nested_calls}
                is_page_mutation = bool(called_names & _STATE_CHANGING_METHODS)
                is_cdp_mutation = bool(called_names & _STATE_CHANGING_CDP_COMMANDS)
                if is_page_mutation or is_cdp_mutation:
                    violations.append(
                        f"{path.relative_to(_PROJECT_ROOT)}:{call.lineno}"
                    )

        self.assertEqual(violations, [])

    def test_every_outcome_unknown_marker_path_retires_at_its_source(self) -> None:
        violations: list[str] = []
        for path in sorted((_PROJECT_ROOT / "hbrowser").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            functions = (
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
            for function in functions:
                calls = [
                    node for node in ast.walk(function) if isinstance(node, ast.Call)
                ]
                called_names = {_called_name(call) for call in calls}
                if not (called_names & _OUTCOME_UNKNOWN_MARKERS):
                    continue
                if called_names.isdisjoint(
                    {
                        "_begin_zendriver_retirement",
                        "_retire_validated_zendriver_owner",
                        "wait_for_zendriver_mutation",
                    }
                ):
                    violations.append(
                        f"{path.relative_to(_PROJECT_ROOT)}:{function.lineno}:"
                        f"{function.name}"
                    )

        self.assertEqual(violations, [])


class BrowserMutationRetirementTests(unittest.IsolatedAsyncioTestCase):
    async def test_lifecycle_attach_failure_is_local_and_closes_coroutine(
        self,
    ) -> None:
        class _RejectLifecycleAttachment:
            def __setattr__(self, name: str, value: object) -> None:
                if name == _LIFECYCLE_ATTRIBUTE:
                    raise AttributeError("lifecycle attachment rejected")
                super().__setattr__(name, value)

        owner = _RejectLifecycleAttachment()
        started = False

        async def mutation() -> None:
            nonlocal started
            started = True

        coroutine = mutation()
        with self.assertRaisesRegex(
            TypeError,
            "must support attached lifecycle state",
        ):
            await wait_for_zendriver_mutation(
                coroutine,
                timeout=1,
                owner=owner,
                operation="Unattachable mutation",
            )

        self.assertFalse(started)
        self.assertEqual(inspect.getcoroutinestate(coroutine), inspect.CORO_CLOSED)
        self.assertNotIn(_LIFECYCLE_ATTRIBUTE, vars(owner))

    async def test_invalid_local_timeout_never_tombstones_or_starts_operation(
        self,
    ) -> None:
        browser = SimpleNamespace()
        started = False

        async def mutation() -> None:
            nonlocal started
            started = True

        coroutine = mutation()
        with self.assertRaises(ValueError):
            await wait_for_zendriver_mutation(
                coroutine,
                timeout=-1,
                owner=browser,
                operation="Invalid mutation",
            )

        self.assertFalse(started)
        self.assertEqual(inspect.getcoroutinestate(coroutine), inspect.CORO_CLOSED)
        self.assertNotIn(_LIFECYCLE_ATTRIBUTE, vars(browser))
        self.assertEqual(
            await wait_for_zendriver(
                asyncio.sleep(0, result="usable"), timeout=1, owner=browser
            ),
            "usable",
        )

    async def test_stale_element_failure_retires_only_its_origin_generation(
        self,
    ) -> None:
        browser_a = SimpleNamespace()
        browser_b = SimpleNamespace()
        page_a = SimpleNamespace(
            browser=browser_a,
            websocket=object(),
            mapper={},
        )
        page_b = SimpleNamespace(
            browser=browser_b,
            websocket=object(),
            mapper={},
        )
        stale_element = SimpleNamespace(_tab=page_a)

        async def failed_mutation() -> None:
            raise RuntimeError("remote outcome unavailable")

        with self.assertRaises(BrowserMutationOutcomeUnknownError):
            await wait_for_zendriver_mutation(
                failed_mutation(),
                timeout=1,
                owner=stale_element,
                operation="Stale element click",
            )

        rejected_started = False

        async def rejected_mutation() -> None:
            nonlocal rejected_started
            rejected_started = True

        rejected = rejected_mutation()
        with self.assertRaises(ZendriverOwnerRetiredError):
            await wait_for_zendriver(rejected, timeout=1, owner=page_a)
        self.assertFalse(rejected_started)
        self.assertEqual(inspect.getcoroutinestate(rejected), inspect.CORO_CLOSED)

        self.assertEqual(
            await wait_for_zendriver(
                asyncio.sleep(0, result="browser-b-live"),
                timeout=1,
                owner=page_b,
            ),
            "browser-b-live",
        )
        lifecycle_b = vars(browser_b)[_LIFECYCLE_ATTRIBUTE]
        self.assertFalse(lifecycle_b.retired)

        retirement_a = _begin_zendriver_retirement(browser_a)
        self.assertEqual(retirement_a.captured_connections(), (page_a,))
        await retirement_a.retire_operations(quiescent_connections=(page_a,))

    async def test_owner_rebind_during_failure_cannot_retire_replacement(
        self,
    ) -> None:
        browser_a = SimpleNamespace()
        browser_b = SimpleNamespace()
        page_a = SimpleNamespace(
            browser=browser_a,
            websocket=object(),
            mapper={},
        )
        page_b = SimpleNamespace(
            browser=browser_b,
            websocket=object(),
            mapper={},
        )
        element = SimpleNamespace(_tab=page_a)

        async def rebind_then_fail() -> None:
            element._tab = page_b
            raise RuntimeError("mutation failed after owner rebind")

        with self.assertRaises(BrowserMutationOutcomeUnknownError):
            await wait_for_zendriver_mutation(
                rebind_then_fail(),
                timeout=1,
                owner=element,
                operation="Rebound element mutation",
            )

        lifecycle_a = vars(browser_a)[_LIFECYCLE_ATTRIBUTE]
        self.assertTrue(lifecycle_a.retired)
        self.assertEqual(lifecycle_a.shutdown_connections, [page_a])
        self.assertNotIn(_LIFECYCLE_ATTRIBUTE, vars(browser_b))

        rejected = asyncio.sleep(0)
        with self.assertRaises(ZendriverOwnerRetiredError):
            await wait_for_zendriver(rejected, timeout=1, owner=page_a)
        self.assertEqual(inspect.getcoroutinestate(rejected), inspect.CORO_CLOSED)
        self.assertEqual(
            await wait_for_zendriver(
                asyncio.sleep(0, result="replacement-live"),
                timeout=1,
                owner=page_b,
            ),
            "replacement-live",
        )
        self.assertFalse(vars(browser_b)[_LIFECYCLE_ATTRIBUTE].retired)

        retirement_a = _begin_zendriver_retirement(browser_a)
        await retirement_a.retire_operations(quiescent_connections=(page_a,))

    async def test_root_owner_failure_captures_its_exact_root_connection(
        self,
    ) -> None:
        browser = SimpleNamespace()
        root = SimpleNamespace(
            _owner=browser,
            websocket=object(),
            mapper={},
        )
        browser.connection = root

        async def failed_mutation() -> None:
            raise RuntimeError("root mutation failed")

        with self.assertRaises(BrowserMutationOutcomeUnknownError):
            await wait_for_zendriver_mutation(
                failed_mutation(),
                timeout=1,
                owner=browser,
                operation="Root mutation",
            )

        retirement = _begin_zendriver_retirement(browser)
        self.assertEqual(retirement.captured_connections(), (root,))
        await retirement.retire_operations(quiescent_connections=(root,))

    async def test_detached_failed_mutation_transport_is_closed_by_shutdown(
        self,
    ) -> None:
        browser = SimpleNamespace(
            connection=None,
            targets=[],
            _tor_process=None,
            stop=AsyncMock(),
        )
        target = SimpleNamespace(
            browser=browser,
            websocket=object(),
            mapper={},
            listener=None,
        )

        async def close_target() -> None:
            target.websocket = None

        target.aclose = AsyncMock(side_effect=close_target)

        async def failed_mutation() -> None:
            raise RuntimeError("detached mutation failed")

        with self.assertRaises(BrowserMutationOutcomeUnknownError):
            await wait_for_zendriver_mutation(
                failed_mutation(),
                timeout=1,
                owner=target,
                operation="Detached target mutation",
            )

        await factory.stop_browser(browser)

        target.aclose.assert_awaited_once_with()
        self.assertIsNone(target.websocket)


if __name__ == "__main__":
    unittest.main()
