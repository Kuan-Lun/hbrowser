from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from hbrowser import BrowserMutationOutcomeUnknownError, ProcessOwnershipError
from hbrowser.gallery.browser import proxy
from hbrowser.gallery.browser.process import _PrivateDirectory
from hbrowser.gallery.utils import Deadline


class ProxyLoggingTests(unittest.TestCase):
    def test_credentials_are_only_returned_to_the_configured_proxy(self) -> None:
        # Execute the generated extension, including the registered callback,
        # without opening a browser or contacting any network service.
        harness = """
const fs = require('node:fs');
const vm = require('node:vm');
let authenticate;
const context = {chrome: {
    runtime: {},
    proxy: {settings: {set(_settings, done) {done();}}},
    webRequest: {onAuthRequired: {addListener(callback) {authenticate = callback;}}}
}};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), context);
const cases = JSON.parse(fs.readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(cases.map(details => {
    let response;
    authenticate(details, value => {response = value;});
    return response;
})));
"""
        details: list[dict[str, object]] = [
            {"isProxy": False, "challenger": {"host": "proxy.example", "port": 8443}},
            {"isProxy": True, "challenger": {"host": "origin.example", "port": 8443}},
            {"isProxy": True, "challenger": {"host": "proxy.example", "port": 443}},
            {"isProxy": True},
            {"challenger": {"host": "proxy.example", "port": 8443}},
            {"isProxy": True, "challenger": {"host": "PROXY.EXAMPLE", "port": 8443}},
        ]
        guard = proxy._create_proxy_extension("proxy.example", 8443, "u", "p")
        directory = guard.path
        self.addCleanup(lambda: guard.remove(deadline=Deadline.after(5).expires_at))
        result = subprocess.run(
            ["node", "-e", harness, str(directory / "background.js")],
            input=json.dumps(details),
            text=True,
            capture_output=True,
            check=True,
            timeout=5,
        )
        self.assertEqual(
            json.loads(result.stdout),
            [
                {},
                {},
                {},
                {},
                {},
                {"authCredentials": {"username": "u", "password": "p"}},
            ],
        )

    def test_extension_receipt_requires_effective_config_and_auth_listener(
        self,
    ) -> None:
        harness = """
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[1], 'utf8');
const token = fs.readFileSync(process.argv[2], 'utf8');
async function probe(change) {
    let callback;
    let configured;
    const runtime = {};
    const context = {chrome: {runtime,
        proxy: {settings: {
            set(value, done) {
                configured = value.value;
                if (change === 'set-error') runtime.lastError = {message: 'failed'};
                done();
                delete runtime.lastError;
            },
            get(_request, done) {
                if (change === 'direct') configured.mode = 'direct';
                if (change === 'wrong-server') configured.rules.singleProxy.port = 1;
                if (change === 'bypass') configured.rules.bypassList = ['*'];
                done({value: configured, levelOfControl:
                    change === 'other-extension' ? 'controlled_by_other_extensions'
                                                : 'controlled_by_this_extension'});
            }
        }},
        webRequest: {onAuthRequired: {
            addListener(value) {callback = value;},
            hasListener(value) {return change !== 'no-listener' && value === callback;}
        }}
    }};
    vm.runInNewContext(source, context);
    return await context.hbrowserProxyReady(change === 'wrong-token' ? 'other' : token);
}
(async () => process.stdout.write(JSON.stringify(await Promise.all([
    'valid', 'set-error', 'direct', 'wrong-server', 'bypass',
    'other-extension', 'no-listener', 'wrong-token'
].map(probe)))))();
"""
        guard = proxy._create_proxy_extension("proxy.example", 8443, "u", "p")
        directory = guard.path
        self.addCleanup(lambda: guard.remove(deadline=Deadline.after(5).expires_at))
        manifest = json.loads((directory / "manifest.json").read_text())
        self.assertEqual(manifest["manifest_version"], 3)
        self.assertIn("webRequestAuthProvider", manifest["permissions"])
        self.assertNotIn("webRequestBlocking", manifest["permissions"])
        result = subprocess.run(
            [
                "node",
                "-e",
                harness,
                str(directory / "background.js"),
                str(directory / "verification-token"),
            ],
            text=True,
            capture_output=True,
            check=True,
            timeout=5,
        )
        self.assertEqual(json.loads(result.stdout), [True] + [False] * 7)

    def test_authenticated_proxy_log_does_not_include_credentials(self) -> None:
        logger = Mock()
        with (
            patch.dict(
                "os.environ",
                {
                    "RP_USERNAME": "secret-user",
                    "RP_PASSWORD": "secret-password",
                    "RP_DNS": "proxy.example:8443",
                },
                clear=False,
            ),
            patch.object(proxy, "logger", logger),
            patch.object(
                proxy,
                "_create_proxy_extension",
                return_value=Mock(spec=_PrivateDirectory),
            ),
        ):
            result = proxy.configure_proxy()

        self.assertIsInstance(result, _PrivateDirectory)
        logger.debug.assert_any_call("Using authenticated residential proxy")
        rendered_arguments = repr(logger.method_calls)
        self.assertNotIn("secret-user", rendered_arguments)
        self.assertNotIn("secret-password", rendered_arguments)
        self.assertNotIn("proxy.example", rendered_arguments)
        self.assertNotIn("8443", rendered_arguments)

    def test_proxy_extension_uses_json_literals_for_untrusted_configuration(
        self,
    ) -> None:
        proxy_host = 'proxy"host\\edge'
        proxy_user = 'user"name\nnext'
        proxy_password = 'pass\\word"\nnext'
        with tempfile.TemporaryDirectory(prefix="hbrowser-proxy-test-") as directory:
            plugin_directory = Path(directory) / "plugin"
            plugin_directory.mkdir()
            with patch.object(
                tempfile,
                "mkdtemp",
                return_value=str(plugin_directory),
            ):
                result = proxy._create_proxy_extension(
                    proxy_host,
                    8443,
                    proxy_user,
                    proxy_password,
                )

            try:
                background = (result.path / "background.js").read_text(encoding="utf-8")
                self.assertIn(f"host: {json.dumps(proxy_host)}", background)
                self.assertIn(f"username: {json.dumps(proxy_user)}", background)
                self.assertIn(f"password: {json.dumps(proxy_password)}", background)
            finally:
                result.remove(deadline=Deadline.after(5).expires_at)

    def test_proxy_extension_cleanup_failure_is_an_ownership_error(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-proxy-test-") as directory:
            plugin_directory = Path(directory) / "plugin"
            plugin_directory.mkdir()
            guard = _PrivateDirectory.capture(plugin_directory)
            creation_error = OSError("write failed")
            cleanup_error = OSError("remove failed")
            try:
                with (
                    patch.object(
                        tempfile,
                        "mkdtemp",
                        return_value=str(plugin_directory),
                    ),
                    patch.object(
                        _PrivateDirectory, "capture", return_value=guard
                    ) as capture,
                    patch.object(Path, "write_text", side_effect=creation_error),
                    patch.object(
                        _PrivateDirectory, "remove", side_effect=cleanup_error
                    ) as remove,
                    self.assertRaises(ProcessOwnershipError) as raised,
                ):
                    deadline = Deadline.after(1)
                    proxy._create_proxy_extension(
                        "host", 8080, "user", "password", cleanup_deadline=deadline
                    )

                capture.assert_called_once_with(plugin_directory)
                self.assertIs(raised.exception.__cause__, cleanup_error)
                self.assertLessEqual(
                    remove.call_args.kwargs["deadline"], deadline.expires_at
                )
                self.assertIn(
                    "Creation failure type: OSError",
                    raised.exception.__notes__,
                )
            finally:
                # The failed remove was a Mock; the fixture still owns this pin.
                guard.remove(deadline=Deadline.after(5).expires_at)

    def test_extension_write_failure_removes_only_its_captured_directory(self) -> None:
        for symlink in (False, True):
            with self.subTest(symlink=symlink):
                self._assert_extension_cleanup_preserves_replacement(
                    symlink, final_write_succeeds=False
                )

    def test_successful_final_write_does_not_transfer_replaced_directory(self) -> None:
        for symlink in (False, True):
            with self.subTest(symlink=symlink):
                self._assert_extension_cleanup_preserves_replacement(
                    symlink, final_write_succeeds=True
                )

    def _assert_extension_cleanup_preserves_replacement(
        self, replace_with_symlink: bool, *, final_write_succeeds: bool
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-proxy-test-") as root:
            parent = Path(root)
            plugin = parent / "plugin"
            plugin.mkdir()
            guard = _PrivateDirectory.capture(plugin)
            original = parent / "original"
            unrelated = parent / "unrelated"
            unrelated.mkdir()
            marker = unrelated / "keep"
            marker.write_text("unrelated material", encoding="utf-8")
            original_write = Path.write_text
            failure = OSError("synthetic extension write failure")

            def replace_during_write(path: Path, content: str, *, encoding: str) -> int:
                target = (
                    "verification-token" if final_write_succeeds else "background.js"
                )
                if path.name != target:
                    return original_write(path, content, encoding=encoding)
                written = (
                    original_write(path, content, encoding=encoding)
                    if final_write_succeeds
                    else 0
                )
                plugin.rename(original)
                if replace_with_symlink:
                    plugin.symlink_to(unrelated, target_is_directory=True)
                else:
                    unrelated.rename(plugin)
                if not final_write_succeeds:
                    raise failure
                return written

            try:
                with (
                    patch.object(tempfile, "mkdtemp", return_value=str(plugin)),
                    patch.object(
                        _PrivateDirectory, "capture", return_value=guard
                    ) as capture,
                    patch.object(Path, "write_text", replace_during_write),
                    self.assertRaises(ProcessOwnershipError) as caught,
                ):
                    proxy._create_proxy_extension("host", 8080, "user", "password")
                capture.assert_called_once_with(plugin)
                self.assertIsInstance(caught.exception.__cause__, ProcessOwnershipError)
                self.assertEqual((plugin / "keep").read_text(), "unrelated material")
                self.assertTrue((original / "manifest.json").is_file())
            finally:
                # Restore the exact captured inode so the retained cleanup
                # obligation can finish without deleting the replacement.
                if original.exists():
                    if replace_with_symlink:
                        plugin.unlink()
                    else:
                        plugin.rename(unrelated)
                    original.rename(plugin)
                    guard.remove(deadline=Deadline.after(5).expires_at)

    def test_extension_write_failure_reaps_owned_cleanup_and_preserves_error(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-proxy-test-") as root:
            plugin = Path(root) / "plugin"
            plugin.mkdir()
            failure = OSError("synthetic write failure")
            with (
                patch.object(tempfile, "mkdtemp", return_value=str(plugin)),
                patch.object(Path, "write_text", side_effect=failure),
                self.assertRaises(OSError) as caught,
            ):
                proxy._create_proxy_extension("host", 8080, "user", "password")
            self.assertIs(caught.exception, failure)
            self.assertFalse(plugin.exists())

    def test_invalid_proxy_port_does_not_echo_configuration(self) -> None:
        secret_port = "SECRET-INVALID-PORT"
        logger = Mock()
        with (
            patch.dict(
                "os.environ",
                {
                    "RP_USERNAME": "user",
                    "RP_PASSWORD": "password",
                    "RP_DNS": f"proxy.example:{secret_port}",
                },
                clear=False,
            ),
            patch.object(proxy, "logger", logger),
            patch.object(proxy, "_create_proxy_extension") as create_extension,
            self.assertRaisesRegex(
                ValueError, "Residential proxy port is invalid"
            ) as raised,
        ):
            proxy.configure_proxy()

        self.assertNotIn(secret_port, str(raised.exception))
        self.assertNotIn(secret_port, repr(logger.method_calls))
        create_extension.assert_not_called()


class ProxyVerificationLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_residential_extension_missing_or_failed_never_navigates(
        self,
    ) -> None:
        guard = proxy._create_proxy_extension("proxy.example", 8443, "u", "p")
        directory = guard.path
        self.addCleanup(lambda: guard.remove(deadline=Deadline.after(5).expires_at))
        for state in ("missing", "rejected", "script-error"):
            with self.subTest(state=state):
                target = SimpleNamespace(
                    type_="service_worker",
                    url="chrome-extension://owned/background.js",
                    send=AsyncMock(
                        return_value=(
                            SimpleNamespace(value=False),
                            object() if state == "script-error" else None,
                        )
                    ),
                )
                browser = SimpleNamespace(
                    targets=[] if state == "missing" else [target]
                )
                with (
                    patch.object(
                        proxy, "navigate_and_wait", new=AsyncMock()
                    ) as navigate,
                    self.assertRaises(proxy.ProxyVerificationError),
                ):
                    await proxy.verify_residential_proxy(
                        browser,
                        str(directory),
                        deadline=Deadline.after(0.02),
                    )
                navigate.assert_not_awaited()

    async def test_residential_extension_accepts_only_boolean_receipt(self) -> None:
        guard = proxy._create_proxy_extension("proxy.example", 8443, "u", "p")
        directory = guard.path
        self.addCleanup(lambda: guard.remove(deadline=Deadline.after(5).expires_at))
        target = SimpleNamespace(
            type_="service_worker",
            url="chrome-extension://owned/background.js",
            send=AsyncMock(return_value=(SimpleNamespace(value=True), None)),
        )
        await proxy.verify_residential_proxy(
            SimpleNamespace(targets=[target]),
            str(directory),
            deadline=Deadline.after(1),
        )
        target.send.assert_awaited_once()

    async def test_direct_ip_hang_is_cancelled_and_fails_closed(self) -> None:
        cancelled = False
        closed = False

        class Client:
            async def __aenter__(self) -> Client:
                return self

            async def __aexit__(self, *_: object) -> None:
                nonlocal closed
                closed = True

            async def get(self, _: str) -> object:
                nonlocal cancelled
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled = True
                    raise
                raise AssertionError("hanging request returned without cancellation")

        with (
            patch(
                "hbrowser.gallery.browser.proxy.httpx.AsyncClient",
                return_value=Client(),
            ),
            self.assertRaisesRegex(
                proxy.ProxyVerificationError,
                "exceeded the shared proxy deadline",
            ),
        ):
            await proxy._read_direct_public_ip(deadline=Deadline.after(0.02))

        self.assertTrue(cancelled)
        self.assertTrue(closed)

    async def test_direct_ip_receipt_completed_after_deadline_is_rejected(
        self,
    ) -> None:
        now = [100.0]
        response = Mock(text="192.0.2.10")

        class Client:
            async def __aenter__(self) -> Client:
                return self

            async def __aexit__(self, *_: object) -> None:
                pass

            async def get(self, _: str) -> object:
                now[0] = 101.0
                return response

        with (
            patch(
                "hbrowser.gallery.utils.deadline.time.monotonic",
                side_effect=lambda: now[0],
            ),
            patch(
                "hbrowser.gallery.browser.proxy.httpx.AsyncClient",
                return_value=Client(),
            ),
            self.assertRaisesRegex(
                proxy.ProxyVerificationError,
                "direct-IP response",
            ),
        ):
            await proxy._read_direct_public_ip(deadline=Deadline(101.0))

        response.raise_for_status.assert_not_called()

    async def test_untrusted_proxy_page_fails_closed_without_logging_content(
        self,
    ) -> None:
        sentinel = "SENSITIVE-PROXY-FAILURE\nSECOND-LINE"
        logger = Mock()
        page = Mock()
        page.get = AsyncMock()
        page.select = AsyncMock(side_effect=ValueError(sentinel))

        with (
            patch.object(proxy, "logger", logger),
            patch(
                "hbrowser.gallery.browser.proxy._read_direct_public_ip",
                new=AsyncMock(return_value="192.0.2.10"),
            ),
            self.assertRaises(proxy.ProxyVerificationError),
        ):
            await proxy.verify_proxy_ip(Mock(), page)

        logger.warning.assert_not_called()
        self.assertNotIn(sentinel, repr(logger.method_calls))

    async def test_navigation_failure_is_terminal_and_skips_proxy_read(self) -> None:
        navigation_error = ValueError("navigation result unavailable")
        page = Mock()
        page.get = AsyncMock(side_effect=navigation_error)
        page.select = AsyncMock()

        with (
            patch(
                "hbrowser.gallery.browser.proxy._read_direct_public_ip",
                new=AsyncMock(return_value="192.0.2.10"),
            ),
            self.assertRaises(BrowserMutationOutcomeUnknownError) as raised,
        ):
            await proxy.verify_proxy_ip(Mock(), page)

        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)
        page.get.assert_awaited_once_with("https://api.ipify.org")
        page.select.assert_not_awaited()

    async def test_equal_ip_failure_does_not_disclose_the_address(self) -> None:
        address = "192.0.2.123"
        page = Mock()
        page.get = AsyncMock()
        page.select = AsyncMock(return_value=Mock(text=address))

        with (
            patch(
                "hbrowser.gallery.browser.proxy._read_direct_public_ip",
                new=AsyncMock(return_value=address),
            ),
            self.assertRaisesRegex(RuntimeError, "local public address") as raised,
        ):
            await proxy.verify_proxy_ip(Mock(), page)

        self.assertNotIn(address, str(raised.exception))
