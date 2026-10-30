import unittest

from hbrowser.gallery.forums_auth import (
    ForumsAuthState,
    detect_forums_auth_state,
)


class _Page:
    def __init__(
        self,
        *,
        url: object = "https://forums.e-hentai.org/",
        guest: bool = False,
        member: bool = False,
    ) -> None:
        self.url = url
        self.guest = guest
        self.member = member

    async def evaluate(self, expression: str) -> object:
        if "document.querySelector('#userlinksguest')" not in expression:
            raise AssertionError(f"Unexpected expression: {expression}")
        return {
            "url": self.url,
            "guest": self.guest,
            "member": self.member,
        }


class ForumsAuthStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_authenticated_requires_member_marker_on_forums_origin(self) -> None:
        state = await detect_forums_auth_state(_Page(member=True))

        self.assertIs(state, ForumsAuthState.AUTHENTICATED)

    async def test_guest_requires_guest_marker_on_forums_origin(self) -> None:
        state = await detect_forums_auth_state(_Page(guest=True))

        self.assertIs(state, ForumsAuthState.GUEST)

    async def test_unknown_states_fail_closed(self) -> None:
        pages = [
            _Page(),
            _Page(guest=True, member=True),
            _Page(url="https://example.com/", member=True),
            _Page(url=None, member=True),
            _Page(url="https://forums.e-hentai.org:8443/", member=True),
            _Page(url="https://forums.e-hentai.org:invalid/", member=True),
            _Page(url="https://user@forums.e-hentai.org/", member=True),
        ]

        for page in pages:
            with self.subTest(page=page):
                state = await detect_forums_auth_state(page)
                self.assertIs(state, ForumsAuthState.UNKNOWN)


class ForumsCredentialTests(unittest.IsolatedAsyncioTestCase):
    async def test_mutation_checks_executing_document_before_writing_secrets(
        self,
    ) -> None:
        # Node executes the actual synchronous browser expression against a tiny
        # DOM, including a redirect after selector readiness and before evaluate.
        import json
        import shutil
        import subprocess
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from hbrowser.exceptions import LoginFailedException
        from hbrowser.gallery.forums_auth import fill_forums_login_form
        from hbrowser.gallery.utils import Deadline

        node = shutil.which("node")
        self.assertIsNotNone(node, "Repository-local Markdown tooling requires Node")
        assert node is not None
        for origin, top_frame, fields, accepted in (
            ("https://forums.e-hentai.org", True, True, True),
            ("https://example.test", True, True, False),
            ("http://forums.e-hentai.org", True, True, False),
            ("https://forums.e-hentai.org:8443", True, True, False),
            ("https://forums.e-hentai.org.evil.test", True, True, False),
            ("https://forums.e-hentai.org", False, True, False),
            ("https://forums.e-hentai.org", True, False, False),
        ):
            with self.subTest(origin=origin, top_frame=top_frame, fields=fields):
                writes: list[object] = []

                async def evaluate(
                    expression: str,
                    origin: str = origin,
                    top_frame: bool = top_frame,
                    fields: bool = fields,
                    writes: list[object] = writes,
                ) -> object:
                    program = """
const fs = require('fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const writes = [];
class HTMLInputElement {
  set value(value) { writes.push(value); }
  dispatchEvent(event) {}
}
const window = {location: {origin: input.origin}};
window.top = input.top ? window : {};
const document = {querySelector: () => input.fields ? new HTMLInputElement() : null};
const result = eval(input.expression);
process.stdout.write(JSON.stringify({result, writes}));
"""
                    result = subprocess.run(
                        [node, "-e", program],
                        input=json.dumps(
                            {
                                "origin": origin,
                                "top": top_frame,
                                "fields": fields,
                                "expression": expression,
                            }
                        ),
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=5,
                    )
                    report = json.loads(result.stdout)
                    writes.extend(report["writes"])
                    return report["result"]

                page = SimpleNamespace(
                    select=AsyncMock(return_value=object()), evaluate=evaluate
                )
                if accepted:
                    await fill_forums_login_form(
                        page,
                        'synthetic-"user',
                        "synthetic-pass",
                        deadline=Deadline.after(5),
                    )
                    self.assertEqual(writes, ['synthetic-"user', "synthetic-pass"])
                else:
                    with self.assertRaises(LoginFailedException):
                        await fill_forums_login_form(
                            page,
                            "synthetic-user",
                            "synthetic-pass",
                            deadline=Deadline.after(5),
                        )
                    self.assertEqual(writes, [])
