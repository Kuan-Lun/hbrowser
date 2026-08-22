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
        ]

        for page in pages:
            with self.subTest(page=page):
                state = await detect_forums_auth_state(page)
                self.assertIs(state, ForumsAuthState.UNKNOWN)
