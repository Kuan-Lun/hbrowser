"""Exercise the pinned Zendriver transport contract without a browser or network."""

import asyncio
import json
import unittest
from typing import cast

from websockets.asyncio.client import ClientConnection
from websockets.protocol import State
from zendriver import cdp
from zendriver.core.connection import (
    Connection,
    Listener,
    ProtocolException,
    Transaction,
)

from hbrowser.gallery.browser import factory
from hbrowser.gallery.utils import Deadline


class _MemoryTransport:
    def __init__(self) -> None:
        self.messages: asyncio.Queue[str] = asyncio.Queue()
        self.state = State.OPEN
        self.close_calls = 0

    async def recv(self) -> str:
        return await self.messages.get()

    async def close(self) -> None:
        self.close_calls += 1
        self.state = State.CLOSED


class ZendriverTransportContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_events_do_not_accumulate_and_late_cancelled_reply_is_safe(
        self,
    ) -> None:
        connection = Connection("ws://unused.invalid")
        transport = _MemoryTransport()
        connection.websocket = cast(ClientConnection, transport)
        pending = Transaction(cdp.runtime.evaluate("1", return_by_value=True))
        cancelled = Transaction(cdp.runtime.evaluate("2", return_by_value=True))
        cancelled.cancel()
        connection.mapper.update({1: pending, 2: cancelled})
        observed: list[float] = []
        all_events = asyncio.Event()

        async def on_event(event: cdp.page.LoadEventFired) -> None:
            observed.append(float(event.timestamp))
            if len(observed) == 1000:
                all_events.set()

        connection.handlers[cdp.page.LoadEventFired].append(on_event)
        connection.listener = Listener(connection)
        transport.messages.put_nowait(
            json.dumps({"id": 2, "result": {"result": {"type": "number", "value": 2}}})
        )
        for index in range(1000):
            transport.messages.put_nowait(
                json.dumps(
                    {"method": "Page.loadEventFired", "params": {"timestamp": index}}
                )
            )
        try:
            await asyncio.wait_for(all_events.wait(), timeout=5)
            self.assertEqual(sorted(observed), list(range(1000)))
            self.assertEqual(connection.mapper, {1: pending})
            self.assertTrue(connection.listener.running)
            self.assertFalse(pending.done())
        finally:
            closed, errors = await factory._close_and_wait_for_connection(
                connection, deadline=Deadline.after(2)
            )
            self.assertTrue(closed)
            self.assertEqual(errors, [])
            with self.assertRaises(ProtocolException):
                await pending

    async def test_root_close_detaches_flat_sessions_and_settles_pending_commands(
        self,
    ) -> None:
        root = Connection("ws://unused.invalid")
        child = Connection("ws://unused.invalid/target")
        transport = _MemoryTransport()
        root.websocket = child.websocket = cast(ClientConnection, transport)
        child.session_id = "session-1"
        child._session_parent = root
        child._inbox = asyncio.Queue()
        root._sessions[child.session_id] = child
        root_pending = Transaction(cdp.runtime.evaluate("1"))
        child_pending = Transaction(cdp.runtime.evaluate("2"))
        root.mapper[1] = root_pending
        child.mapper[2] = child_pending
        delivered = asyncio.Event()

        async def on_event(_: cdp.page.LoadEventFired) -> None:
            delivered.set()

        child.handlers[cdp.page.LoadEventFired].append(on_event)
        root.listener = Listener(root)
        child.listener = Listener(child)
        transport.messages.put_nowait(
            json.dumps(
                {
                    "sessionId": "session-1",
                    "method": "Page.loadEventFired",
                    "params": {"timestamp": 1},
                }
            )
        )
        try:
            await asyncio.wait_for(delivered.wait(), timeout=2)
        finally:
            closed, errors = await factory._close_and_wait_for_browser_connections(
                (root, child), deadline=Deadline.after(2)
            )
            self.assertTrue(closed)
            self.assertEqual(errors, [])
            for pending in (root_pending, child_pending):
                with self.assertRaises(ProtocolException):
                    await pending
        self.assertEqual(transport.close_calls, 1)
        self.assertEqual(root._sessions, {})
        self.assertIsNone(child.session_id)
        self.assertIsNone(child._session_parent)
        self.assertIsNone(child._inbox)
        for connection in (root, child):
            self.assertIsNone(connection.websocket)
            self.assertEqual(connection.mapper, {})
            assert connection.listener is not None
            assert connection.listener.task is not None
            self.assertTrue(connection.listener.task.done())
