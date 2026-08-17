import asyncio
import unittest
from types import SimpleNamespace

from zendriver import cdp
from zendriver.core.connection import EventTransaction, Transaction

from hbrowser.gallery.browser.mapper import (
    prune_zendriver_browser_mappers,
    prune_zendriver_connection_mapper,
    run_zendriver_mapper_janitor,
    start_zendriver_mapper_janitor,
    stop_zendriver_mapper_janitor,
)


def _event_transaction() -> EventTransaction:
    return EventTransaction(object())


def _command_transaction() -> Transaction:
    return Transaction(cdp.runtime.evaluate("1", return_by_value=True))


class MapperPruningTests(unittest.IsolatedAsyncioTestCase):
    async def test_prunes_only_completed_event_transactions(self) -> None:
        event = _event_transaction()
        pending_command = _command_transaction()
        cancelled_command = _command_transaction()
        cancelled_command.cancel()
        completed_command = _command_transaction()
        completed_command(result={"result": {"type": "number", "value": 1}})
        connection = SimpleNamespace(
            mapper={
                1: event,
                2: pending_command,
                3: cancelled_command,
                4: completed_command,
            }
        )

        removed = prune_zendriver_connection_mapper(connection)

        self.assertEqual(removed, 1)
        self.assertEqual(
            connection.mapper,
            {
                2: pending_command,
                3: cancelled_command,
                4: completed_command,
            },
        )

    async def test_prunes_browser_and_target_connections_once_each(self) -> None:
        browser_connection = SimpleNamespace(mapper={1: _event_transaction()})
        page_connection = SimpleNamespace(mapper={2: _event_transaction()})
        browser = SimpleNamespace(
            connection=browser_connection,
            targets=[browser_connection, page_connection],
        )

        removed = prune_zendriver_browser_mappers(browser)

        self.assertEqual(removed, 2)
        self.assertEqual(browser_connection.mapper, {})
        self.assertEqual(page_connection.mapper, {})

    async def test_periodic_runner_discovers_targets_added_after_start(self) -> None:
        wakeups: asyncio.Queue[None] = asyncio.Queue()
        observed_intervals: list[float] = []
        browser = SimpleNamespace(
            connection=SimpleNamespace(mapper={}),
            targets=[],
        )

        async def controlled_sleep(interval: float) -> None:
            observed_intervals.append(interval)
            await wakeups.get()

        janitor = asyncio.create_task(
            run_zendriver_mapper_janitor(
                browser,
                interval=7,
                sleep=controlled_sleep,
            )
        )
        await asyncio.sleep(0)
        page = SimpleNamespace(mapper={3: _event_transaction()})
        browser.targets.append(page)

        async def wait_until_pruned() -> None:
            while page.mapper:
                await asyncio.sleep(0)

        try:
            wakeups.put_nowait(None)
            await asyncio.wait_for(wait_until_pruned(), timeout=1)

            self.assertEqual(page.mapper, {})
            self.assertTrue(observed_intervals)
            self.assertTrue(all(interval == 7 for interval in observed_intervals))
        finally:
            janitor.cancel()
            await asyncio.gather(janitor, return_exceptions=True)

    async def test_start_prunes_immediately_and_stop_is_idempotent(self) -> None:
        connection = SimpleNamespace(mapper={1: _event_transaction()})
        browser = SimpleNamespace(connection=connection, targets=[])

        first = start_zendriver_mapper_janitor(browser, interval=60)
        second = start_zendriver_mapper_janitor(browser, interval=60)

        self.assertIs(first, second)
        self.assertEqual(connection.mapper, {})
        self.assertFalse(first.done())

        await stop_zendriver_mapper_janitor(browser)
        await stop_zendriver_mapper_janitor(browser)

        self.assertTrue(first.cancelled())

    async def test_rejects_invalid_interval_before_starting_task(self) -> None:
        browser = SimpleNamespace(connection=None, targets=[])

        for interval in (0, -1, float("inf"), float("nan"), True):
            with (
                self.subTest(interval=interval),
                self.assertRaises(ValueError),
            ):
                start_zendriver_mapper_janitor(browser, interval=interval)


if __name__ == "__main__":
    unittest.main()
