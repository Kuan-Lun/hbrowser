from __future__ import annotations

import asyncio
import secrets
import socket
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from hbrowser.gallery.browser import _page_diagnostic_transport as transport
from hbrowser.gallery.utils.diagnostic import _PAGE_DIAGNOSTIC_MAX_FILE_BYTES

_TEST_TIMEOUT = 3.0


@contextmanager
def _listener() -> Iterator[socket.socket]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        yield listener


async def _read_exact(connection: socket.socket, count: int) -> bytes:
    output = bytearray()
    loop = asyncio.get_running_loop()
    async with asyncio.timeout(_TEST_TIMEOUT):
        while len(output) < count:
            chunk = await loop.sock_recv(connection, count - len(output))
            if not chunk:
                raise AssertionError("peer closed before the expected test frame")
            output.extend(chunk)
    return bytes(output)


async def _connect(port: int) -> socket.socket:
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.setblocking(False)
    try:
        await asyncio.wait_for(
            asyncio.get_running_loop().sock_connect(connection, ("127.0.0.1", port)),
            _TEST_TIMEOUT,
        )
    except BaseException:
        connection.close()
        raise
    return connection


@pytest.mark.parametrize("size", [0, 1, _PAGE_DIAGNOSTIC_MAX_FILE_BYTES])
def test_authenticated_transfer_preserves_exact_bounded_payload(size: int) -> None:
    capability, nonce = secrets.token_bytes(32), secrets.token_hex(16)
    payload = b"x" * size
    with _listener() as listener, ThreadPoolExecutor(max_workers=1) as executor:
        port = listener.getsockname()[1]
        deadline = time.monotonic() + _TEST_TIMEOUT
        received = executor.submit(
            transport.receive_payload,
            listener,
            capability,
            nonce,
            size,
            deadline=deadline,
        )
        asyncio.run(
            transport.send_payload(port, capability, nonce, payload, deadline=deadline)
        )
        assert received.result(timeout=_TEST_TIMEOUT) == payload
        assert listener.fileno() == -1


@pytest.mark.parametrize(
    "mismatch",
    ["capability", "request", "size", "role", "client_nonce", "server_nonce"],
)
def test_untrusted_replacement_listener_never_receives_page_or_capability(
    mismatch: str,
) -> None:
    capability, nonce = secrets.token_bytes(32), secrets.token_hex(16)
    payload = b"synthetic private page that must not reach an untrusted listener"

    async def scenario(listener: socket.socket) -> None:
        loop = asyncio.get_running_loop()
        listener.setblocking(False)

        async def replacement_server() -> bytes:
            connection, _ = await loop.sock_accept(listener)
            with connection:
                challenge = await _read_exact(connection, 32)
                server_nonce = secrets.token_bytes(32)
                proof = transport._proof(
                    b"client" if mismatch == "role" else b"server",
                    secrets.token_bytes(32) if mismatch == "capability" else capability,
                    secrets.token_hex(16) if mismatch == "request" else nonce,
                    len(payload) + (1 if mismatch == "size" else 0),
                    secrets.token_bytes(32)
                    if mismatch == "client_nonce"
                    else challenge,
                    secrets.token_bytes(32)
                    if mismatch == "server_nonce"
                    else server_nonce,
                )
                await loop.sock_sendall(connection, server_nonce + proof)
                after_challenge = await asyncio.wait_for(
                    loop.sock_recv(connection, 4096), _TEST_TIMEOUT
                )
                assert after_challenge == b""
                return challenge

        server = asyncio.create_task(replacement_server())
        try:
            with pytest.raises(ValueError, match="authentication"):
                await transport.send_payload(
                    listener.getsockname()[1],
                    capability,
                    nonce,
                    payload,
                    deadline=time.monotonic() + _TEST_TIMEOUT,
                )
            observed = await asyncio.wait_for(server, _TEST_TIMEOUT)
            assert len(observed) == 32
            assert capability not in observed
            assert payload not in observed
        finally:
            server.cancel()
            await asyncio.gather(server, return_exceptions=True)

    with _listener() as listener:
        asyncio.run(scenario(listener))


def test_truncated_server_proof_closes_connection_without_payload() -> None:
    capability, nonce = secrets.token_bytes(32), secrets.token_hex(16)

    async def scenario(listener: socket.socket) -> None:
        loop = asyncio.get_running_loop()
        listener.setblocking(False)

        async def truncated_server() -> None:
            connection, _ = await loop.sock_accept(listener)
            with connection:
                await _read_exact(connection, 32)
                await loop.sock_sendall(connection, secrets.token_bytes(63))
                connection.shutdown(socket.SHUT_WR)
                assert await loop.sock_recv(connection, 1) == b""

        server = asyncio.create_task(truncated_server())
        try:
            with pytest.raises(ValueError, match="truncated"):
                await transport.send_payload(
                    listener.getsockname()[1],
                    capability,
                    nonce,
                    b"private page",
                    deadline=time.monotonic() + _TEST_TIMEOUT,
                )
            await asyncio.wait_for(server, _TEST_TIMEOUT)
        finally:
            server.cancel()
            await asyncio.gather(server, return_exceptions=True)

    with _listener() as listener:
        asyncio.run(scenario(listener))


@pytest.mark.parametrize("fault", ["reflection", "truncated", "oversize"])
def test_receiver_rejects_invalid_client_or_inexact_payload(fault: str) -> None:
    capability, nonce = secrets.token_bytes(32), secrets.token_hex(16)
    payload = b"bounded diagnostic payload"

    async def send_fault(port: int) -> None:
        loop = asyncio.get_running_loop()
        with await _connect(port) as connection:
            challenge = secrets.token_bytes(32)
            await loop.sock_sendall(connection, challenge)
            response = await _read_exact(connection, 64)
            if fault == "reflection":
                await loop.sock_sendall(connection, response[32:])
            else:
                proof = transport._proof(
                    b"client", capability, nonce, len(payload), challenge, response[:32]
                )
                wire_payload = payload[:-1] if fault == "truncated" else payload + b"x"
                await loop.sock_sendall(connection, proof + wire_payload)
            connection.shutdown(socket.SHUT_WR)

    with _listener() as listener, ThreadPoolExecutor(max_workers=1) as executor:
        port = listener.getsockname()[1]
        received = executor.submit(
            transport.receive_payload,
            listener,
            capability,
            nonce,
            len(payload),
            deadline=time.monotonic() + _TEST_TIMEOUT,
        )
        asyncio.run(send_fault(port))
        message = {
            "reflection": "authentication",
            "truncated": "truncated",
            "oversize": "exceeded",
        }[fault]
        with pytest.raises(ValueError, match=message):
            received.result(timeout=_TEST_TIMEOUT)
        assert listener.fileno() == -1


@pytest.mark.parametrize("action", ["deadline", "cancel"])
def test_sender_interrupts_stalled_peer_and_closes_socket(action: str) -> None:
    capability, nonce = secrets.token_bytes(32), secrets.token_hex(16)

    async def scenario(listener: socket.socket) -> None:
        loop = asyncio.get_running_loop()
        listener.setblocking(False)
        challenged = asyncio.Event()

        async def stalled_server() -> None:
            connection, _ = await loop.sock_accept(listener)
            with connection:
                await _read_exact(connection, 32)
                challenged.set()
                assert await loop.sock_recv(connection, 1) == b""

        server = asyncio.create_task(stalled_server())
        sender = asyncio.create_task(
            transport.send_payload(
                listener.getsockname()[1],
                capability,
                nonce,
                b"must not be sent",
                deadline=time.monotonic() + (0.3 if action == "deadline" else 2.0),
            )
        )
        try:
            await asyncio.wait_for(challenged.wait(), _TEST_TIMEOUT)
            if action == "cancel":
                sender.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await sender
            else:
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(sender, _TEST_TIMEOUT)
            await asyncio.wait_for(server, _TEST_TIMEOUT)
        finally:
            sender.cancel()
            server.cancel()
            await asyncio.gather(sender, server, return_exceptions=True)

    with _listener() as listener:
        asyncio.run(scenario(listener))


@pytest.mark.parametrize("phase", ["accept", "challenge", "payload", "eof"])
def test_receiver_uses_one_absolute_deadline_through_transfer(phase: str) -> None:
    capability, nonce = secrets.token_bytes(32), secrets.token_hex(16)
    payload = b"bounded payload"

    async def stall_client(port: int) -> None:
        loop = asyncio.get_running_loop()
        with await _connect(port) as connection:
            if phase != "challenge":
                challenge = secrets.token_bytes(32)
                await loop.sock_sendall(connection, challenge)
                response = await _read_exact(connection, 64)
                proof = transport._proof(
                    b"client", capability, nonce, len(payload), challenge, response[:32]
                )
                await loop.sock_sendall(connection, proof)
                if phase == "eof":
                    await loop.sock_sendall(connection, payload)
            # Keep the write side open: success requires the exact payload plus EOF.
            assert await asyncio.wait_for(loop.sock_recv(connection, 1), 2.0) == b""

    with _listener() as listener, ThreadPoolExecutor(max_workers=1) as executor:
        port = listener.getsockname()[1]
        deadline = time.monotonic() + 0.3
        received = executor.submit(
            transport.receive_payload,
            listener,
            capability,
            nonce,
            len(payload),
            deadline=deadline,
        )
        if phase != "accept":
            asyncio.run(stall_client(port))
        with pytest.raises(TimeoutError):
            received.result(timeout=_TEST_TIMEOUT)
        assert time.monotonic() < deadline + 1.0


def test_over_limit_sender_is_rejected_before_connecting() -> None:
    with _listener() as listener:
        listener.settimeout(0.1)
        with pytest.raises(ValueError, match="authority"):
            asyncio.run(
                transport.send_payload(
                    listener.getsockname()[1],
                    secrets.token_bytes(32),
                    secrets.token_hex(16),
                    b"x" * (_PAGE_DIAGNOSTIC_MAX_FILE_BYTES + 1),
                    deadline=time.monotonic() + _TEST_TIMEOUT,
                )
            )
        with pytest.raises(TimeoutError):
            listener.accept()
