"""One authenticated loopback transfer inside the existing diagnostic owner."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import math
import secrets
import socket
import time

from ..utils.diagnostic import (
    _PAGE_DIAGNOSTIC_MAX_FILE_BYTES,
    _PAGE_DIAGNOSTIC_NONCE_PATTERN,
    _PAGE_DIAGNOSTIC_TOTAL_BYTES,
)

_CHALLENGE_BYTES = 32
_PROOF_BYTES = 32
_CAPABILITY_BYTES = 32


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if not math.isfinite(deadline) or remaining <= 0:
        raise TimeoutError("Page diagnostic payload deadline expired")
    return remaining


def _validate(
    capability: bytes, request_nonce: str, payload_size: int, deadline: float
) -> None:
    if (
        len(capability) != _CAPABILITY_BYTES
        or _PAGE_DIAGNOSTIC_NONCE_PATTERN.fullmatch(request_nonce) is None
        or not 0
        <= payload_size
        <= min(_PAGE_DIAGNOSTIC_MAX_FILE_BYTES, _PAGE_DIAGNOSTIC_TOTAL_BYTES)
    ):
        raise ValueError("Invalid page diagnostic payload authority")
    _remaining(deadline)


def _proof(
    role: bytes,
    capability: bytes,
    request_nonce: str,
    payload_size: int,
    client_nonce: bytes,
    server_nonce: bytes,
) -> bytes:
    message = (
        b"hbrowser-page-diagnostic-v1\0"
        + role
        + b"\0"
        + request_nonce.encode("ascii")
        + payload_size.to_bytes(8, "big")
        + client_nonce
        + server_nonce
    )
    return hmac.digest(capability, message, hashlib.sha256)


async def _receive_async(
    connection: socket.socket, size: int, *, deadline: float
) -> bytes:
    output = bytearray()
    loop = asyncio.get_running_loop()
    while len(output) < size:
        chunk = await asyncio.wait_for(
            loop.sock_recv(connection, size - len(output)), _remaining(deadline)
        )
        _remaining(deadline)
        if not chunk:
            raise ValueError("Page diagnostic payload handshake was truncated")
        output.extend(chunk)
    return bytes(output)


async def send_payload(
    port: int,
    capability: bytes,
    request_nonce: str,
    payload: bytes,
    *,
    deadline: float,
) -> None:
    """Authenticate the worker before exposing any page bytes to its socket."""
    _validate(capability, request_nonce, len(payload), deadline)
    if isinstance(port, bool) or not 1 <= port <= 65535:
        raise ValueError("Invalid page diagnostic payload port")
    loop = asyncio.get_running_loop()
    client_nonce = secrets.token_bytes(_CHALLENGE_BYTES)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.setblocking(False)
        await asyncio.wait_for(
            loop.sock_connect(connection, ("127.0.0.1", port)), _remaining(deadline)
        )
        await asyncio.wait_for(
            loop.sock_sendall(connection, client_nonce), _remaining(deadline)
        )
        response = await _receive_async(
            connection, _CHALLENGE_BYTES + _PROOF_BYTES, deadline=deadline
        )
        server_nonce = response[:_CHALLENGE_BYTES]
        expected = _proof(
            b"server",
            capability,
            request_nonce,
            len(payload),
            client_nonce,
            server_nonce,
        )
        if not hmac.compare_digest(response[_CHALLENGE_BYTES:], expected):
            raise ValueError("Page diagnostic worker authentication failed")
        client_proof = _proof(
            b"client",
            capability,
            request_nonce,
            len(payload),
            client_nonce,
            server_nonce,
        )
        await asyncio.wait_for(
            loop.sock_sendall(connection, client_proof + payload), _remaining(deadline)
        )
        _remaining(deadline)
        connection.shutdown(socket.SHUT_WR)


def _receive(connection: socket.socket, size: int, *, deadline: float) -> bytes:
    output = bytearray()
    while len(output) < size:
        connection.settimeout(_remaining(deadline))
        chunk = connection.recv(size - len(output))
        _remaining(deadline)
        if not chunk:
            raise ValueError("Page diagnostic payload was truncated")
        output.extend(chunk)
    return bytes(output)


def receive_payload(
    listener: socket.socket,
    capability: bytes,
    request_nonce: str,
    payload_size: int,
    *,
    deadline: float,
) -> bytes:
    """Accept one authenticated transfer; reject extra or incomplete bytes."""
    _validate(capability, request_nonce, payload_size, deadline)
    listener.settimeout(_remaining(deadline))
    connection, _address = listener.accept()
    listener.close()
    with connection:
        client_nonce = _receive(connection, _CHALLENGE_BYTES, deadline=deadline)
        server_nonce = secrets.token_bytes(_CHALLENGE_BYTES)
        server_proof = _proof(
            b"server",
            capability,
            request_nonce,
            payload_size,
            client_nonce,
            server_nonce,
        )
        connection.settimeout(_remaining(deadline))
        connection.sendall(server_nonce + server_proof)
        supplied_proof = _receive(connection, _PROOF_BYTES, deadline=deadline)
        expected = _proof(
            b"client",
            capability,
            request_nonce,
            payload_size,
            client_nonce,
            server_nonce,
        )
        if not hmac.compare_digest(supplied_proof, expected):
            raise ValueError("Page diagnostic parent authentication failed")
        payload = _receive(connection, payload_size, deadline=deadline)
        connection.settimeout(_remaining(deadline))
        extra = connection.recv(1)
        _remaining(deadline)
        if extra:
            raise ValueError("Page diagnostic payload exceeded its declared size")
    return payload
