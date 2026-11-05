from __future__ import annotations

import hmac
import json
import logging
import secrets
import socket
import struct
import unittest
from threading import Event, Thread, current_thread
from typing import Any
from unittest.mock import Mock, patch

from hbrowser.gallery.utils import _log_forwarding as forwarding


def _frame(value: dict[str, object]) -> bytes:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return struct.pack(">I", len(payload)) + payload


def _receive(connection: socket.socket) -> dict[str, Any]:
    def exact(size: int) -> bytes:
        result = bytearray()
        while len(result) < size:
            chunk = connection.recv(size - len(result))
            if not chunk:
                raise AssertionError("peer closed before its expected frame")
            result.extend(chunk)
        return bytes(result)

    size = struct.unpack(">I", exact(4))[0]
    if not 0 < size <= 32 * 1024:
        raise AssertionError("peer sent an unbounded frame")
    value = json.loads(exact(size))
    if not isinstance(value, dict):
        raise AssertionError("peer sent a non-object frame")
    return value


def _proof(token: str, role: bytes, client_nonce: str, server_nonce: str) -> str:
    return hmac.digest(
        bytes.fromhex(token),
        b"hbrowser-log-forwarding-v2\0"
        + role
        + b"\0"
        + bytes.fromhex(client_nonce)
        + bytes.fromhex(server_nonce),
        "sha256",
    ).hex()


class ForwardingAuthenticationTests(unittest.TestCase):
    def test_replaced_listener_receives_neither_capability_nor_records(self) -> None:
        for reply_type in ("old-ready", "false-proof"):
            with self.subTest(reply_type=reply_type):
                self._assert_replaced_listener_rejected(reply_type)

    def _assert_replaced_listener_rejected(self, reply_type: str) -> None:
        receiver = forwarding.LogForwardingReceiver(delivery_callback=lambda _: True)
        capability = receiver.child_environment()
        token = capability[forwarding.FORWARD_TOKEN_ENVIRONMENT_VARIABLE]
        endpoint = receiver.endpoint
        host, port = endpoint.split(":")
        receiver.close()
        with socket.socket() as listener:
            listener.settimeout(2)
            listener.bind((host, int(port)))
            listener.listen(1)
            hello_received, allow_reply = Event(), Event()
            received: list[bytes] = []
            peer_errors: list[BaseException] = []

            def impersonate() -> None:
                try:
                    connection, _ = listener.accept()
                    with connection:
                        connection.settimeout(2)
                        hello = _receive(connection)
                        received.append(_frame(hello))
                        hello_received.set()
                        if not allow_reply.wait(2):
                            raise AssertionError("test did not release the reply")
                        reply: dict[str, object] = {"type": "ready"}
                        if reply_type == "false-proof":
                            reply = {
                                "type": "challenge",
                                "nonce": secrets.token_hex(32),
                                "proof": secrets.token_hex(32),
                            }
                        connection.sendall(_frame(reply))
                        while chunk := connection.recv(4096):
                            received.append(chunk)
                except BaseException as error:
                    peer_errors.append(error)

            peer = Thread(target=impersonate)
            peer.start()
            failure = Mock()
            handler = forwarding.ForwardingHandler(
                endpoint, token, _failure_recorder=failure
            )
            try:
                self.assertTrue(hello_received.wait(2))
                handler.emit(
                    logging.makeLogRecord(
                        {
                            "name": "hbrowser.test",
                            "msg": "private queued message",
                            "levelname": "INFO",
                            "levelno": logging.INFO,
                        }
                    )
                )
                self.assertEqual(handler._queue.qsize(), 1)
                allow_reply.set()
                self.assertTrue(handler._stopped.wait(2))
            finally:
                allow_reply.set()
                handler.close_forwarding()
                peer.join(3)
            self.assertFalse(peer.is_alive())
            self.assertFalse(handler._worker.is_alive())
            self.assertEqual(peer_errors, [])
            self.assertEqual(len(received), 1)
            self.assertNotIn(token.encode(), b"".join(received))
            self.assertNotIn(b"private queued message", b"".join(received))
            self.assertEqual(failure.call_args.args[0], "forward-configure")

    def test_receiver_proof_binds_both_nonces_and_peer_role(self) -> None:
        token, client_nonce, server_nonce = (secrets.token_hex(32) for _ in range(3))
        for variant in ("client-replay", "server-replay", "reflected-role"):
            with self.subTest(variant=variant):
                proof = _proof(
                    token,
                    b"sender" if variant == "reflected-role" else b"receiver",
                    secrets.token_hex(32)
                    if variant == "client-replay"
                    else client_nonce,
                    secrets.token_hex(32)
                    if variant == "server-replay"
                    else server_nonce,
                )
                response = _frame(
                    {"type": "challenge", "nonce": server_nonce, "proof": proof}
                )
                connection = Mock()
                connection.recv.side_effect = [response[:4], response[4:]]
                with (
                    patch(
                        "hbrowser.gallery.utils._log_forwarding.secrets.token_hex",
                        return_value=client_nonce,
                    ),
                    self.assertRaises(forwarding.ForwardingProtocolError),
                ):
                    forwarding.ForwardingHandler._authenticate(connection, token)
                connection.sendall.assert_called_once()
                self.assertNotIn(token.encode(), connection.sendall.call_args.args[0])

    def test_sender_handshake_has_one_deadline_despite_partial_reads(self) -> None:
        token, client_nonce, server_nonce = (secrets.token_hex(32) for _ in range(3))
        response = bytearray(
            _frame(
                {
                    "type": "challenge",
                    "nonce": server_nonce,
                    "proof": _proof(token, b"receiver", client_nonce, server_nonce),
                }
            )
        )
        clock = [0.0]

        def partial_read(_size: int) -> bytes:
            clock[0] += 0.005
            return bytes([response.pop(0)])

        connection = Mock()
        connection.recv.side_effect = partial_read
        with (
            patch(
                "hbrowser.gallery.utils._log_forwarding.time.monotonic",
                side_effect=lambda: clock[0],
            ),
            patch.object(forwarding, "_AUTHENTICATION_TIMEOUT_SECONDS", 0.02),
            patch(
                "hbrowser.gallery.utils._log_forwarding.secrets.token_hex",
                return_value=client_nonce,
            ),
            self.assertRaises(forwarding.ForwardingReceiveError),
        ):
            forwarding.ForwardingHandler._authenticate(connection, token)
        connection.sendall.assert_called_once()
        self.assertTrue(response)
        self.assertLessEqual(clock[0], 0.025)

    def test_replayed_sender_proof_cannot_authenticate_a_new_connection(self) -> None:
        failure, deliver = Mock(), Mock(return_value=True)
        receiver = forwarding.LogForwardingReceiver(
            delivery_callback=deliver, _failure_recorder=failure
        )
        token = receiver.child_environment()[
            forwarding.FORWARD_TOKEN_ENVIRONMENT_VARIABLE
        ]
        host, port = receiver.endpoint.split(":")
        nonce = secrets.token_hex(32)
        hello = _frame({"type": "hello", "version": 2, "nonce": nonce})
        try:
            with socket.create_connection((host, int(port)), timeout=2) as first:
                first.sendall(hello)
                challenge = _receive(first)
                authenticate = _frame(
                    {
                        "type": "authenticate",
                        "proof": _proof(token, b"sender", nonce, challenge["nonce"]),
                    }
                )
                first.sendall(authenticate)
                self.assertEqual(_receive(first), {"type": "ready"})
                first.sendall(_frame({"type": "close"}))
                self.assertEqual(first.recv(1), b"")
            with socket.create_connection((host, int(port)), timeout=2) as second:
                second.sendall(hello)
                second_challenge = _receive(second)
                self.assertNotEqual(challenge["nonce"], second_challenge["nonce"])
                second.sendall(authenticate)
                self.assertEqual(second.recv(1), b"")
        finally:
            receiver.close()
        failure.assert_not_called()
        deliver.assert_not_called()
        self.assertNotIn(token.encode(), hello + authenticate)

    def test_valid_proof_with_invalid_schema_is_authenticated_failure(self) -> None:
        failure = Mock()
        receiver = forwarding.LogForwardingReceiver(
            delivery_callback=lambda _: True, _failure_recorder=failure
        )
        token = receiver.child_environment()[
            forwarding.FORWARD_TOKEN_ENVIRONMENT_VARIABLE
        ]
        host, port = receiver.endpoint.split(":")
        nonce = secrets.token_hex(32)
        try:
            with socket.create_connection((host, int(port)), timeout=2) as peer:
                peer.sendall(_frame({"type": "hello", "version": 2, "nonce": nonce}))
                challenge = _receive(peer)
                peer.sendall(
                    _frame(
                        {
                            "type": "authenticate",
                            "extra": "invalid",
                            "proof": _proof(
                                token, b"sender", nonce, challenge["nonce"]
                            ),
                        }
                    )
                )
                self.assertEqual(peer.recv(1), b"")
        finally:
            receiver.close()
        self.assertEqual(failure.call_args.args[0], "forward-protocol")


class ReceiverAdmissionTests(unittest.TestCase):
    def test_close_seals_paused_accept_and_requires_accept_thread_exit(self) -> None:
        for release_before_deadline in (False, True):
            with self.subTest(release_before_deadline=release_before_deadline):
                self._assert_paused_accept_close(release_before_deadline)

    def _assert_paused_accept_close(self, release_before_deadline: bool) -> None:
        accepted, release, served = Event(), Event(), Event()
        original_settimeout = socket.socket.settimeout
        outcomes: list[BaseException] = []
        failure = Mock()

        def pause_accepted(connection: socket.socket, timeout: float | None) -> None:
            original_settimeout(connection, timeout)
            if current_thread().name == "hbrowser-log-forwarding-accept":
                accepted.set()
                if not release.wait(3):
                    raise AssertionError("test did not release the accept thread")

        def unexpected_client(_receiver: object, _client: object) -> None:
            served.set()

        with (
            patch.object(socket.socket, "settimeout", pause_accepted),
            patch.object(
                forwarding.LogForwardingReceiver, "_serve_client", unexpected_client
            ),
        ):
            receiver = forwarding.LogForwardingReceiver(
                delivery_callback=lambda _: True, _failure_recorder=failure
            )
            host, port = receiver.endpoint.split(":")

            def close() -> None:
                try:
                    receiver.close(drain_timeout=1 if release_before_deadline else 0.05)
                except BaseException as error:
                    outcomes.append(error)

            closer = Thread(target=close)
            with socket.create_connection((host, int(port)), timeout=2) as peer:
                try:
                    self.assertTrue(accepted.wait(2))
                    closer.start()
                    self.assertTrue(receiver._stopping.wait(2))
                    if release_before_deadline:
                        release.set()
                    closer.join(2)
                    self.assertFalse(closer.is_alive())
                    if release_before_deadline:
                        self.assertEqual(outcomes, [])
                        failure.assert_not_called()
                    else:
                        self.assertEqual(len(outcomes), 1)
                        self.assertIsInstance(
                            outcomes[0], forwarding.ForwardingDrainError
                        )
                        self.assertTrue(receiver._accept_thread.is_alive())
                        self.assertEqual(failure.call_args.args[0], "forward-drain")
                finally:
                    release.set()
                    receiver._accept_thread.join(3)
                    if closer.ident is not None:
                        closer.join(3)
                self.assertEqual(peer.recv(1), b"")
            self.assertFalse(receiver._accept_thread.is_alive())
            self.assertFalse(served.is_set())
            self.assertFalse(receiver._clients)
