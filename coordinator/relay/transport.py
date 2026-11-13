"""HTTP transport for the relay, using only the standard library.

POST /send/<recipient>     body = tensor bytes, envelope fields in headers
GET  /receive/<recipient>  long-polls; 200 with a message, 204 if none yet

Envelope fields travel as headers so the server never has to open the body.
Requests are not logged.

TLS: give `RelayServer` a `TLSFiles` (certificate and key) to serve HTTPS, and
`RelayClient` the CA certificate that signed it to verify the server. The TLS
handshake runs in each request's own thread, so a slow or bad client cannot
stall the relay. `tls_from_env()` reads the paths each machine sets.
"""

import os
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from coordinator.relay.relay import Envelope, Kind, Relay, RelayError, RelayTimeout

_HEADERS = {
    "kind": "X-MedWeave-Kind",
    "institution_id": "X-MedWeave-Institution",
    "batch_id": "X-MedWeave-Batch",
    "backbone_version": "X-MedWeave-Backbone-Version",
}
MAX_POLL_SECONDS = 20.0

# Environment variables naming each machine's TLS files (paths differ per laptop).
ENV_CERT, ENV_KEY, ENV_CA = "MEDWEAVE_TLS_CERT", "MEDWEAVE_TLS_KEY", "MEDWEAVE_TLS_CA"


@dataclass(frozen=True)
class TLSFiles:
    """The coordinator's server certificate (PEM, signed by the CA) and its private key."""

    certificate: Path
    private_key: Path

    def server_context(self) -> ssl.SSLContext:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(self.certificate, self.private_key)
        return context


def tls_from_env() -> TLSFiles | None:
    """The relay's certificate and key from MEDWEAVE_TLS_CERT / MEDWEAVE_TLS_KEY, or None if unset."""
    certificate, key = os.environ.get(ENV_CERT), os.environ.get(ENV_KEY)
    if not certificate and not key:
        return None
    if not (certificate and key):
        raise RelayError(f"set both {ENV_CERT} and {ENV_KEY} to serve the relay over HTTPS")
    return TLSFiles(Path(certificate), Path(key))


def ca_from_env() -> Path | None:
    """The CA certificate institutions verify the relay with (MEDWEAVE_TLS_CA), or None if unset."""
    ca = os.environ.get(ENV_CA)
    return Path(ca) if ca else None


class _TLSHTTPServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], context: ssl.SSLContext):
        self.tls_context = context
        super().__init__(address, handler)

    def finish_request(self, request: socket.socket, client_address: tuple[str, int]) -> None:
        # Runs in the request's thread: the handshake never blocks the accept loop.
        with self.tls_context.wrap_socket(request, server_side=True) as tls_request:
            super().finish_request(tls_request, client_address)

    def handle_error(self, request: object, client_address: tuple[str, int]) -> None:
        pass  # failed handshakes (wrong CA, plain HTTP) are the client's error; nothing about them is logged


class RelayServer:
    """Serves a `Relay` over HTTP, or HTTPS when given `tls`, on a background thread."""

    def __init__(
        self, relay: Relay, host: str = "127.0.0.1", port: int = 8765, tls: TLSFiles | None = None
    ) -> None:
        self.relay = relay
        self.scheme = "https" if tls else "http"
        handler = _handler_for(relay)
        self._server = (
            _TLSHTTPServer((host, port), handler, tls.server_context()) if tls else ThreadingHTTPServer((host, port), handler)
        )
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"{self.scheme}://{host}:{port}"

    def start(self) -> "RelayServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> "RelayServer":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


def _handler_for(relay: Relay) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            recipient = self._recipient("/send/")
            if recipient is None:
                return
            try:
                fields = {name: self.headers[header] for name, header in _HEADERS.items()}
                if any(value is None for value in fields.values()):
                    raise RelayError("missing envelope headers")
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                relay.send(recipient, Envelope(Kind(fields.pop("kind")), **fields, body=body))
            except (RelayError, ValueError) as error:
                self._reply(403, str(error).encode())
                return
            self._reply(204)

        def do_GET(self) -> None:
            parsed = urllib.parse.urlsplit(self.path)
            self.path = parsed.path
            recipient = self._recipient("/receive/")
            if recipient is None:
                return
            query = urllib.parse.parse_qs(parsed.query)
            timeout = min(float(query.get("timeout", [MAX_POLL_SECONDS])[0]), MAX_POLL_SECONDS)
            try:
                envelope = relay.receive(recipient, timeout)
            except RelayTimeout:
                self._reply(204)
                return
            except RelayError as error:
                self._reply(403, str(error).encode())
                return
            headers = {header: str(getattr(envelope, name)) for name, header in _HEADERS.items()}
            self._reply(200, envelope.body, headers)

        def _recipient(self, prefix: str) -> str | None:
            if not self.path.startswith(prefix) or len(self.path) == len(prefix):
                self._reply(404)
                return None
            return urllib.parse.unquote(self.path.removeprefix(prefix))

        def _reply(self, status: int, body: bytes = b"", headers: dict[str, str] | None = None) -> None:
            self.send_response(status)
            for header, value in (headers or {}).items():
                self.send_header(header, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass  # payloads and their metadata are never written to logs (SPEC §5.4)

    return Handler


class RelayClient:
    """What institutions use to reach the coordinator's relay. Same methods as `Relay`.

    For an `https://` relay, `ca_file` is the CA certificate that signed the
    relay's certificate; without it the system's trusted CAs are used.
    """

    def __init__(self, base_url: str, ca_file: Path | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._context = ssl.create_default_context(cafile=ca_file) if ca_file else None

    def send(self, recipient: str, envelope: Envelope) -> None:
        headers = {header: str(getattr(envelope, name)) for name, header in _HEADERS.items()}
        request = urllib.request.Request(
            f"{self.base_url}/send/{urllib.parse.quote(recipient)}",
            data=envelope.body,
            headers=headers,
            method="POST",
        )
        status, body, _ = self._open(request, timeout=30)
        if status != 204:
            raise RelayError(f"relay refused message ({status}): {body.decode(errors='replace')}")

    def receive(self, recipient: str, timeout: float) -> Envelope:
        deadline = time.monotonic() + timeout
        while (remaining := deadline - time.monotonic()) > 0:
            poll = min(remaining, MAX_POLL_SECONDS)
            url = f"{self.base_url}/receive/{urllib.parse.quote(recipient)}?timeout={poll:.3f}"
            status, body, headers = self._open(urllib.request.Request(url), timeout=poll + 10)
            if status == 200:
                fields = {name: headers.get(header, "") for name, header in _HEADERS.items()}
                return Envelope(Kind(fields.pop("kind")), **fields, body=body)
            if status != 204:
                raise RelayError(f"relay refused receive ({status}): {body.decode(errors='replace')}")
        raise RelayTimeout(f"nothing arrived for {recipient!r} within {timeout}s")

    def _open(self, request: urllib.request.Request, timeout: float) -> tuple[int, bytes, Message]:
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=self._context) as response:
                return response.status, response.read(), response.headers
        except urllib.error.HTTPError as error:
            return error.code, error.read(), error.headers
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            raise RelayError(f"cannot reach relay at {request.full_url}: {error}") from error
