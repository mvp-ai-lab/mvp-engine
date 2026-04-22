from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mvp_engine.utils.log.backend.auto_research import AutoResearchBackend


def test_auto_research_backend_binds_emits_and_flushes():
    requests: list[tuple[str, dict]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length).decode("utf-8")) if content_length else {}
            requests.append((self.path, payload))

            if self.path == "/v1/runs/bind":
                body = {
                    "ok": True,
                    "binding": {
                        "run_id": payload["run_id"],
                        "session_id": payload.get("session_id") or "generated-session",
                        "cwd": payload.get("cwd"),
                        "transport": payload.get("transport"),
                        "metadata": payload.get("metadata", {}),
                    },
                }
            elif self.path == "/v1/events":
                body = {"ok": True, "queued": True, "event": payload}
            elif self.path == "/v1/admin/flush":
                body = {"ok": True, "idle": True, "status": {}}
            else:  # pragma: no cover
                body = {"ok": False}

            encoded = json.dumps(body).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format, *args):  # noqa: A003
            return None

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    try:
        port = httpd.server_address[1]
        backend = AutoResearchBackend(
            endpoint=f"http://127.0.0.1:{port}",
            run_id="run-1",
            run_output_dir="/tmp/run-1",
            cwd="/tmp/workspace",
            metadata={"owner": "auto"},
            lmms_eval={"enabled": True, "command": ["python3", "-m", "lmms_eval"]},
            strict=True,
        )

        assert backend.bind() is True
        emit_result = backend.emit_event(
            event_type="progress_updated",
            step=12,
            metrics={"time/batch": 2.5},
            artifacts={"checkpoint_dir": "/tmp/run-1/checkpoints/iter_12"},
        )
        flush_result = backend.flush(timeout_seconds=1.5)

        assert emit_result is not None
        assert flush_result["ok"] is True
        assert requests[0][0] == "/v1/runs/bind"
        assert requests[0][1]["metadata"]["run_output_dir"] == "/tmp/run-1"
        assert requests[0][1]["metadata"]["lmms_eval"]["enabled"] is True
        assert requests[1][0] == "/v1/events"
        assert requests[1][1]["artifacts"]["run_output_dir"] == "/tmp/run-1"
        assert requests[1][1]["metrics"]["time/batch"] == 2.5
        assert requests[2][0] == "/v1/admin/flush"
        assert requests[2][1]["timeout_seconds"] == 1.5
    finally:
        httpd.shutdown()
        thread.join(timeout=2.0)
        httpd.server_close()
