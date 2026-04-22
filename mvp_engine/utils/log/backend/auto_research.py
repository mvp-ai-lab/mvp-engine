from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any



@dataclass(slots=True)
class AutoResearchBackend:
    endpoint: str
    run_id: str
    run_output_dir: str
    cwd: str | None = None
    session_id: str | None = None
    transport: str = "codex_mvp_exec"
    metadata: dict[str, Any] = field(default_factory=dict)
    lmms_eval: dict[str, Any] = field(default_factory=dict)
    strict: bool = False
    auto_bind: bool = True
    _bound: bool = False

    @classmethod
    def from_engine_config(cls, config) -> "AutoResearchBackend":
        auto_cfg = config.auto_research
        endpoint = os.environ.get("AUTO_RESEARCH_ENDPOINT") or str(auto_cfg.endpoint)
        session_id = os.environ.get("CODEX_SESSION_ID") or auto_cfg.session_id
        cwd = auto_cfg.cwd or os.getcwd()
        metadata = dict(auto_cfg.metadata)
        lmms_eval = dict(auto_cfg.lmms_eval)
        return cls(
            endpoint=endpoint,
            run_id=config.runtime.run_id,
            run_output_dir=config.runtime.output_dir,
            cwd=cwd,
            session_id=session_id,
            transport=auto_cfg.transport,
            metadata=metadata,
            lmms_eval=lmms_eval,
            strict=bool(auto_cfg.strict),
            auto_bind=bool(auto_cfg.auto_bind),
        )

    def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.endpoint}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5.0) as response:
            return json.loads(response.read().decode("utf-8"))

    def _handle_error(self, action: str, exc: Exception) -> None:
        message = f"auto-research {action} failed for run={self.run_id}: {type(exc).__name__}: {exc}"
        if self.strict:
            raise RuntimeError(message) from exc
        from mvp_engine.utils.log import simple_info

        simple_info(message, level="warning")

    def _binding_metadata(self) -> dict[str, Any]:
        payload = dict(self.metadata)
        payload["run_output_dir"] = self.run_output_dir
        if self.lmms_eval:
            payload["lmms_eval"] = self.lmms_eval
        return payload

    def bind(self) -> bool:
        try:
            response = self._request(
                "/v1/runs/bind",
                {
                    "run_id": self.run_id,
                    "session_id": self.session_id,
                    "cwd": self.cwd,
                    "transport": self.transport,
                    "metadata": self._binding_metadata(),
                },
            )
        except Exception as exc:
            self._bound = False
            self._handle_error("bind", exc)
            return False

        binding = response.get("binding") or {}
        session_id = binding.get("session_id")
        if session_id:
            self.session_id = str(session_id)
        self._bound = bool(response.get("ok"))
        return self._bound

    def _normalize_metrics(self, metrics: dict[str, Any] | None) -> dict[str, float | int | str]:
        normalized: dict[str, float | int | str] = {}
        for key, value in (metrics or {}).items():
            if value is None:
                continue
            if isinstance(value, bool):
                normalized[str(key)] = int(value)
            elif isinstance(value, (int, float, str)):
                normalized[str(key)] = value
            else:
                normalized[str(key)] = str(value)
        return normalized

    def _normalize_artifacts(self, artifacts: dict[str, Any] | None) -> dict[str, str]:
        normalized = {str(key): str(value) for key, value in (artifacts or {}).items() if value is not None}
        normalized.setdefault("run_output_dir", self.run_output_dir)
        return normalized

    def _normalize_metadata(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        normalized = dict(metadata or {})
        normalized.setdefault("run_output_dir", self.run_output_dir)
        return normalized

    def emit_event(
        self,
        *,
        event_type: str,
        summary: str | None = None,
        step: int | None = None,
        epoch: int | None = None,
        metrics: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if self.auto_bind and not self._bound and not self.bind():
            return None

        payload = {
            "run_id": self.run_id,
            "event_type": event_type,
            "summary": summary,
            "step": step,
            "epoch": epoch,
            "metrics": self._normalize_metrics(metrics),
            "artifacts": self._normalize_artifacts(artifacts),
            "metadata": self._normalize_metadata(metadata),
        }
        try:
            return self._request("/v1/events", payload)
        except urllib.error.URLError as exc:
            if self.auto_bind:
                self._bound = False
                if self.bind():
                    try:
                        return self._request("/v1/events", payload)
                    except Exception as retry_exc:  # pragma: no cover - defensive runtime path
                        self._handle_error(f"emit {event_type}", retry_exc)
                        return None
            self._handle_error(f"emit {event_type}", exc)
            return None
        except Exception as exc:
            self._handle_error(f"emit {event_type}", exc)
            return None

    def flush(self, timeout_seconds: float = 60.0) -> dict[str, Any]:
        if self.auto_bind and not self._bound and not self.bind():
            return {"ok": True, "idle": False, "skipped": True}
        try:
            return self._request("/v1/admin/flush", {"timeout_seconds": float(timeout_seconds)})
        except Exception as exc:
            self._handle_error("flush", exc)
            return {"ok": False, "idle": False, "error": str(exc)}
