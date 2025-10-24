"""Observability utilities providing optional Langfuse tracing and OpenTelemetry spans.

This module centralizes lightweight observability features:
- Langfuse integration via a minimal Trace wrapper that becomes a safe no-op when
  Langfuse is not installed or not configured by environment variables.
- OpenTelemetry span context manager that gracefully degrades to a no-op when
  OpenTelemetry is not available. A basic console exporter is configured by default
  so users can plug in a different exporter externally if desired.

Environment/config dependencies are read from app.config.settings.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from app.config import settings

# Optional Langfuse
try:
    from langfuse import Langfuse
    from langfuse.client import StatefulTraceClient
except Exception as e:  # pragma: no cover
    print(f"Exception occured: {e}")
    Langfuse = None
    StatefulTraceClient = None  # type: ignore

# Optional OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
except Exception:  # pragma: no cover
    trace = None  # type: ignore
    TracerProvider = None  # type: ignore
    BatchSpanProcessor = None  # type: ignore
    ConsoleSpanExporter = None  # type: ignore


_langfuse_client: Optional[Langfuse] = None
_otel_inited: bool = False


def _init_langfuse() -> Optional[Langfuse]:
    """Initialize and memoize a Langfuse client if configuration is present.

    Returns:
        Optional[Langfuse]: A Langfuse client instance when LANGFUSE_HOST,
            LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY are configured and the
            langfuse package is available; otherwise None.
    """
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client
    if (
        settings.LANGFUSE_HOST
        and settings.LANGFUSE_PUBLIC_KEY
        and settings.LANGFUSE_SECRET_KEY
        and Langfuse is not None
    ):
        _langfuse_client = Langfuse(
            host=settings.LANGFUSE_HOST,
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
        )
        return _langfuse_client
    return None


def _init_otel() -> None:
    """Initialize a basic OpenTelemetry tracer provider with console export.

    Sets a global tracer provider once. If OpenTelemetry packages are not
    available, this function safely no-ops.
    """
    global _otel_inited
    if _otel_inited:
        return
    if trace is None or TracerProvider is None:
        return
    # Basic tracer to console (users can configure OTLP exporter externally)
    tp = TracerProvider()
    span_processor = BatchSpanProcessor(ConsoleSpanExporter())
    tp.add_span_processor(span_processor)
    trace.set_tracer_provider(tp)
    _otel_inited = True


@contextmanager
def span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Lightweight context manager for OpenTelemetry span.
    Falls back to no-op if OTel not available.
    """
    _init_otel()
    start = time.time()
    otel_span = None
    if trace is not None:
        try:
            tracer = trace.get_tracer(__name__)
            otel_span = tracer.start_span(name=name)
            if attributes:
                for k, v in attributes.items():
                    try:
                        otel_span.set_attribute(k, v)
                    except Exception:
                        pass
        except Exception:
            otel_span = None
    try:
        yield
    finally:
        if otel_span is not None:
            try:
                otel_span.end()
            except Exception:
                pass


class Trace:
    """
    Minimal wrapper for Langfuse trace with safe no-op methods if not configured.
    """

    def __init__(self, name: str, input: Optional[Dict[str, Any]] = None):
        """Create a trace that wraps optional Langfuse state.

        Args:
            name: Logical name of the trace.
            input: Initial input payload to attach to the trace.

        Notes:
            If Langfuse is not configured or import fails, the instance is a no-op
            and methods will silently return without side effects.
        """
        self.name = name
        self.enabled = False
        self._trace: Optional[StatefulTraceClient] = None
        client = _init_langfuse()
        if client is not None:
            try:
                print(f"Client: {client}")
                self._trace = client.trace(name=name, input=input or {})
                print(f"Trace: {self._trace}")
                self.enabled = True
            except Exception:
                self._trace = None
                self.enabled = False

    def event(self, name: str, data: Optional[Dict[str, Any]] = None):
        """Record a structured event on the trace if Langfuse is enabled.

        Args:
            name: Event name.
            data: Optional dictionary payload to store with the event.
        """
        if not self.enabled or self._trace is None:
            return
        try:
            self._trace.event(name=name, input=data or {})
        except Exception:
            pass

    def generation(self, name: str, prompt: str, output: str, metadata: Optional[Dict[str, Any]] = None):
        """Record a generation with input/output text and optional metadata.

        Args:
            name: Logical generation name.
            prompt: The input prompt text.
            output: The generated text output.
            metadata: Optional metadata to attach.

        Notes:
            Uses settings.OPENAI_MODEL for the model field when present.
        """
        if not self.enabled or self._trace is None:
            return
        try:
            self._trace.generation(
                name=name,
                input=prompt,
                output=output,
                metadata=metadata or {},
                model=settings.OPENAI_MODEL,
            )
        except Exception:
            pass

    def end(self, output: Optional[Dict[str, Any]] = None):
        """Finalize the trace, optionally updating a final output payload.

        Args:
            output: Optional final structured payload to persist before ending.
        """
        if not self.enabled or self._trace is None:
            return
        try:
            self._trace.update(output=output or {})
            self._trace.end()
        except Exception:
            pass
