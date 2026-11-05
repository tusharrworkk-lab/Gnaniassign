"""Provider-neutral handoff from final STT events to a conversation session."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Protocol

from gnaniassignment.voice.events import VoiceEvent, VoiceTranscript


class _TurnSink(Protocol):
    def feed_final(self, phrase: str) -> dict[str, Any]: ...

    def feed_end(self, on_chunk: Callable[[str], None] | None = None) -> dict[str, Any]: ...


class TurnBridge:
    """Deduplicate final transcripts before submitting one complete turn."""

    def __init__(self, session: _TurnSink, auto_end_turn: bool = True) -> None:
        self.session = session
        self.auto_end_turn = auto_end_turn
        self._seen_segment_ids: set[str] = set()

    def handle(self, event: VoiceEvent, on_chunk: Callable[[str], None] | None = None) -> dict[str, Any] | None:
        if not isinstance(event, VoiceTranscript) or not event.text.strip():
            return None
        if event.segment_id and event.segment_id in self._seen_segment_ids:
            return None
        if event.segment_id:
            self._seen_segment_ids.add(event.segment_id)
        pending = self.session.feed_final(event.text)
        return self.end_turn(on_chunk=on_chunk) if self.auto_end_turn else pending

    async def handle_async(self, event: VoiceEvent, on_chunk: Callable[[str], None] | None = None) -> dict[str, Any] | None:
        return await asyncio.to_thread(self.handle, event, on_chunk)

    def end_turn(self, on_chunk: Callable[[str], None] | None = None) -> dict[str, Any]:
        return self.session.feed_end(on_chunk=on_chunk)
