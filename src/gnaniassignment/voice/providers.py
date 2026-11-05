"""The one registry for realtime speech adapters.

Add a provider by implementing the contracts and adding one named factory here.
The microphone pipeline never imports a vendor transport directly.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from gnaniassignment.voice.contracts import RealtimeSttProvider, RealtimeTtsProvider


def _name(value: str | None, variable: str) -> str:
    return (value or os.getenv(variable, "gnani")).strip().lower()


def _gnani_stt() -> RealtimeSttProvider:
    from gnaniassignment.stt.gnani import GnaniRealtimeStt

    return GnaniRealtimeStt()


def _gnani_tts() -> RealtimeTtsProvider:
    from gnaniassignment.tts.gnani import GnaniRealtimeTts

    return GnaniRealtimeTts()


# Add one entry here when a new adapter is implemented. Removing an adapter is
# equally explicit: delete its entry without touching orchestration code.
STT_PROVIDER_FACTORIES: dict[str, Callable[[], RealtimeSttProvider]] = {"gnani": _gnani_stt}
TTS_PROVIDER_FACTORIES: dict[str, Callable[[], RealtimeTtsProvider]] = {"gnani": _gnani_tts}


def create_stt_provider(provider: str | None = None) -> RealtimeSttProvider:
    name = _name(provider, "STT_PROVIDER")
    factory = STT_PROVIDER_FACTORIES.get(name)
    if factory is None:
        raise ValueError(f"Unsupported STT_PROVIDER {name!r}. Available: {', '.join(sorted(STT_PROVIDER_FACTORIES))}")
    return factory()


def create_tts_provider(provider: str | None = None) -> RealtimeTtsProvider:
    name = _name(provider, "TTS_PROVIDER")
    factory = TTS_PROVIDER_FACTORIES.get(name)
    if factory is None:
        raise ValueError(f"Unsupported TTS_PROVIDER {name!r}. Available: {', '.join(sorted(TTS_PROVIDER_FACTORIES))}")
    return factory()
