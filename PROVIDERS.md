# Provider architecture

The orchestration layer is vendor-neutral. Provider selection belongs in
configuration and provider construction belongs in factories, never inside an
agent prompt or business workflow.

| Capability | Runtime setting | Factory / contract |
| --- | --- | --- |
| LLM | `LLM_PROVIDER` | `create_chat_model` / `ChatModel` |
| Intent routing | `INTENT_PROVIDER` | `create_intent_router` / `IntentRouter` |
| Speech-to-text | `STT_PROVIDER` | `create_stt_provider` / `RealtimeSttProvider` |
| Text-to-speech | `TTS_PROVIDER` | `create_tts_provider` / `RealtimeTtsProvider` |
| Episodic memory | injected store | `EpisodicMemoryStore` |
| Knowledge retrieval | injected retriever | `VectorStore` |

## Adding a speech provider

1. Implement `RealtimeSttProvider` or `RealtimeTtsProvider` from
   `src/gnaniassignment/voice/contracts.py`.
2. Make STT adapters emit `VoiceProcessing`, `VoiceTranscript`, and
   `VoiceError` from `voice/events.py`.
3. Add one named factory to `voice/providers.py`.
4. Set `STT_PROVIDER` or `TTS_PROVIDER` to that name.
5. Add an adapter contract test. The microphone pipeline itself must not need
   editing.

The same rule applies to LLM, intent, memory, and retrieval adapters: add the
implementation behind its existing protocol, then register it at the relevant
composition factory. Credentials remain in environment variables or a secret
manager, never prompts or source code.
