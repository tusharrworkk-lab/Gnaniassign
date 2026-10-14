"""Minimal CLI demonstration of the enquiry LLM pipeline."""

from __future__ import annotations

from gnaniassignment.app.container import create_session


def main() -> None:
    session = create_session()
    session.feed_final("Where do you operate?")
    print("answer: ", end="", flush=True)
    result = session.feed_end(on_chunk=lambda chunk: print(chunk, end="", flush=True))
    print()
    print(f"intent={result['intent']!r} topic={result['topic']!r}")


if __name__ == "__main__":
    main()
