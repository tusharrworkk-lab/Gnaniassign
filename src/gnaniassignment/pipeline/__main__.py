"""Minimal CLI demonstration of the enquiry LLM pipeline."""

from __future__ import annotations

from gnaniassignment.app.container import create_session


def _show(label: str, result: dict) -> None:
    print(
        f"{label}: intent={result.get('intent')!r} topic={result.get('topic')!r} "
        f"spoken={result.get('spoken')!r}"
    )


def main() -> None:
    session = create_session()

    _show("final 1", session.feed_final("Where do you operate?"))
    _show("pause", session.feed_end())


if __name__ == "__main__":
    main()
