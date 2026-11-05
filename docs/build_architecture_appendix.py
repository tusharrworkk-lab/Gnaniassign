"""Append a repeatable, line-numbered code appendix to the architecture HTML.

Run from repository root:
    python docs/build_architecture_appendix.py
"""

from __future__ import annotations

from datetime import UTC, datetime
from html import escape
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "docs" / "voice-agent-architecture.html"
START = "<!-- CODE_APPENDIX_START -->"
END = "<!-- CODE_APPENDIX_END -->"

GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "A. Composition and API boundary",
        "These files create the HTTP boundary, load settings, and compose every dependency used by a session.",
        (
            "src/gnaniassignment/api/server.py",
            "src/gnaniassignment/app/config.py",
            "src/gnaniassignment/app/container.py",
        ),
    ),
    (
        "B. Conversation orchestration and durable state",
        "These files create the internal conversation UUID, manage working/episodic memory, persist analytics, and store order context.",
        (
            "src/gnaniassignment/orchestration/session.py",
            "src/gnaniassignment/orchestration/graph.py",
            "src/gnaniassignment/orchestration/episodic_memory.py",
            "src/gnaniassignment/orchestration/working_memory.py",
            "src/gnaniassignment/conversation_analytics.py",
            "src/gnaniassignment/intent_feedback.py",
            "src/gnaniassignment/order_data.py",
        ),
    ),
    (
        "C. Prompt, knowledge, and generic-agent layer",
        "These files resolve the versioned prompt tree, retrieve approved information, build the generic agent, and call the configured hosted LLM.",
        (
            "src/gnaniassignment/prompts/tree.py",
            "src/gnaniassignment/llm/retrieval.py",
            "src/gnaniassignment/llm/agents/generic.py",
            "src/gnaniassignment/llm/generic_router.py",
            "src/gnaniassignment/llm/chat.py",
        ),
    ),
    (
        "D. Voice provider abstraction and Gnani adapters",
        "These files keep STT/TTS replaceable. The microphone runner demonstrates real-time orchestration and latency-cue behavior.",
        (
            "src/gnaniassignment/voice/contracts.py",
            "src/gnaniassignment/voice/events.py",
            "src/gnaniassignment/voice/providers.py",
            "src/gnaniassignment/voice/turn_bridge.py",
            "src/gnaniassignment/stt/gnani.py",
            "src/gnaniassignment/tts/gnani.py",
            "src/gnaniassignment/pipeline/gnani_microphone.py",
        ),
    ),
    (
        "E. MiniLM review and human-feedback workflow",
        "These files define candidate prediction, nightly classification, review, and export. They intentionally keep final labels human-controlled.",
        (
            "src/gnaniassignment/llm/retail_minilm.py",
            "src/gnaniassignment/pipeline/nightly_intent_review.py",
            "src/gnaniassignment/pipeline/review_conversations.py",
            "src/gnaniassignment/pipeline/export_reviewed_conversations.py",
            "src/gnaniassignment/pipeline/review_shadow_intents.py",
            "src/gnaniassignment/pipeline/export_reviewed_intents.py",
        ),
    ),
    (
        "F. Database schema and seedable prompt configuration",
        "This is the authoritative SQL shape used by the API and review jobs. Seed files are included because they document the initial product configuration.",
        (
            "infra/postgres/episodic_memory.sql",
            "infra/postgres/conversation_analytics.sql",
            "infra/postgres/intent_feedback.sql",
            "infra/postgres/customer_orders.sql",
            "infra/postgres/knowledge_base.sql",
            "infra/postgres/prompt_trees.sql",
            "infra/postgres/default_generic_prompt_tree.sql",
            "infra/postgres/personal_phone_prompt_tree.sql",
        ),
    ),
    (
        "G. Container and deployment configuration",
        "These files define the API image, local cloud rehearsal topology, and the CI/CD ECS deployment path. Environment examples contain variable names only, not secrets.",
        (
            "Dockerfile",
            "docker-compose.local-cloud.yml",
            ".env.example",
            ".github/workflows/deploy-ecs.yml",
            "infra/aws/ecs-task-definition.json.tpl",
        ),
    ),
)


def code_block(path: Path, relative: str) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    numbered = "\n".join(f"{number:04d}  {line}" for number, line in enumerate(lines, start=1))
    return f"<h4><code>{escape(relative)}</code> <span class=\"small\">({len(lines)} lines)</span></h4>\n<pre>{escape(numbered)}</pre>"


def appendix() -> str:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        START,
        '<section class="page">',
        '<h2>Appendix B: Code-level construction guide and source snapshot</h2>',
        '<p>This appendix explains the build order, then embeds line-numbered snapshots of the active architecture files. It is intentionally implementation-specific: treat it as a reference for this repository revision, not a substitute for Git history or tests.</p>',
        f'<p class="small">Generated from the working tree at {timestamp}. Files containing secrets, downloaded model weights, local environment values, and unrelated experiments are excluded.</p>',
        '<h3>Build order</h3>',
        '<ol><li>Define dependency factories and configuration.</li><li>Create the HTTP/session boundary.</li><li>Create the session graph, working memory, and persistence contracts.</li><li>Add prompt and knowledge retrieval before model invocation.</li><li>Attach swappable STT/TTS providers and a real-time turn bridge.</li><li>Persist conversations independently from live response latency.</li><li>Add offline MiniLM classification and human-approved feedback export.</li><li>Declare PostgreSQL schema, container runtime, and deployment configuration.</li><li>Finally, add a vendor-specific VoiceLink adapter once its exact WSS contract is available.</li></ol>',
        '<div class="warn"><strong>Important:</strong> The source snapshot is comprehensive for the active architecture path, but the VoiceLink media adapter is not present because it has not been implemented. Do not infer VoiceLink payload fields from this appendix.</div>',
    ]
    for title, description, files in GROUPS:
        parts.append(f"<h3>{escape(title)}</h3><p>{escape(description)}</p>")
        for relative in files:
            path = ROOT / relative
            if not path.exists():
                parts.append(f"<p class=\"warn\">Missing at generation time: <code>{escape(relative)}</code></p>")
            else:
                parts.append(code_block(path, relative))
    parts.extend(["</section>", END])
    return "\n".join(parts)


def main() -> None:
    source = DOCUMENT.read_text(encoding="utf-8")
    source = re.sub(re.escape(START) + r".*?" + re.escape(END), "", source, flags=re.S)
    insertion = appendix()
    if "</body>" not in source:
        raise RuntimeError(f"Expected closing body tag in {DOCUMENT}")
    DOCUMENT.write_text(source.replace("</body>", insertion + "\n</body>"), encoding="utf-8")
    print(f"Updated {DOCUMENT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
