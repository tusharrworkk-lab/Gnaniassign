# System One decision data

System One is the fast policy that decides the next safe action from a voice turn. It begins as a calibrated supervised classifier; it becomes a contextual-bandit/RL policy only after we have reviewed decision-outcome data.

Keep three datasets separate:

| File | Purpose | May train model weights? |
|---|---|---|
| `supervised.jsonl` | Human-approved utterance to action/intent labels | Yes |
| `calibration.jsonl` | Locked, group-disjoint labels used for temperature/threshold fitting | Never |
| `outcomes.jsonl` | Logged production decisions and reviewed rewards for offline policy learning | Only after reward audit |

Use the JSON schemas in this directory. Do not put names, phone numbers, addresses, order IDs, payment details, audio, or unredacted transcripts into the repository. Replace them with typed placeholders, for example `<ORDER_ID>`.

## Required collection rules

- A `group_id` identifies one original customer meaning and all its paraphrases/ASR variants. A group must occur in exactly one of train, calibration, or test.
- Collect Hindi-script, Roman Hindi/Hinglish, English, code-mixed, short answers, ASR errors, and genuine out-of-scope turns.
- Label the *next action*, not just the topic. If there is insufficient evidence, use the configured abstention label/action (`other` → `clarify`) rather than guessing a retail intent.
- Two reviewers should label disagreements. Store the adjudicated label and reviewer agreement outside the runtime prompt/log path.

## Reward design for later RL

Do not reward a model merely because it answered quickly. Log a reviewable reward vector, then approve a scalar reward offline. A safe initial vector is:

```json
{"correct_action": 1, "resolved_without_reprompt": 1, "unnecessary_clarification": -0.25, "unsafe_or_hallucinated_action": -5, "human_override": -1}
```

For every logged decision, `policy_probability` is mandatory: it is the probability with which the serving policy selected that action. Without it, offline counterfactual evaluation and conservative contextual-bandit learning are invalid. Start by shadow-scoring candidate policies; do not let an RL policy choose customer-facing actions until it clears the locked calibration and safety evaluation sets.
