# Retail intent dataset

Each line is one JSON object with these required fields:

```json
{"text":"mera order kahan hai","intent":"track_order","language":"hinglish","source":"human","group_id":"call-0042-turn-3"}
```

`intent` must match [retail-v1.json](../../configs/intent/retail-v1.json). `group_id` identifies a source utterance and all of its paraphrases. It prevents a paraphrase from appearing in both train and evaluation splits.

Use real, consented, de-identified customer utterances where possible. Include Hindi script, Roman Hindi/Hinglish, English, ASR-like spelling errors, short answers, and noise/out-of-scope utterances in `other`.

The example JSONL is a schema demonstration only. Do not train or evaluate a production model on it. Target at least 300 distinct groups per retail label and 600+ for `other`; reserve 15–20% of groups for evaluation.
