"""Generate a balanced, split-aware synthetic retail-intent dataset.

Rows follow the IndicJevBench state/question/options/expected format so they
can be used by the MiniLM dual-encoder trainer. The data is synthetic and must
not be reported as a real-customer benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


LABELS = [
    "track_order", "return_item", "cancel_order", "refund_status", "wrong_item",
    "size_issue", "payment_issue", "delivery_address", "other",
]
PRODUCTS = ["jacket", "headphones", "phone case", "shoes", "kitchen mixer", "shirt", "bag", "watch"]
TIMES = ["two days", "three days", "a week", "since yesterday", "several hours"]
OTHER_TOPICS = [
    "tomorrow's weather", "a vegetarian recipe", "a movie for tonight", "a football score", "a train timetable",
    "a maths problem", "a travel destination", "a phone wallpaper", "a song recommendation", "a local library",
    "a language translation", "a study plan", "a news headline", "a fitness routine", "a cooking tip",
    "a book recommendation", "a calendar reminder", "a taxi route", "a joke", "a festival date",
    "a museum opening time", "a nearby restaurant", "a coding exercise", "a meditation exercise", "a grocery list",
    "a museum ticket", "a school holiday", "a cricket fixture", "a science fact", "a currency conversion",
    "a photography tip", "a gardening idea", "a podcast", "a dance class", "a music playlist",
    "a bus route", "a hotel suggestion", "a skincare tip", "a birthday message", "a grammar rule",
    "a historical event", "a pet-care tip", "a laptop shortcut", "a painting idea", "a quiz question",
    "a hiking route", "a public holiday", "a breakfast idea", "a yoga pose", "a career tip",
]
INSTRUCTIONS = [
    "What is the customer's intent?",
    "Choose the best retail-support action.",
    "Which issue is the customer describing?",
    "Customer ka intent kya hai?",
    "ग्राहक की समस्या किस प्रकार की है?",
]
WRAPPERS = {
    "en": [("", ""), ("Please help: ", ""), ("I need an update. ", ""), ("", " Please advise.")],
    "hinglish": [("", ""), ("Please help, ", ""), ("Jaldi batao, ", ""), ("", " please check.")],
    "hi": [("", ""), ("कृपया मदद करें, ", ""), ("जल्दी बताइए, ", ""), ("", " कृपया जांचें।")],
}
DETAILS = {
    "en": ["", " The app has not changed.", " This is my first request.", " I need clarity today.", " The status looks confusing."],
    "hinglish": ["", " App mein koi update nahi hai.", " Ye meri pehli request hai.", " Aaj clear answer chahiye.", " Status samajh nahi aa raha."],
    "hi": ["", " ऐप में कोई बदलाव नहीं दिख रहा है।", " यह मेरी पहली रिक्वेस्ट है।", " आज स्पष्ट जानकारी चाहिए।", " स्थिति समझ में नहीं आ रही है।"],
}
CONTEXTS = {
    "en": [
        " I checked the app today.", " I contacted support once already.", " This is for a recent purchase.",
        " I can provide the order details if needed.", " The notification arrived this morning.",
        " I am using the mobile app.", " I saw this after checking my account.", " This needs a clear response.",
        " I am writing from the order screen.", " The issue is still unresolved.",
    ],
    "hinglish": [
        " Aaj app check kiya hai.", " Support ko pehle bhi message kiya tha.", " Ye recent purchase ke liye hai.",
        " Zarurat ho to order details de sakta hoon.", " Aaj subah notification aaya tha.", " Main mobile app use kar raha hoon.",
        " Account check karte waqt ye dikha.", " Clear response chahiye.", " Order screen se message kar raha hoon.", " Issue abhi solve nahi hua hai.",
    ],
    "hi": [
        " मैंने आज ऐप जांचा है।", " मैंने पहले भी सहायता टीम को संदेश दिया था।", " यह हाल की खरीद के लिए है।",
        " जरूरत पड़ने पर ऑर्डर विवरण दे सकता हूं।", " आज सुबह सूचना मिली थी।", " मैं मोबाइल ऐप इस्तेमाल कर रहा हूं।",
        " खाता जांचते समय यह दिखा।", " स्पष्ट जवाब चाहिए।", " मैं ऑर्डर स्क्रीन से लिख रहा हूं।", " समस्या अभी हल नहीं हुई है।",
    ],
}
FOLLOWUPS = {
    "en": ["", " Please help.", " Need an answer."],
    "hinglish": ["", " Please help.", " Jaldi batao."],
    "hi": ["", " कृपया मदद करें।", " जल्दी बताइए।"],
}

# Each class has an English, Roman-Hindi/Hinglish, and Devanagari-Hindi template.
TEMPLATES = {
    "track_order": [
        ("en", "My {product} order has shown in transit for {time}. Where is it now?"),
        ("hinglish", "Mera {product} order {time} se transit mein hai, tracking update chahiye."),
        ("hi", "मेरा {product} ऑर्डर {time} से रास्ते में दिख रहा है, इसकी स्थिति बताइए।"),
    ],
    "return_item": [
        ("en", "The delivered {product} is unused, but I do not want to keep it. I need a return."),
        ("hinglish", "Delivered {product} use nahi kiya hai, mujhe return request raise karni hai."),
        ("hi", "डिलीवर हुआ {product} इस्तेमाल नहीं किया है, इसे वापस करने की प्रक्रिया चाहिए।"),
    ],
    "cancel_order": [
        ("en", "Please cancel my {product} order before it is dispatched."),
        ("hinglish", "Mera {product} order dispatch hone se pehle cancel karna hai."),
        ("hi", "मेरा {product} ऑर्डर भेजने से पहले रद्द कर दीजिए।"),
    ],
    "refund_status": [
        ("en", "My {product} return was accepted {time} ago, but the refund is not in my account."),
        ("hinglish", "{product} return approve hue {time} ho gaye, refund abhi tak credit nahi hua."),
        ("hi", "{product} का रिटर्न {time} पहले स्वीकार हुआ था, लेकिन रिफंड अभी तक नहीं आया है।"),
    ],
    "wrong_item": [
        ("en", "I ordered a {product}, but the parcel contains a completely different item."),
        ("hinglish", "Maine {product} order kiya tha, parcel mein koi aur item nikla."),
        ("hi", "मैंने {product} मंगाया था, लेकिन पैकेट में अलग सामान मिला है।"),
    ],
    "size_issue": [
        ("en", "The {product} is the correct product but the size does not fit. I need another size."),
        ("hinglish", "{product} sahi aaya hai lekin size fit nahi ho raha, size exchange chahiye."),
        ("hi", "{product} सही है, लेकिन इसका आकार फिट नहीं है; दूसरा साइज़ चाहिए।"),
    ],
    "payment_issue": [
        ("en", "Payment for my {product} was deducted, but checkout failed and no order was created."),
        ("hinglish", "{product} ke liye payment cut gaya lekin checkout fail ho gaya aur order nahi bana."),
        ("hi", "{product} के लिए भुगतान कट गया, लेकिन चेकआउट असफल रहा और ऑर्डर नहीं बना।"),
    ],
    "delivery_address": [
        ("en", "My {product} has not shipped yet. I need to change the delivery address."),
        ("hinglish", "{product} abhi dispatch nahi hua hai, delivery address update karna hai."),
        ("hi", "{product} अभी भेजा नहीं गया है, डिलीवरी का पता बदलना है।"),
    ],
    "other": [
        ("en", "Can you help me with {topic}?"),
        ("hinglish", "Mujhe {topic} ke baare mein batao."),
        ("hi", "मुझे {topic} के बारे में जानकारी चाहिए।"),
    ],
}


def split_for(group_id: str) -> str:
    value = int.from_bytes(hashlib.sha256(group_id.encode()).digest()[:8], "big") / 2**64
    return "train" if value < 0.80 else "validation" if value < 0.90 else "test"


def state_for(label: str, rng: random.Random) -> tuple[str, str]:
    language, template = rng.choice(TEMPLATES[label])
    state = template.format(product=rng.choice(PRODUCTS), time=rng.choice(TIMES), topic=rng.choice(OTHER_TOPICS))
    prefix, suffix = rng.choice(WRAPPERS[language])
    # Small ASR-like surface variants retain the same meaning and label.
    if language == "hinglish" and rng.random() < 0.30:
        state = state.replace("hai", "h").replace("nahi", "nhi")
    elif language == "en" and rng.random() < 0.20:
        state = state.lower()
    return language, prefix + state + suffix + rng.choice(DETAILS[language]) + rng.choice(CONTEXTS[language])


def build_rows(total: int, seed: int):
    rng = random.Random(seed)
    counts = {label: total // len(LABELS) + (index < total % len(LABELS)) for index, label in enumerate(LABELS)}
    seen: set[str] = set()
    for label in LABELS:
        made = 0
        group_number = 0
        while made < counts[label]:
            group_id = f"retail-synthetic-v1-{label}-group-{group_number:05d}"
            split = split_for(group_id)
            group_number += 1
            language, base_state = state_for(label, rng)
            for variant in range(3):
                if made >= counts[label]:
                    break
                state = base_state + FOLLOWUPS[language][variant]
                # Ensure generated text is unique without changing its label.
                if state in seen:
                    continue
                seen.add(state)
                options = LABELS.copy()
                rng.shuffle(options)
                yield {
                    "id": f"retail-synth-v1-{label}-{made:05d}",
                    "family": "retail_intent",
                    "lang": language,
                    "state": state,
                    "question": {"type": "choice", "instructions": rng.choice(INSTRUCTIONS), "criteria": {}, "options": options},
                    "expected": options.index(label),
                    "split": split,
                    "source": "synthetic_retail_v1",
                    "license": "CC0-1.0",
                    "provenance": {"origin": "template-generated", "license": "CC0-1.0", "notes": "Synthetic training data; not customer data."},
                    "group_id": group_id,
                }
                made += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/external/retail-intent-synthetic-v1.jsonl"))
    args = parser.parse_args()
    if args.rows < len(LABELS):
        raise ValueError(f"--rows must be at least {len(LABELS)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = {label: 0 for label in LABELS}
    split_counts = {"train": 0, "validation": 0, "test": 0}
    with args.output.open("w", encoding="utf-8") as file:
        for row in build_rows(args.rows, args.seed):
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[row["question"]["options"][row["expected"]]] += 1
            split_counts[row["split"]] += 1
    print(f"Wrote {sum(counts.values()):,} rows to {args.output}")
    print(f"Labels: {counts}")
    print(f"Splits: {split_counts}")


if __name__ == "__main__":
    main()
