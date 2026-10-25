"""Create a new, source-matched synthetic benchmark for IndicJevBench models.

The examples are deliberately authored outside IndicJevBench and never used by
the trainer. They are a transfer benchmark, not a substitute for a held-out
human-labelled production set.
"""
from __future__ import annotations

import json
from pathlib import Path


OUT = Path(__file__).with_name("validation.unseen.indicjevbench-v1.jsonl")
ROWS: list[dict] = []


def add(source: str, row_id: str, state: str, instruction: str, options: list[str], expected: str) -> None:
    ROWS.append({"id": row_id, "source": source, "state": state, "instruction": instruction,
                 "options": options, "expected": options.index(expected)})


MASSIVE = ["alarm_set", "alarm_query", "play_music", "calendar_set", "email_send", "weather_query", "transport_taxi", "general_joke"]
for intent, states in {
    "alarm_set": ["Set an alarm for six thirty tomorrow morning.", "Kal subah 6:30 ka alarm laga do.", "मुझे सोमवार को सात बजे जगाने के लिए अलार्म चाहिए।"],
    "alarm_query": ["Do I have any alarms scheduled for this weekend?", "Mera next alarm kab bajega?", "कल के लिए कौन सा अलार्म सेट है?"],
    "play_music": ["Play some calm instrumental music.", "Koi upbeat Hindi song chalao.", "आराम के लिए संगीत चलाइए।"],
    "calendar_set": ["Add a dentist appointment at 3 PM on Friday.", "Friday 3 baje dentist ka calendar event bana do.", "सोमवार सुबह मीटिंग कैलेंडर में जोड़ें।"],
    "email_send": ["Send an email to my manager saying I will be late.", "Maa ko email bhejo ki main safely pahunch gaya.", "मेरी टीम को छुट्टी का ईमेल भेजना है।"],
    "weather_query": ["Will it rain in Chennai this evening?", "Aaj Jaipur ka weather kaisa rahega?", "कल बेंगलुरु का तापमान बताइए।"],
    "transport_taxi": ["Book me a taxi to the railway station.", "Airport jaane ke liye cab bula do.", "मुझे अस्पताल के लिए टैक्सी चाहिए।"],
    "general_joke": ["Tell me a short joke.", "Ek funny joke sunao.", "कोई मजेदार चुटकुला बताइए।"],
}.items():
    for index, state in enumerate(states, start=1):
        add("intent_massive", f"massive-{intent}-{index}", state, "What does the user want to do?", MASSIVE, intent)

for intent, states in {
    "alarm_set": ["Wake me at quarter past five tomorrow.", "Agle Tuesday subah 7 baje ka alarm set karo.", "हर दिन शाम छह बजे याद दिलाने वाला अलार्म लगाइए।"],
    "play_music": ["Start a playlist for a road trip.", "Workout ke liye energetic music bajao.", "कोई पुराना फिल्मी गाना चलाइए।"],
    "email_send": ["Write and send a thank-you email to Riya.", "Client ko meeting postpone karne ka email bhejo.", "मेरे शिक्षक को अनुपस्थिति का ईमेल भेजें।"],
    "weather_query": ["Is it going to be windy in Kolkata tomorrow?", "Hyderabad mein weekend par barish hogi kya?", "मुंबई में आज शाम मौसम कैसा है?"],
}.items():
    for index, state in enumerate(states, start=4):
        add("intent_massive", f"massive-extra-{intent}-{index}", state, "What does the user want to do?", MASSIVE, intent)


BANKING = ["cash_withdrawal", "card_payment_fee_charged", "pending_transfer", "card_swallowed", "card_arrival", "card_linking", "cash_withdrawal_charge", "beneficiary_not_allowed"]
for intent, states in {
    "cash_withdrawal": ["I need to take cash out using my debit card.", "ATM se paise nikalne hain, kaise karun?", "मुझे कार्ड से नकद निकालना है।"],
    "card_payment_fee_charged": ["Why was I charged an extra fee when I paid by card?", "Card payment par additional fee kyun lagi?", "कार्ड से भुगतान करने पर शुल्क क्यों कटा?"],
    "pending_transfer": ["My bank transfer still says pending after two hours.", "Transfer kiya tha par abhi tak pending dikha raha hai.", "मेरा पैसा भेजने वाला ट्रांसफर पूरा नहीं हुआ है।"],
    "card_swallowed": ["The ATM kept my card after the transaction.", "ATM ne mera card andar rakh liya, kya karun?", "मशीन ने मेरा कार्ड वापस नहीं दिया।"],
    "card_arrival": ["When will my new physical card be delivered?", "Naya card kab tak ghar aayega?", "मेरा कार्ड डिलीवरी में कहां तक पहुंचा है?"],
    "card_linking": ["How can I link my new card to the banking app?", "Naya card app mein add karna hai.", "नए कार्ड को खाते से जोड़ने की प्रक्रिया बताइए।"],
    "cash_withdrawal_charge": ["The ATM deducted a withdrawal charge from my account.", "Cash nikalne par extra ATM charge laga hai.", "नकद निकालने के लिए मुझसे अतिरिक्त फीस ली गई।"],
    "beneficiary_not_allowed": ["The app will not let me add this payee as a beneficiary.", "Beneficiary add karte time not allowed error aa raha hai.", "मैं नए लाभार्थी को जोड़ नहीं पा रहा हूं।"],
}.items():
    for index, state in enumerate(states, start=1):
        add("fintech_banking77", f"banking-{intent}-{index}", state, "What is the customer's banking intent?", BANKING, intent)

for intent, states in {
    "pending_transfer": ["The recipient has not received the transfer I made this morning.", "Money transfer abhi bhi processing mein hai.", "भेजा हुआ ट्रांसफर अभी तक पूरा नहीं हुआ।"],
    "card_arrival": ["My replacement card has not arrived yet.", "Card dispatch hua tha, delivery date batao.", "मेरा नया कार्ड कब मिलेगा?"],
    "card_linking": ["The app says it cannot connect my card to my profile.", "Debit card ko app se link karne mein error aa raha hai.", "कार्ड को मोबाइल बैंकिंग से जोड़ना है।"],
    "beneficiary_not_allowed": ["Why is this recipient blocked when I try to save them?", "Naya payee add nahi ho raha, beneficiary reject ho raha hai.", "लाभार्थी जोड़ते समय अनुमति नहीं मिल रही है।"],
}.items():
    for index, state in enumerate(states, start=4):
        add("fintech_banking77", f"banking-extra-{intent}-{index}", state, "What is the customer's banking intent?", BANKING, intent)


LID_OPTIONS = ["Devanagari-only", "Latin-only", "Both Devanagari and Latin"]
LID_ROWS = [
    ("आज बहुत गर्मी है", "Devanagari-only"), ("मुझे ट्रेन का समय बताओ", "Devanagari-only"), ("कल ऑफिस जल्दी जाना है", "Devanagari-only"),
    ("Aaj bahut garmi hai", "Latin-only"), ("Mujhe train ka time batao", "Latin-only"), ("Kal office jaldi jana hai", "Latin-only"),
    ("Aaj meeting hai kya?", "Latin-only"), ("Please mujhe update bhejo", "Latin-only"), ("Mera account active hai", "Latin-only"),
    ("आज meeting कितने बजे है?", "Both Devanagari and Latin"), ("मुझे office जल्दी जाना है", "Both Devanagari and Latin"), ("कल train से Delhi जाना है", "Both Devanagari and Latin"),
    ("मेरा password reset करना है", "Both Devanagari and Latin"), ("आज weather बहुत अच्छा है", "Both Devanagari and Latin"), ("इस app में login नहीं हो रहा", "Both Devanagari and Latin"),
    ("Good morning", "Latin-only"), ("नमस्ते मित्र", "Devanagari-only"), ("भाई please जल्दी बताओ", "Both Devanagari and Latin"),
    ("Kya tum free ho", "Latin-only"), ("मेरी बुकिंग सफल हुई", "Devanagari-only"), ("Order का status check करो", "Both Devanagari and Latin"),
    ("Maine payment kar diya", "Latin-only"), ("मुझे सहायता चाहिए", "Devanagari-only"), ("Phone का battery low है", "Both Devanagari and Latin"),
]
for index, (state, label) in enumerate(LID_ROWS, start=1):
    add("hinglish_lid", f"lid-{index}", state, "Which script is used in this message?", LID_OPTIONS, label)

for index, (state, label) in enumerate([
    ("आज घर जल्दी पहुंचना है", "Devanagari-only"), ("मुझे कल डॉक्टर के पास जाना है", "Devanagari-only"),
    ("यह संदेश केवल हिंदी में लिखा है", "Devanagari-only"), ("कृपया मुझे रास्ता बताइए", "Devanagari-only"),
    ("Mujhe kal doctor ke paas jana hai", "Latin-only"), ("Yeh message sirf Hindi mein likha hai", "Latin-only"),
    ("Kripya mujhe rasta bataiye", "Latin-only"), ("Aaj ghar jaldi pahunchna hai", "Latin-only"),
    ("आज doctor के पास जाना है", "Both Devanagari and Latin"), ("यह message बहुत जरूरी है", "Both Devanagari and Latin"),
    ("कृपया route जल्दी बताइए", "Both Devanagari and Latin"), ("घर पहुंचकर laptop charge करना है", "Both Devanagari and Latin"),
], start=25):
    add("hinglish_lid", f"lid-extra-{index}", state, "Which script is used in this message?", LID_OPTIONS, label)


RETAIL = ["track_order", "return_item", "cancel_order", "refund_status", "wrong_item", "size_issue", "payment_issue", "delivery_address", "other"]
ENTERPRISE = [
    ("Mera parcel teen din se hub par hai, kab deliver hoga?", "What is the customer's intent?", RETAIL, "track_order"),
    ("This delivered product is unused and I want to send it back.", "What is the customer's intent?", RETAIL, "return_item"),
    ("ऑर्डर भेजने से पहले इसे रद्द कर दीजिए।", "What is the customer's intent?", RETAIL, "cancel_order"),
    ("Return complete hua par refund bank mein nahi aaya.", "What is the customer's intent?", RETAIL, "refund_status"),
    ("The package contains a different item than the one in my invoice.", "What is the customer's intent?", RETAIL, "wrong_item"),
    ("मिले हुए जूते बहुत छोटे हैं, बड़ा साइज़ चाहिए।", "What is the customer's intent?", RETAIL, "size_issue"),
    ("UPI amount was debited but checkout failed.", "What is the customer's intent?", RETAIL, "payment_issue"),
    ("Delivery se pehle flat number change karna hai.", "What is the customer's intent?", RETAIL, "delivery_address"),
    ("दो बार पैसे कट गए हैं और व्यापारी को भुगतान नहीं मिला है।", "Rate the urgency (1=very low, 5=critical)", ["1 - very low", "2 - low", "3 - medium", "4 - high", "5 - critical"], "5 - critical"),
    ("I ordered a lamp but it arrived one day late.", "Rate the urgency (1=very low, 5=critical)", ["1 - very low", "2 - low", "3 - medium", "4 - high", "5 - critical"], "2 - low"),
    ("मेरा कार्ड चोरी हो गया और अनजान भुगतान दिख रहे हैं।", "Should this be escalated to a human agent?", ["no", "yes"], "yes"),
    ("Password reset page is not loading, can it wait for email support?", "Should this be escalated to a human agent?", ["no", "yes"], "no"),
]
for index, (state, instruction, options, expected) in enumerate(ENTERPRISE, start=1):
    add("synthetic_enterprise", f"enterprise-{index}-a", state, instruction, options, expected)
    add("synthetic_enterprise", f"enterprise-{index}-b", state + " Please help.", instruction, options, expected)

ENTERPRISE_EXTRA = [
    ("Courier delivered my package to the wrong city.", "What is the customer's intent?", RETAIL, "delivery_address"),
    ("Maine wrong colour ka item receive kiya hai.", "What is the customer's intent?", RETAIL, "wrong_item"),
    ("The shop charged my card but no receipt or order appeared.", "What is the customer's intent?", RETAIL, "payment_issue"),
    ("रिफंड का संदेश आया था लेकिन पैसा नहीं मिला।", "What is the customer's intent?", RETAIL, "refund_status"),
    ("A stranger used my account for several payments today.", "Should this be escalated to a human agent?", ["no", "yes"], "yes"),
    ("Mujhe product ke size guide ka link chahiye.", "Should this be escalated to a human agent?", ["no", "yes"], "no"),
]
for index, (state, instruction, options, expected) in enumerate(ENTERPRISE_EXTRA, start=13):
    add("synthetic_enterprise", f"enterprise-extra-{index}-a", state, instruction, options, expected)
    add("synthetic_enterprise", f"enterprise-extra-{index}-b", state + " कृपया सहायता करें।", instruction, options, expected)


def expand_to_1000(rows: list[dict]) -> list[dict]:
    """Create 250 distinct, label-preserving probes for each source family."""
    target_per_source = 250
    prefixes = ["Please help with this: ", "I need assistance: ", "Quick question: ", "For my request, ", "Could you handle this: "]
    suffixes = ["", " Please advise.", " I need an update.", " Kindly assist.", " This is important."]
    instructions = {
        "intent_massive": ["What does the user want to do?", "Classify the user's request.", "Select the appropriate assistant intent."],
        "fintech_banking77": ["What is the customer's banking intent?", "Choose the banking issue being described.", "Classify this banking request."],
        "synthetic_enterprise": [],
    }
    lid_tails = {
        "Devanagari-only": ["", " कृपया बताइए", " अभी मदद चाहिए", " धन्यवाद", " जल्द बताएं", " सही जानकारी दें", " मुझे समझाइए"],
        "Latin-only": ["", " please help", " jaldi batao", " thanks", " abhi batao", " detail mein bolo", " mujhe samjhao"],
        "Both Devanagari and Latin": ["", " please बताइए", " जल्दी help करें", " thanks दोस्त", " update चाहिए", " सही answer दें", " मुझे explain करो"],
    }
    expanded: list[dict] = []
    for source in ("intent_massive", "fintech_banking77", "hinglish_lid", "synthetic_enterprise"):
        seeds = [row for row in rows if row["source"] == source]
        if not seeds:
            raise ValueError(f"No seeds for {source}")
        for number in range(target_per_source):
            seed = dict(seeds[number % len(seeds)])
            variant = number // len(seeds)
            seed["id"] = f"{seed['id']}-v{variant + 1}"
            expected_option = seed["options"][seed["expected"]]
            if source == "hinglish_lid":
                seed["state"] = seed["state"] + lid_tails[expected_option][variant % len(lid_tails[expected_option])]
            else:
                seed["state"] = prefixes[variant % len(prefixes)] + seed["state"] + suffixes[(variant // len(prefixes)) % len(suffixes)]
                instruction_variants = instructions.get(source)
                if instruction_variants:
                    seed["instruction"] = instruction_variants[variant % len(instruction_variants)]
            expanded.append(seed)
    return expanded


if __name__ == "__main__":
    rows = expand_to_1000(ROWS)
    OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} source-matched synthetic examples to {OUT}")
