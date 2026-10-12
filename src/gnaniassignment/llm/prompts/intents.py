"""Intent questions for the decision model."""

"""The six mutually exclusive intents used by the enquiry assistant."""

INTENT_KEYS = (
    "user_core",
    "user_operations",
    "user_timeline",
    "product_what",
    "product_why",
    "product_when",
)

QUESTIONS = {
    "user_core": {
        "type": "noul",
        "instructions": "Is this a core user/company question about what it is, why it exists, or where it operates? Examples: 'what do you do?', 'why were you founded?', 'where do you operate?'.",
    },
    "user_operations": {
        "type": "noul",
        "instructions": "Is this a user/company operational question about how it works, process, delivery, support, or implementation?",
    },
    "user_timeline": {"type": "noul", "instructions": "Is this a user/company timeline question: when something starts, is available, launches, or is completed?"},
    "product_what": {"type": "noul", "instructions": "Is this a product enquiry asking what a product is, its features, capabilities, compatibility, or pricing?"},
    "product_why": {"type": "noul", "instructions": "Is this a product enquiry asking why to use it, its value, benefits, or comparison?"},
    "product_when": {"type": "noul", "instructions": "Is this a product enquiry asking when it launches, is available, will be delivered, renewed, or supported?"},
}
