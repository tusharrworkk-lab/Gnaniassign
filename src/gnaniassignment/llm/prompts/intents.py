"""One Jev Choice question for the assistant's primary intent."""

INTENT_KEYS = (
    "user_core",
    "user_operations",
    "user_timeline",
    "product_what",
    "product_why",
    "product_when",
)

QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "What is the primary intent of `current_query`? Use `recent_conversation` only to resolve references and follow-up questions. Choose one option.",
        "criteria": {
            "user_core": "A question about what the company or person does, why it exists, or where it operates.",
            "user_operations": "A question about how the company or person works: process, delivery, support, or implementation.",
            "user_timeline": "A question about when the company or person starts, becomes available, or completes work.",
            "product_what": "A question about what a product is or does, its features, compatibility, or price.",
            "product_why": "A question about why to choose a product, its benefits, value, fit, or comparisons.",
            "product_when": "A question about when a product launches, is available, is delivered, renewed, or supported.",
            "other": "The current query cannot be assigned to any of these categories, even with recent conversation context.",
        },
    },
}
