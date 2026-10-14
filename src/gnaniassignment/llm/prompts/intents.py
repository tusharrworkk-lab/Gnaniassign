"""One Laya Choice question for the assistant's primary intent."""

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
        "instructions": "`current_query` का मुख्य उद्देश्य क्या है? `recent_conversation` का उपयोग केवल संदर्भ और फ़ॉलो-अप प्रश्न समझने के लिए करें। एक विकल्प चुनें।",
        "criteria": {
            "user_core": "कंपनी या व्यक्ति क्या करता है, क्यों मौजूद है, या कहाँ काम करता है।",
            "user_operations": "कंपनी या व्यक्ति कैसे काम करता है: प्रक्रिया, डिलीवरी, सहायता या कार्यान्वयन।",
            "user_timeline": "कंपनी या व्यक्ति कब शुरू होता है, उपलब्ध होता है, या काम पूरा करता है।",
            "product_what": "उत्पाद क्या है या क्या करता है, उसकी विशेषताएँ, अनुकूलता या कीमत।",
            "product_why": "उत्पाद क्यों चुनें: फायदे, मूल्य, उपयुक्तता या तुलना।",
            "product_when": "उत्पाद कब लॉन्च, उपलब्ध, डिलीवर, नवीनीकृत या समर्थित होता है।",
            "other": "वर्तमान प्रश्न इनमें से किसी भी श्रेणी में नहीं आता है।",
        },
    },
}
