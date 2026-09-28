"""Everything the assistant says outside the interview itself.

Scripted, not generated: these lines must come out identical every session, and a local
model paraphrasing "say resume or document" is how a kiosk becomes unusable. They live on
the server so the page and the speech cache share one copy - the warm-up synthesizes
exactly what the page will ask for, which is what makes the greeting instant.
"""

SCRIPT: dict[str, dict[str, str]] = {
    "askLanguage": {
        "en": "Welcome to TePMA. Which language would you like to continue in? You can say English, Hindi, or Punjabi.",
        "hi": "टेपमा में आपका स्वागत है। आप किस भाषा में बात करना चाहेंगे? अंग्रेज़ी, हिंदी, या पंजाबी।",
        "pa": "ਟੇਪਮਾ ਵਿੱਚ ਤੁਹਾਡਾ ਸਵਾਗਤ ਹੈ। ਤੁਸੀਂ ਕਿਸ ਭਾਸ਼ਾ ਵਿੱਚ ਗੱਲ ਕਰਨਾ ਚਾਹੋਗੇ? ਅੰਗਰੇਜ਼ੀ, ਹਿੰਦੀ, ਜਾਂ ਪੰਜਾਬੀ।"
    },
    "languageUnclear": {
        "en": "Sorry, I did not catch that. Please say English, Hindi, or Punjabi.",
        "hi": "माफ़ कीजिए, मैं समझ नहीं पाया। कृपया कहिए अंग्रेज़ी, हिंदी, या पंजाबी।",
        "pa": "ਮਾਫ਼ ਕਰਨਾ, ਮੈਂ ਸਮਝ ਨਹੀਂ ਸਕਿਆ। ਕਿਰਪਾ ਕਰਕੇ ਕਹੋ ਅੰਗਰੇਜ਼ੀ, ਹਿੰਦੀ, ਜਾਂ ਪੰਜਾਬੀ।"
    },
    "finishedEmailed": {
        "en": "Your online profile has been made and sent to your email. You can collect the printed resume from the printer. Thank you.",
        "hi": "आपका ऑनलाइन प्रोफ़ाइल बन गया है और आपके ईमेल पर भेज दिया गया है। छपा हुआ रिज़्यूमे आप प्रिंटर से ले सकते हैं। धन्यवाद।",
        "pa": "ਤੁਹਾਡੀ ਆਨਲਾਈਨ ਪ੍ਰੋਫ਼ਾਈਲ ਬਣ ਗਈ ਹੈ ਅਤੇ ਤੁਹਾਡੇ ਈਮੇਲ ਉੱਤੇ ਭੇਜ ਦਿੱਤੀ ਗਈ ਹੈ। ਛਪਿਆ ਹੋਇਆ ਰਿਜ਼ਿਊਮੇ ਤੁਸੀਂ ਪ੍ਰਿੰਟਰ ਤੋਂ ਲੈ ਸਕਦੇ ਹੋ। ਧੰਨਵਾਦ।"
    },
    "finishedEmailedNoPrint": {
        "en": "Your online profile has been made and sent to your email. The printer is not available right now, so please use the copy in your email. Thank you.",
        "hi": "आपका ऑनलाइन प्रोफ़ाइल बन गया है और आपके ईमेल पर भेज दिया गया है। प्रिंटर अभी उपलब्ध नहीं है, इसलिए कृपया ईमेल वाली कॉपी का उपयोग करें। धन्यवाद।",
        "pa": "ਤੁਹਾਡੀ ਆਨਲਾਈਨ ਪ੍ਰੋਫ਼ਾਈਲ ਬਣ ਗਈ ਹੈ ਅਤੇ ਤੁਹਾਡੇ ਈਮੇਲ ਉੱਤੇ ਭੇਜ ਦਿੱਤੀ ਗਈ ਹੈ। ਪ੍ਰਿੰਟਰ ਹੁਣ ਉਪਲਬਧ ਨਹੀਂ ਹੈ, ਇਸ ਲਈ ਕਿਰਪਾ ਕਰਕੇ ਈਮੇਲ ਵਾਲੀ ਕਾਪੀ ਵਰਤੋ। ਧੰਨਵਾਦ।",
    },
    # Neither sent nor printed. The kiosk must not tell someone to collect a page from a
    # printer that is not connected - the resume is on the screen and nowhere else.
    "finishedSavedOnly": {
        "en": "Your online profile has been made and is ready on the screen. The printer is not available right now, so please save or download it from here. Thank you.",
        "hi": "आपका ऑनलाइन प्रोफ़ाइल बन गया है और स्क्रीन पर तैयार है। प्रिंटर अभी उपलब्ध नहीं है, इसलिए कृपया इसे यहीं से सेव या डाउनलोड कर लीजिए। धन्यवाद।",
        "pa": "ਤੁਹਾਡੀ ਆਨਲਾਈਨ ਪ੍ਰੋਫ਼ਾਈਲ ਬਣ ਗਈ ਹੈ ਅਤੇ ਸਕ੍ਰੀਨ ਉੱਤੇ ਤਿਆਰ ਹੈ। ਪ੍ਰਿੰਟਰ ਹੁਣ ਉਪਲਬਧ ਨਹੀਂ ਹੈ, ਇਸ ਲਈ ਕਿਰਪਾ ਕਰਕੇ ਇਸਨੂੰ ਇੱਥੋਂ ਹੀ ਸੇਵ ਜਾਂ ਡਾਊਨਲੋਡ ਕਰ ਲਵੋ। ਧੰਨਵਾਦ।",
    },
    "finishedPrintOnly": {
        "en": "Your online profile has been made. You can collect the printed resume from the printer. Thank you.",
        "hi": "आपका ऑनलाइन प्रोफ़ाइल बन गया है। छपा हुआ रिज़्यूमे आप प्रिंटर से ले सकते हैं। धन्यवाद।",
        "pa": "ਤੁਹਾਡੀ ਆਨਲਾਈਨ ਪ੍ਰੋਫ਼ਾਈਲ ਬਣ ਗਈ ਹੈ। ਛਪਿਆ ਹੋਇਆ ਰਿਜ਼ਿਊਮੇ ਤੁਸੀਂ ਪ੍ਰਿੰਟਰ ਤੋਂ ਲੈ ਸਕਦੇ ਹੋ। ਧੰਨਵਾਦ।"
    },
    "askIntent": {
        "en": "How can I help you? Say résumé if you want me to build your résumé, or say document if you need a document.",
        "hi": "मैं आपकी कैसे मदद कर सकता हूँ? रिज़्यूमे बनवाने के लिए कहिए रिज़्यूमे, या कोई दस्तावेज़ चाहिए तो कहिए दस्तावेज़।",
        "pa": "ਮੈਂ ਤੁਹਾਡੀ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ? ਰਿਜ਼ਿਊਮੇ ਬਣਾਉਣ ਲਈ ਕਹੋ ਰਿਜ਼ਿਊਮੇ, ਜਾਂ ਕੋਈ ਦਸਤਾਵੇਜ਼ ਚਾਹੀਦਾ ਹੈ ਤਾਂ ਕਹੋ ਦਸਤਾਵੇਜ਼।"
    },
    "intentUnclear": {
        "en": "Sorry, I did not understand. Please say résumé, or document.",
        "hi": "माफ़ कीजिए, मैं समझ नहीं पाया। कृपया कहिए रिज़्यूमे, या दस्तावेज़।",
        "pa": "ਮਾਫ਼ ਕਰਨਾ, ਮੈਂ ਸਮਝ ਨਹੀਂ ਸਕਿਆ। ਕਿਰਪਾ ਕਰਕੇ ਕਹੋ ਰਿਜ਼ਿਊਮੇ, ਜਾਂ ਦਸਤਾਵੇਜ਼।"
    },
    "askDocument": {
        "en": "Tell me which document you need. I will look in the library first, and if it is not there I will collect the details and write it.",
        "hi": "बताइए आपको कौन-सा दस्तावेज़ चाहिए। मैं पहले लाइब्रेरी में देखूँगा, और न मिलने पर ज़रूरी जानकारी लेकर उसे तैयार करूँगा।",
        "pa": "ਦੱਸੋ ਤੁਹਾਨੂੰ ਕਿਹੜਾ ਦਸਤਾਵੇਜ਼ ਚਾਹੀਦਾ ਹੈ। ਮੈਂ ਪਹਿਲਾਂ ਲਾਇਬ੍ਰੇਰੀ ਵਿੱਚ ਦੇਖਾਂਗਾ, ਅਤੇ ਨਾ ਮਿਲਣ ਤੇ ਲੋੜੀਂਦੀ ਜਾਣਕਾਰੀ ਲੈ ਕੇ ਉਸਨੂੰ ਤਿਆਰ ਕਰਾਂਗਾ।"
    },
    "notHeard": {
        "en": "Sorry, I did not hear anything. Please say that again.",
        "hi": "माफ़ कीजिए, मुझे कुछ सुनाई नहीं दिया। कृपया फिर से कहिए।",
        "pa": "ਮਾਫ਼ ਕਰਨਾ, ਮੈਨੂੰ ਕੁਝ ਸੁਣਾਈ ਨਹੀਂ ਦਿੱਤਾ। ਕਿਰਪਾ ਕਰਕੇ ਦੁਬਾਰਾ ਕਹੋ।"
    },
    "ready": {
        "en": "Your document is ready.",
        "hi": "आपका दस्तावेज़ तैयार है।",
        "pa": "ਤੁਹਾਡਾ ਦਸਤਾਵੇਜ਼ ਤਿਆਰ ਹੈ।"
    },
    "anythingElse": {
        "en": "Is there anything else? Say résumé, say document, or say stop to finish.",
        "hi": "कुछ और चाहिए? कहिए रिज़्यूमे, कहिए दस्तावेज़, या समाप्त करने के लिए कहिए बंद।",
        "pa": "ਹੋਰ ਕੁਝ ਚਾਹੀਦਾ ਹੈ? ਕਹੋ ਰਿਜ਼ਿਊਮੇ, ਕਹੋ ਦਸਤਾਵੇਜ਼, ਜਾਂ ਖ਼ਤਮ ਕਰਨ ਲਈ ਕਹੋ ਬੰਦ।"
    },
    "goodbye": {
        "en": "Thank you for using TePMA. Goodbye.",
        "hi": "टेपमा का उपयोग करने के लिए धन्यवाद। नमस्ते।",
        "pa": "ਟੇਪਮਾ ਵਰਤਣ ਲਈ ਧੰਨਵਾਦ। ਸਤ ਸ੍ਰੀ ਅਕਾਲ।"
    }
}

# Spoken before anyone has chosen a language, so they are said in all three. These are the
# lines every single visitor hears, and the ones worth having in the cache before opening.
GREETING_KEYS = ("askLanguage", "languageUnclear")


def spoken_lines() -> list[tuple[str, str]]:
    """(text, language) for every scripted line, for pre-synthesis."""
    return [(text, language)
            for phrases in list(SCRIPT.values()) + list(SECTION_QUESTIONS.values())
            for language, text in phrases.items()]


def greeting_lines() -> list[tuple[str, str]]:
    return [(SCRIPT[key][language], language)
            for key in GREETING_KEYS
            for language in ("en", "hi", "pa")]


# Plain-language versions of every interview question, in all three languages.
#
# The interviewer model normally phrases these itself, more naturally and with follow-ups.
# These are what the kiosk falls back on when it cannot: a Punjabi turn once ran until the
# 300-second client timeout and killed the session outright. A stiffly worded question is
# a far better outcome than an interview that dies in the middle, and because these are
# fixed strings they are pre-synthesized into the speech cache like the rest of the script.
SECTION_QUESTIONS: dict[str, dict[str, str]] = {
    "identity": {
        "en": "What is your full name? Please spell it out letter by letter.",
        "hi": "आपका पूरा नाम क्या है? कृपया एक-एक अक्षर करके बताइए।",
        "pa": "ਤੁਹਾਡਾ ਪੂਰਾ ਨਾਮ ਕੀ ਹੈ? ਕਿਰਪਾ ਕਰਕੇ ਇੱਕ-ਇੱਕ ਅੱਖਰ ਕਰਕੇ ਦੱਸੋ।",
    },
    "phone": {
        "en": "What is your 10-digit mobile number? Please say it digit by digit.",
        "hi": "आपका दस अंकों का मोबाइल नंबर क्या है? कृपया एक-एक अंक करके बताइए।",
        "pa": "ਤੁਹਾਡਾ ਦਸ ਅੰਕਾਂ ਦਾ ਮੋਬਾਈਲ ਨੰਬਰ ਕੀ ਹੈ? ਕਿਰਪਾ ਕਰਕੇ ਇੱਕ-ਇੱਕ ਅੰਕ ਕਰਕੇ ਦੱਸੋ।",
    },
    "email": {
        "en": "What is your email address? Please spell it out letter by letter.",
        "hi": "आपका ईमेल पता क्या है? कृपया एक-एक अक्षर करके बताइए।",
        "pa": "ਤੁਹਾਡਾ ਈਮੇਲ ਪਤਾ ਕੀ ਹੈ? ਕਿਰਪਾ ਕਰਕੇ ਇੱਕ-ਇੱਕ ਅੱਖਰ ਕਰਕੇ ਦੱਸੋ।",
    },
    "target_role": {
        "en": "Which job or role are you applying for?",
        "hi": "आप किस नौकरी या पद के लिए आवेदन कर रहे हैं?",
        "pa": "ਤੁਸੀਂ ਕਿਹੜੀ ਨੌਕਰੀ ਜਾਂ ਅਹੁਦੇ ਲਈ ਅਰਜ਼ੀ ਦੇ ਰਹੇ ਹੋ?",
    },
    "education": {
        "en": "What is your highest qualification, from which institution, in which town or city, and in which year?",
        "hi": "आपकी सबसे बड़ी योग्यता क्या है, किस संस्थान से, वह संस्थान किस शहर में है, और किस वर्ष में?",
        "pa": "ਤੁਹਾਡੀ ਸਭ ਤੋਂ ਵੱਡੀ ਯੋਗਤਾ ਕੀ ਹੈ, ਕਿਸ ਸੰਸਥਾ ਤੋਂ, ਉਹ ਸੰਸਥਾ ਕਿਹੜੇ ਸ਼ਹਿਰ ਵਿੱਚ ਹੈ, ਅਤੇ ਕਿਹੜੇ ਸਾਲ ਵਿੱਚ?",
    },
    "experience": {
        "en": "Tell me about your work experience, and whether you are working, studying, or looking for work right now.",
        "hi": "अपने काम के अनुभव के बारे में बताइए, और यह भी कि आप अभी नौकरी कर रहे हैं, पढ़ाई कर रहे हैं, या नौकरी ढूंढ रहे हैं।",
        "pa": "ਆਪਣੇ ਕੰਮ ਦੇ ਤਜਰਬੇ ਬਾਰੇ ਦੱਸੋ, ਅਤੇ ਇਹ ਵੀ ਕਿ ਤੁਸੀਂ ਹੁਣ ਨੌਕਰੀ ਕਰ ਰਹੇ ਹੋ, ਪੜ੍ਹਾਈ ਕਰ ਰਹੇ ਹੋ, ਜਾਂ ਨੌਕਰੀ ਲੱਭ ਰਹੇ ਹੋ।",
    },
    "projects": {
        "en": "Tell me about a project or any work you can show as proof of your experience.",
        "hi": "किसी प्रोजेक्ट या ऐसे काम के बारे में बताइए जिसे आप अपने अनुभव के सबूत के तौर पर दिखा सकें।",
        "pa": "ਕਿਸੇ ਪ੍ਰੋਜੈਕਟ ਜਾਂ ਅਜਿਹੇ ਕੰਮ ਬਾਰੇ ਦੱਸੋ ਜਿਸਨੂੰ ਤੁਸੀਂ ਆਪਣੇ ਤਜਰਬੇ ਦੇ ਸਬੂਤ ਵਜੋਂ ਦਿਖਾ ਸਕੋ।",
    },
    "gaps": {
        "en": "One last thing I still need from you.",
        "hi": "आखिरी एक बात जो मुझे आपसे चाहिए।",
        "pa": "ਆਖ਼ਰੀ ਇੱਕ ਗੱਲ ਜੋ ਮੈਨੂੰ ਤੁਹਾਡੇ ਤੋਂ ਚਾਹੀਦੀ ਹੈ।",
    },
    # The closing gap pass. Each of these matches a key in _profile_gaps(); they exist
    # because "one last thing I need" does not tell the candidate WHAT to say, and the
    # gap itself is an English description written for the model, not for a person.
    "gap_name": {
        "en": "I did not catch your full name. Could you say your first and last name, spelling them out?",
        "hi": "मैं आपका पूरा नाम ठीक से नहीं समझ पाया। कृपया अपना पहला और आखिरी नाम, अक्षर-अक्षर करके बताइए।",
        "pa": "ਮੈਂ ਤੁਹਾਡਾ ਪੂਰਾ ਨਾਮ ਠੀਕ ਤਰ੍ਹਾਂ ਨਹੀਂ ਸਮਝ ਸਕਿਆ। ਕਿਰਪਾ ਕਰਕੇ ਆਪਣਾ ਪਹਿਲਾ ਅਤੇ ਆਖ਼ਰੀ ਨਾਮ ਅੱਖਰ-ਅੱਖਰ ਕਰਕੇ ਦੱਸੋ।",
    },
    "gap_email": {
        "en": "I did not get your email address correctly. Could you spell it out, letter by letter?",
        "hi": "मुझे आपका ईमेल पता ठीक से नहीं मिला। कृपया उसे एक-एक अक्षर करके बताइए।",
        "pa": "ਮੈਨੂੰ ਤੁਹਾਡਾ ਈਮੇਲ ਪਤਾ ਠੀਕ ਨਹੀਂ ਮਿਲਿਆ। ਕਿਰਪਾ ਕਰਕੇ ਉਸਨੂੰ ਇੱਕ-ਇੱਕ ਅੱਖਰ ਕਰਕੇ ਦੱਸੋ।",
    },
    "gap_phone": {
        "en": "I did not get your mobile number correctly. Could you say all ten digits, one by one?",
        "hi": "मुझे आपका मोबाइल नंबर ठीक से नहीं मिला। कृपया दसों अंक एक-एक करके बताइए।",
        "pa": "ਮੈਨੂੰ ਤੁਹਾਡਾ ਮੋਬਾਈਲ ਨੰਬਰ ਠੀਕ ਨਹੀਂ ਮਿਲਿਆ। ਕਿਰਪਾ ਕਰਕੇ ਦਸੇ ਅੰਕ ਇੱਕ-ਇੱਕ ਕਰਕੇ ਦੱਸੋ।",
    },
    "gap_target_role": {
        "en": "Which job or role should I put on your resume?",
        "hi": "मैं आपके रिज़्यूमे पर कौन सी नौकरी या पद लिखूँ?",
        "pa": "ਮੈਂ ਤੁਹਾਡੇ ਰਿਜ਼ਿਊਮੇ ਉੱਤੇ ਕਿਹੜੀ ਨੌਕਰੀ ਜਾਂ ਅਹੁਦਾ ਲਿਖਾਂ?",
    },
    "gap_education": {
        "en": "What is your highest qualification, from which institution, in which town or city, and in which year?",
        "hi": "आपकी सबसे बड़ी योग्यता क्या है, किस संस्थान से, वह संस्थान किस शहर में है, और किस वर्ष में?",
        "pa": "ਤੁਹਾਡੀ ਸਭ ਤੋਂ ਵੱਡੀ ਯੋਗਤਾ ਕੀ ਹੈ, ਕਿਸ ਸੰਸਥਾ ਤੋਂ, ਉਹ ਸੰਸਥਾ ਕਿਹੜੇ ਸ਼ਹਿਰ ਵਿੱਚ ਹੈ, ਅਤੇ ਕਿਹੜੇ ਸਾਲ ਵਿੱਚ?",
    },
    "gap_experience": {
        "en": "Could you tell me about any work, internship or project you can show?",
        "hi": "क्या आप किसी काम, इंटर्नशिप या प्रोजेक्ट के बारे में बता सकते हैं?",
        "pa": "ਕੀ ਤੁਸੀਂ ਕਿਸੇ ਕੰਮ, ਇੰਟਰਨਸ਼ਿਪ ਜਾਂ ਪ੍ਰੋਜੈਕਟ ਬਾਰੇ ਦੱਸ ਸਕਦੇ ਹੋ?",
    },
    "gap_skills": {
        "en": "Which skills would you like on your resume?",
        "hi": "आप अपने रिज़्यूमे पर कौन से कौशल लिखवाना चाहेंगे?",
        "pa": "ਤੁਸੀਂ ਆਪਣੇ ਰਿਜ਼ਿਊਮੇ ਉੱਤੇ ਕਿਹੜੇ ਹੁਨਰ ਲਿਖਵਾਉਣਾ ਚਾਹੋਗੇ?",
    },
    "gap_location": {
        "en": "Which city do you live in, and what is its 6-digit PIN code?",
        "hi": "आप किस शहर में रहते हैं, और उसका 6 अंकों का पिन कोड क्या है?",
        "pa": "ਤੁਸੀਂ ਕਿਹੜੇ ਸ਼ਹਿਰ ਵਿੱਚ ਰਹਿੰਦੇ ਹੋ, ਅਤੇ ਉਸਦਾ 6 ਅੰਕਾਂ ਦਾ ਪਿੰਨ ਕੋਡ ਕੀ ਹੈ?",
    },
    "closing": {
        "en": "Thank you, that is everything I need. Your profile is being prepared now.",
        "hi": "धन्यवाद, मुझे सारी जानकारी मिल गई। आपका प्रोफ़ाइल अभी तैयार किया जा रहा है।",
        "pa": "ਧੰਨਵਾਦ, ਮੈਨੂੰ ਸਾਰੀ ਜਾਣਕਾਰੀ ਮਿਲ ਗਈ। ਤੁਹਾਡੀ ਪ੍ਰੋਫ਼ਾਈਲ ਹੁਣ ਤਿਆਰ ਕੀਤੀ ਜਾ ਰਹੀ ਹੈ।",
    },
}


def section_question(section: str, language: str) -> str:
    phrases = SECTION_QUESTIONS.get(section) or SECTION_QUESTIONS["gaps"]
    return phrases.get(language) or phrases["en"]


_SCRIPTED: set[tuple[str, str]] | None = None


def is_scripted(text: str, language: str) -> bool:
    """Is this exact line one of the kiosk's own? Only those are worth caching."""
    global _SCRIPTED
    if _SCRIPTED is None:
        _SCRIPTED = set(spoken_lines())
    return (text, language) in _SCRIPTED
