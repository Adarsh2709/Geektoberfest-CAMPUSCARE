import os
import re
import unicodedata
import pandas as pd

from django.conf import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Multilingual configuration & Lexicons
# -----------------------------

TOKEN_PATTERN = r"[\u0900-\u097Fa-zA-Z0-9_]+"

MARATHI_INDICATORS = {
    "नाही", "आहे", "मध्ये", "येत", "वसतिगृह", "दुरुस्ती", "बाक", "वेळापत्रक",
    "महाविद्यालय", "उशिरा", "चालक", "थांब्या", "मुलांचे", "मुलींचे", "दिवा",
    "नळ", "स्वच्छतागृह", "हजेरी", "गुण", "प्रयोगशाळा", "झाली", "पडले", "चालू",
    "करा", "इमारत", "वर्गातील", "तक्रार", "वाहतूक", "अडचण", "होतात", "केली"
}

HINGLISH_INDICATORS = {
    "mein", "me", "nahi", "nahin", "hai", "hain", "raha", "rahi", "rahe",
    "kaam", "paani", "dikkat", "toota", "tooti", "kharab", "khatra", "jaldi",
    "bohot", "bahut", "aaj", "ka", "ki", "ke", "ko", "mere", "meri", "mera",
    "chahiye", "band", "karo", "chal", "kuch", "kare", "hoga", "hogi", "tha",
    "thi", "the", "bhi", "shuru", "aur", "par", "bhai", "yaar", "ab", "kab"
}

HIGH_PRIORITY_WORDS = [
    # English
    "emergency", "urgent", "danger", "fire", "accident", "unsafe",
    "medical", "security", "immediately", "power outage", "no water",
    # Hindi
    "आपातकालीन", "तुरंत", "खतरा", "आग", "दुर्घटना", "असुरक्षित",
    "मेडिकल", "सुरक्षा", "तत्काल", "बिजली गुल", "पानी नहीं", "पानी बंद",
    # Hinglish
    "khatra", "aag", "asurakshit", "turant", "jaldi", "tatkaal",
    "bijli gul", "paani nahi", "power cut", "light nahi",
    # Marathi
    "आणीबाणी", "तातडीने", "धोका", "अपघात", "वैद्यकीय", "लगेच",
    "वीज गेली", "पाणी बंद"
]

MEDIUM_PRIORITY_WORDS = [
    # English
    "not working", "broken", "slow", "problem", "issue", "damaged", "missing",
    # Hindi
    "काम नहीं कर रहा", "काम नहीं", "खराब", "टूटा", "टूटी", "धीमा",
    "धीमी", "समस्या", "दिक्कत", "गायब",
    # Hinglish
    "kaam nahi kar raha", "kaam nahi", "work nahi kar raha", "kharab",
    "dikkat", "toota", "tooti", "chal nahi raha", "gayab",
    # Marathi
    "चालत नाही", "काम करत नाही", "नादुरुस्त", "बंद पडले", "तुटलेले",
    "तुटला", "हळू", "अडचण", "दिसत नाही"
]


# -----------------------------
# Text Preprocessing & Language Detection
# -----------------------------

def preprocess_text(text):
    """Normalize unicode and strip surrounding whitespace."""
    if not text:
        return ""
    return unicodedata.normalize("NFC", str(text).strip())


def detect_language(text):
    """
    Detects whether the complaint is in English, Hindi, Hinglish, or Marathi.
    - If Devanagari characters are present: distinguishes Marathi vs Hindi based on lexical indicators.
    - If Latin characters: checks for common Hinglish markers; otherwise defaults to English.
    """
    if not text:
        return "English"

    text = preprocess_text(text)

    # Devanagari script detection (\u0900-\u097F)
    if re.search(r"[\u0900-\u097F]", text):
        devanagari_words = set(re.findall(r"[\u0900-\u097F]+", text))
        if devanagari_words & MARATHI_INDICATORS:
            return "Marathi"
        return "Hindi"

    # Latin script: Hinglish vs English detection
    tokens = set(re.findall(r"[a-zA-Z]+", text.lower()))
    if tokens & HINGLISH_INDICATORS:
        return "Hinglish"

    return "English"


# -----------------------------
# Load training dataset
# -----------------------------

DATA_PATH = os.path.join(
    settings.BASE_DIR,
    "data",
    "complaints.csv"
)

data = pd.read_csv(DATA_PATH)


# -----------------------------
# Train category model
# -----------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    token_pattern=TOKEN_PATTERN
)

X = vectorizer.fit_transform(data["complaint"])

category_model = LogisticRegression(
    max_iter=1000
)

category_model.fit(
    X,
    data["category"]
)


# -----------------------------
# Predict complaint category
# -----------------------------

def predict_category(text):
    cleaned = preprocess_text(text)
    text_vector = vectorizer.transform([cleaned])
    return category_model.predict(text_vector)[0]


# -----------------------------
# Predict priority
# -----------------------------

def predict_priority(text):
    text_lower = preprocess_text(text).lower()

    if any(word in text_lower for word in HIGH_PRIORITY_WORDS):
        return "High"

    if any(word in text_lower for word in MEDIUM_PRIORITY_WORDS):
        return "Medium"

    return "Low"


# -----------------------------
# Detect duplicate complaints
# -----------------------------

def detect_duplicate(text):
    from .models import Complaint

    complaints = list(
        Complaint.objects.values_list(
            "description",
            flat=True
        )
    )

    if not complaints:
        return False

    cleaned_text = preprocess_text(text)
    all_text = [preprocess_text(c) for c in complaints] + [cleaned_text]

    duplicate_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=TOKEN_PATTERN
    )

    matrix = duplicate_vectorizer.fit_transform(all_text)

    similarities = cosine_similarity(
        matrix[-1],
        matrix[:-1]
    )[0]

    return similarities.max() >= 0.75