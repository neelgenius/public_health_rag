import re
import unicodedata

def clean_text(text: str) -> str:
    """
    Normalize text without changing medical meaning.
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\u2022", "-", text)  # bullet normalization
    text = text.strip()
    return text

