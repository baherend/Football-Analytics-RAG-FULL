"""
src/knowledge/preprocessing.py -- text normalization for the offline pipeline.

Migration Step 6: extracted verbatim from 02_preprocessing.py, which keeps its
CLI `main()` and re-exports these for compatibility.

Pure transformation: text in, text out. No filesystem access, no artifact
paths, no competition logic -- the CLI orchestrator owns those.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["normalize_unicode", "clean_whitespace", "normalize_punctuation",
           "remove_control_chars", "preprocess_text"]


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters (NFC form)."""
    return unicodedata.normalize("NFC", text)


def clean_whitespace(text: str) -> str:
    """Collapse multiple whitespace into single space, strip ends."""
    return re.sub(r'\s+', ' ', text).strip()


def normalize_punctuation(text: str) -> str:
    """Normalize smart quotes and special punctuation."""
    replacements = {
        '‘': "'", '’': "'",  # smart single quotes
        '“': '"', '”': '"',  # smart double quotes
        '–': '-', '—': '-',  # en/em dashes
        '…': '...',               # ellipsis
        ' ': ' ',                 # non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def remove_control_chars(text: str) -> str:
    """Remove control characters except newline and tab."""
    return ''.join(
        ch for ch in text
        if unicodedata.category(ch)[0] != 'C' or ch in '\n\t'
    )


def preprocess_text(text: str) -> str:
    """Apply all preprocessing steps to text."""
    text = normalize_unicode(text)
    text = normalize_punctuation(text)
    text = remove_control_chars(text)
    text = clean_whitespace(text)
    return text
