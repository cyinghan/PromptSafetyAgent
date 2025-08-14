import re
import json
import pickle
import numpy as np
import unicodedata
import spacy
from collections import Counter

# must download first
# python -m spacy download en_core_web_sm

nlp = spacy.load('en_core_web_sm')
def count_words(text):
    doc = nlp(text)
    return len([token for token in doc if not token.is_punct and not token.is_space])

def preprocess_spacy(text):
    doc = nlp(text.lower())
    tokens = [
        token.lemma_  # lemmatized form
        for token in doc
        if not token.is_stop and token.is_alpha
    ]
    return " ".join(tokens)

def get_numbered_list(items, sep=", "):
    """return a numbered list of items as string"""
    numbered = [f"{i}. {item}" for i, item in enumerate(items, 1)]
    return sep.join(numbered)

def separator(text="", width=80):
    """
    Creates a separator line with optional centered text, padded with '=' characters.
    """
    if text:
        formatted_text = f" {text} "
        return formatted_text.center(width, "=")
    else:
        return "=" * width

# Check and print for any duplicates in the prompts
def print_duplicates(values):
    counter = Counter(values)
    for key, count in counter.items():
        if count > 1:
            print(f"Count: {count} for Prompt: {key[:100]}")


# --- common indicators (raw phrases) ---
INDICATORS = [
    "DAN",
    "ignore all previous instructions",
    "disregard previous instructions",
    "forget everything",
    "override prior instructions",
    "disable safety",
    "turn off filters",
    "bypass authentication",
    "bypass.*2fa",
    "how to bypass",
    "enable developer mode",
    "developer mode",
    "pretend to be",
    "act as if you are",
    "provide (?:password|credentials|credit card)",
    "give me instructions to",
    "jailbreak",
    "prompt injection",
    "sudo",
    "root access",
    "do not follow (?:policy|rules|guidelines)",
]

# --- Normalization helpers ---
def normalize_text(s):
    # unicode normal form, lower, remove zero-width chars, collapse whitespace
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r'[\u200b\u200c\u200d\uFEFF]', '', s)  # zero-width
    s = s.lower()
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# optional simple leetspeak normalizer (a few common subs)
LEET_MAP = str.maketrans({
    '0':'o','1':'i','3':'e','4':'a','5':'s','7':'t','@':'a','$':'s'
})
def deobfuscate_leet(s):
    return s.translate(LEET_MAP)

# compile robust regex patterns from indicators
def build_patterns(indicators):
    patterns = []
    for raw in indicators:
        # if raw already contains regex-like tokens keep it; otherwise escape and allow separators
        if any(ch in raw for ch in ".*?[]()|"):
            pat = raw
        else:
            # create flexible pattern allowing non-word separators between words
            words = raw.split()
            pat = r'\b' + r'\W*'.join(re.escape(w) for w in words) + r'\b'
        patterns.append(re.compile(pat, flags=re.IGNORECASE))
    return patterns

PATTERNS = build_patterns(INDICATORS)

# detector function
def detect_indicators(text, use_leet=False, return_snippets=True, snippet_window=40):
    orig = text
    text_norm = normalize_text(text)
    if use_leet:
        text_norm = deobfuscate_leet(text_norm)

    hits = []
    for pat in PATTERNS:
        for m in pat.finditer(text_norm):
            start, end = m.span()
            snippet = None
            if return_snippets:
                # map back to original snippet roughly by using character indices on normalized text
                snippet = text_norm[max(0, start-snippet_window):min(len(text_norm), end+snippet_window)]
            hits.append({
                "pattern": pat.pattern,
                "matched_text": m.group(0),
                "start": start,
                "end": end,
                "snippet": snippet
            })
    found = len(hits) > 0
    return {"found": found, "matches": hits, "normalized_text": text_norm}

def clean_text(example):
    text = example["text"]

    # Normalize Unicode (e.g., é → e if needed)
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters
    text = re.sub(r"[\x00-\x1F\x7F]", "", text)

    # Remove odd symbols (keep letters, numbers, punctuation)
    text = re.sub(r"[^\w\s.,;:!?'\-\"()]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    example["text"] = text
    return example

def clean_single_text(text):
    # Normalize Unicode (e.g., é → e if needed)
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters
    text = re.sub(r"[\x00-\x1F\x7F]", "", text)

    # Remove odd symbols (keep letters, numbers, punctuation)
    text = re.sub(r"[^\w\s.,;:!?'\-\"()]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def extract_key_contexts(
    doc: str,
    vectorizer,
    top_k: int = 5,
    min_n_words: int = 1,
    max_n_words: int = 4,
    normalize_by_length: bool = False
):
    """
    Return up to top_k non-overlapping key contexts (short phrases) from doc.
    Each result is (phrase, score, start_token_idx, end_token_idx).
    - normalize_by_length: if True, uses mean TF-IDF across tokens (helps avoid bias to longer windows).
    """
    # 1) Analyzer that matches how vectorizer tokenizes
    analyzer = vectorizer.build_analyzer()
    tokens = analyzer(doc)  # list of tokens in order

    if len(tokens) == 0:
        return []

    # 2) get tf-idf vector for this doc (dense for easy lookup)
    tfidf_vec = vectorizer.transform([doc])
    doc_tfidf = tfidf_vec.toarray()[0]  # shape = (n_features,)

    # mapping token -> feature index (fast membership)
    feat_names = vectorizer.get_feature_names_out()
    feat_index = {w: i for i, w in enumerate(feat_names)}

    # 3) per-token weights aligned with tokens list
    token_weights = np.array([doc_tfidf[feat_index[t]] if t in feat_index else 0.0 for t in tokens])

    # 4) build candidate windows and score them
    candidates = []
    n = len(tokens)
    for start in range(n):
        for L in range(min_n_words, max_n_words + 1):
            end = start + L
            if end > n:
                break
            w = token_weights[start:end]
            score = float(w.sum())
            if normalize_by_length and L > 0:
                score /= L
            phrase = " ".join(tokens[start:end])
            # skip empty/zero-score candidates optionally (but keep zeros in case doc has low tfidf)
            candidates.append((score, start, end, phrase))

    # 5) sort by descending score
    candidates.sort(key=lambda x: x[0], reverse=True)

    # 6) greedy pick non-overlapping top_k windows
    selected = []
    occupied = [False] * n  # token index occupancy flag
    for score, start, end, phrase in candidates:
        if len(selected) >= top_k:
            break
        # check overlap
        if any(occupied[i] for i in range(start, end)):
            continue
        # accept and mark occupied
        selected.append((phrase, score, start, end))
        for i in range(start, end):
            occupied[i] = True

    return selected

def simple_json_parse(json_str):
    matches = re.findall(r'"(.*?)":\s*(?:"(.*?)\n|([\d.]+))', json_str)
    return {key: (val if val else float(num)) for key, val, num in matches}

def parse_json_with_repair(text):
    repaired = re.sub(r'"\s+"', '", "', text)  # fix missing commas between keys
    repaired = re.sub(r'}\s*{', '}, {', repaired)  # fix object concatenation
    try:
        return json.loads(repaired) # retry parsing
    except:
        simple_json = simple_json_parse(repaired)
        if validate_prompt_json_output(simple_json):
            return simple_json
        raise ValueError(f"Could not parse or repair JSON")

def validate_prompt_json_output(data):
    required_fields = {
        "reasoning": str,
        "recommendation": str,
        "confidence_score": (float, int, str)  # allow int for flexibility
    }

    for field, expected_type in required_fields.items():
        if field not in data:
            raise ValueError(f"Missing required field: '{field}'")

        if not isinstance(data[field], expected_type):
            raise ValueError(f"Field '{field}' must be of type {expected_type}, got {type(data[field])}")

    score = float(data["confidence_score"])
    if not (0.0 <= score <= 1.0):
        raise ValueError("Field 'confidence_score' must be between 0.0 and 1.0")
    return True

def truncate_prompt(prompt, first_n_words=100):
    words = prompt.split()[:first_n_words]
    return ' '.join(words)

def save_with_pickle(vectorizer, filepath):
    """Save vectorizer using pickle"""
    with open(filepath, 'wb') as f:
        pickle.dump(vectorizer, f)

def load_with_pickle(filepath):
    """Load vectorizer using pickle"""
    with open(filepath, 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer