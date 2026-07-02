"""
Shared utilities for loading the SHL catalog and building lexical/semantic
retrieval structures.

Why this file exists (see APPROACH.md for the full writeup):
- The original build_index.py threw away most of the scraped fields
  (duration, languages, job_levels, entity_id) and only embedded
  name+keys+job_levels+description. That meant the LLM never *saw*
  duration/language/test-type data at generation time, so it had to
  guess/hallucinate those columns.
- There was also no official test_type field in the raw catalog. The
  sample conversations use single-letter codes (A/B/C/D/E/K/P/S) that map
  1:1 to the `keys` category list SHL actually publishes on each product
  page ("A Ability & Aptitude, B Biodata & Situational Judgement,
  C Competencies, D Development & 360, E Assessment Exercises,
  K Knowledge & Skills, P Personality & Behavior, S Simulations"),
  confirmed against the live SHL product page for one of the catalog
  items. We derive test_type deterministically from `keys` instead of
  asking the LLM to invent it.
"""

import json
import re
import math
from collections import Counter, defaultdict

CATALOG_PATH_DEFAULT = "shl_product_catalog.json"

# Canonical SHL test-type taxonomy, in the order SHL itself lists it.
CANONICAL_KEY_ORDER = [
    ("Ability & Aptitude", "A"),
    ("Biodata & Situational Judgment", "B"),
    ("Biodata & Situational Judgement", "B"),  # UK spelling variant seen on-site
    ("Competencies", "C"),
    ("Development & 360", "D"),
    ("Assessment Exercises", "E"),
    ("Knowledge & Skills", "K"),
    ("Personality & Behavior", "P"),
    ("Personality & Behaviour", "P"),
    ("Simulations", "S"),
]
KEY_TO_LETTER = {name: letter for name, letter in CANONICAL_KEY_ORDER}
LETTER_ORDER = ["A", "B", "C", "D", "E", "K", "P", "S"]


def clean_json_text(raw_text: str) -> str:
    """Strip illegal control characters that break json.loads on this file."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw_text)


def load_raw_catalog(path: str = CATALOG_PATH_DEFAULT):
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    clean_text = clean_json_text(raw_text)
    return json.loads(clean_text, strict=False)


def derive_test_type(keys) -> str:
    """Map a list of full category names to the canonical letter-code string,
    e.g. ['Knowledge & Skills', 'Simulations'] -> 'K,S', preserving SHL's
    canonical A,B,C,D,E,K,P,S ordering (not source order, not alphabetical)."""
    if not keys:
        return ""
    letters = {KEY_TO_LETTER[k] for k in keys if k in KEY_TO_LETTER}
    ordered = [l for l in LETTER_ORDER if l in letters]
    return ",".join(ordered)


def normalize_duration(item) -> str:
    dur = (item.get("duration") or "").strip()
    if dur:
        return dur
    raw = item.get("duration_raw") or ""
    m = re.search(r"(\d+)", raw)
    if m:
        return f"{m.group(1)} minutes"
    return ""


_TRAILING_QUALIFIER_RE = re.compile(r"(\s*\([^)]*\))+$")


def core_name(name: str) -> str:
    """Strip trailing parenthetical qualifiers like '(New)', '(US)',
    '(General)' so 'SVAR Spoken English (US) (New)' -> 'SVAR Spoken English'.
    Real users refer to products by their clean name; SHL's internal
    version/region tags shouldn't be required for an exact-mention match."""
    return _TRAILING_QUALIFIER_RE.sub("", name or "").strip()


def build_record(item) -> dict:
    """Normalize one raw catalog item into the record shape used everywhere
    downstream (index metadata, LLM context, post-generation validation)."""
    name = (item.get("name") or "").strip()
    desc = (item.get("description") or "").strip()
    keys = item.get("keys") or []
    languages = item.get("languages") or []
    job_levels = item.get("job_levels") or []
    return {
        "entity_id": str(item.get("entity_id", "")),
        "name": name,
        "url": (item.get("link") or "").strip(),
        "description": desc,
        "keys": keys,
        "test_type": derive_test_type(keys),
        "duration": normalize_duration(item),
        "languages": languages,
        "job_levels": job_levels,
        "adaptive": item.get("adaptive", ""),
        "remote": item.get("remote", ""),
    }


def build_document_text(rec: dict) -> str:
    """Text that gets embedded / tokenized for retrieval. Deliberately
    includes duration/language/job-level/test-type signal, not just the
    free-text description, so queries like 'Spanish', 'entry-level',
    'under 15 minutes' etc. have something to match against."""
    keys_str = ", ".join(rec["keys"])
    langs_str = ", ".join(rec["languages"][:8])
    levels_str = ", ".join(rec["job_levels"])
    parts = [
        f"Test Name: {rec['name']}",
        f"Test Type Codes: {rec['test_type']}",
        f"Categories: {keys_str}",
        f"Target Job Levels: {levels_str}",
        f"Duration: {rec['duration']}",
        f"Languages: {langs_str}",
        f"Description: {rec['description']}",
    ]
    return "\n".join(p for p in parts if p.split(": ", 1)[-1])


# --------------------------------------------------------------------------
# Lightweight tokenizer + BM25 (dependency-free — requirements.txt already
# pulls in faiss-cpu + sentence-transformers for the dense side; adding a
# lexical index shouldn't add a third heavy dependency, so this is a small
# self-contained BM25Okapi implementation).
# --------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "with",
    "is", "are", "we", "our", "need", "needs", "you", "your", "i", "it",
    "that", "this", "what", "which", "who", "do", "does", "should", "be",
}

_TOKEN_RE = re.compile(r"[a-z0-9\+\#\.]+")

_VOWELS = set("aeiou")


def _is_vowel(word, i):
    c = word[i]
    if c in _VOWELS:
        return True
    if c == "y" and i > 0:
        return not _is_vowel(word, i - 1)
    return False


def _measure(stem):
    """Porter's VC measure: count of VC sequences after the initial (C)."""
    form = "".join("V" if _is_vowel(stem, i) else "C" for i in range(len(stem)))
    return form.count("VC")


def _contains_vowel(stem):
    return any(_is_vowel(stem, i) for i in range(len(stem)))


def _ends_double_consonant(stem):
    return len(stem) >= 2 and stem[-1] == stem[-2] and not _is_vowel(stem, len(stem) - 1)


def _cvc(stem):
    if len(stem) < 3:
        return False
    return (
        not _is_vowel(stem, len(stem) - 3)
        and _is_vowel(stem, len(stem) - 2)
        and not _is_vowel(stem, len(stem) - 1)
        and stem[-1] not in ("w", "x", "y")
    )


def _stem(word: str) -> str:
    """Minimal Porter stemmer covering steps 1a (plurals) and 1b (-ed/-ing
    with silent-e restoration, e.g. 'hiring' -> 'hire' not 'hir'). This is a
    deliberately small subset of the full algorithm -- enough to collapse
    the common recruiting-vocabulary variants (hiring/hire, screening/
    screen, graduates/graduate) that show up across the sample
    conversations, without pulling in an NLTK dependency for a take-home
    assignment that otherwise keeps requirements minimal."""
    w = word
    if len(w) <= 2:
        return w

    # Step 1a: plurals
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-3] + "i"
    elif w.endswith("ss"):
        pass
    elif w.endswith("s") and len(w) > 3:
        w = w[:-1]

    # Step 1b: -eed / -ed / -ing
    if w.endswith("eed"):
        if _measure(w[:-3]) > 0:
            w = w[:-1]
    else:
        stripped = None
        if w.endswith("ed") and _contains_vowel(w[:-2]):
            stripped = w[:-2]
        elif w.endswith("ing") and _contains_vowel(w[:-3]):
            stripped = w[:-3]
        if stripped is not None:
            w = stripped
            if w.endswith(("at", "bl", "iz")):
                w = w + "e"
            elif _ends_double_consonant(w) and w[-1] not in ("l", "s", "z"):
                w = w[:-1]
            elif _measure(w) == 1 and _cvc(w):
                w = w + "e"

    return w


def tokenize(text: str):
    text = (text or "").lower()
    tokens = _TOKEN_RE.findall(text)
    return [_stem(t) for t in tokens if t not in _STOPWORDS and len(t) > 1]


class BM25:
    """Minimal BM25Okapi over a fixed corpus of tokenized documents."""

    def __init__(self, tokenized_docs, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = tokenized_docs
        self.doc_len = [len(d) for d in tokenized_docs]
        self.avgdl = sum(self.doc_len) / max(1, len(tokenized_docs))
        self.doc_freqs = []
        df = Counter()
        for doc in tokenized_docs:
            freqs = Counter(doc)
            self.doc_freqs.append(freqs)
            for term in freqs:
                df[term] += 1
        n_docs = len(tokenized_docs)
        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))

    def score(self, query_tokens):
        scores = [0.0] * len(self.docs)
        for term in query_tokens:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, freqs in enumerate(self.doc_freqs):
                f = freqs.get(term, 0)
                if f == 0:
                    continue
                dl = self.doc_len[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores

    def top_k(self, query_tokens, k=25):
        scores = self.score(query_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, scores[i]) for i in ranked[:k] if scores[i] > 0]


def reciprocal_rank_fusion(ranked_lists, k_const: int = 60):
    """ranked_lists: list of lists of doc indices, best-first.
    Returns dict {doc_index: fused_score}."""
    fused = defaultdict(float)
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            fused[idx] += 1.0 / (k_const + rank + 1)
    return fused


# --------------------------------------------------------------------------
# Alias table for common abbreviations that show up in real conversations
# (OPQ, GSA, G+, DSI, ...). Exact/abbreviation mentions should always
# resolve to the right catalog item regardless of embedding rank, since
# missing a test the user *named* is a straight hallucination/recall miss.
# --------------------------------------------------------------------------

ALIAS_HINTS = {
    "opq32r": "occupational personality questionnaire opq32r",
    "opq": "occupational personality questionnaire opq32r",
    "gsa": "global skills assessment",
    "g+": "shl verify interactive g+",
    "verify g+": "shl verify interactive g+",
    "verify interactive": "shl verify interactive g+",
    "dsi": "dependability and safety instrument",
    "ucf": "opq universal competency report",
    "mq": "motivation questionnaire",
    "svar": "svar spoken english",
}


def extract_alias_hits(text: str):
    """Return the set of canonical-name substrings hinted at by abbreviations
    present in `text` (lowercased)."""
    lower = (text or "").lower()
    hits = set()
    for alias, canonical_substr in ALIAS_HINTS.items():
        pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
        if re.search(pattern, lower):
            hits.add(canonical_substr)
    return hits
