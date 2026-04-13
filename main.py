"""
AralMate NLP Microservice
Tagalog linguistic analysis using calamanCy (spaCy-based)

Endpoints:
  GET  /health        — health check, confirms model is loaded
  POST /filter-words  — filter a word list by POS and NER
  POST /analyze       — full linguistic analysis of a text block
"""

import os
import calamancy
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# ── App setup ────────────────────────────────────────────────────

app = FastAPI(title="AralMate NLP Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-AralMate-NLP-Secret"],
)

# ── Auth ─────────────────────────────────────────────────────────

NLP_SECRET = os.environ.get("NLP_SERVICE_SECRET", "")

def verify_secret(request: Request):
    secret = request.headers.get("X-AralMate-NLP-Secret", "")
    if not NLP_SECRET or secret != NLP_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ── Model loading ─────────────────────────────────────────────────
# Load on startup — keeps model in memory for fast inference
# tl_calamancy_md: ~50MB, good accuracy, fits on free tier

print("Loading calamanCy model tl_calamancy_md...")
nlp = calamancy.load("tl_calamancy_md-0.1.0")
print("Model loaded successfully.")

# ── POS rules ─────────────────────────────────────────────────────
# POS tags to KEEP as vocabulary words (content words)
KEEP_POS = {"NOUN", "VERB", "ADJ", "ADV", "PRON"}

# POS tags to EXCLUDE (function words, punctuation)
EXCLUDE_POS = {"DET", "ADP", "PART", "PUNCT", "CONJ", "SCONJ", "AUX", "INTJ", "X", "SYM", "NUM"}

# NER labels to EXCLUDE (named entities — people, places, organizations)
EXCLUDE_NER = {"PER", "ORG", "LOC"}

# Minimum word length to be considered vocabulary
MIN_WORD_LENGTH = 3

# ── Request / Response models ─────────────────────────────────────

class FilterWordsRequest(BaseModel):
    words: List[str]

class FilterWordsResponse(BaseModel):
    filtered: List[str]
    removed: List[dict]

class AnalyzeRequest(BaseModel):
    text: str

class TokenInfo(BaseModel):
    word: str
    pos: str
    is_entity: bool
    entity_type: Optional[str] = None
    is_content_word: bool

class AnalyzeResponse(BaseModel):
    tokens: List[TokenInfo]
    entities: List[dict]
    content_words: List[str]
    language: str

# ── Helper ────────────────────────────────────────────────────────

def is_content_word(token, entity_set: set) -> bool:
    """Determine if a token is a valid vocabulary word."""
    word = token.text.strip()

    # Too short
    if len(word) < MIN_WORD_LENGTH:
        return False

    # Named entity — exclude
    if word in entity_set:
        return False

    # Wrong POS — exclude
    if token.pos_ in EXCLUDE_POS:
        return False

    # Must be a content POS
    if token.pos_ not in KEEP_POS:
        return False

    return True

# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — confirms service is running and model is loaded."""
    return {"status": "ok", "model": "tl_calamancy_md-0.1.0", "service": "AralMate NLP"}


@app.post("/filter-words", response_model=FilterWordsResponse)
async def filter_words(body: FilterWordsRequest, request: Request):
    """
    Filter a list of words by POS and NER.

    Keeps: NOUN, VERB, ADJ, ADV, PRON
    Removes: DET, ADP, PART, PUNCT, named entities (PER, ORG, LOC), short words

    Input:  { "words": ["alkansiya", "Zarah", "ang", "parirala", "Impong Sela"] }
    Output: { "filtered": ["alkansiya", "parirala"], "removed": [...] }
    """
    verify_secret(request)

    filtered = []
    removed = []

    for word in body.words:
        word = word.strip()
        if not word:
            continue

        # Run calamanCy on the word in isolation
        doc = nlp(word)

        # Check NER on the full word (catches multi-word proper nouns)
        entities = {ent.text for ent in doc.ents if ent.label_ in EXCLUDE_NER}

        if word in entities:
            removed.append({"word": word, "reason": f"named_entity"})
            continue

        # For single-token words, check POS
        if len(doc) == 1:
            token = doc[0]
            if token.pos_ in EXCLUDE_POS or token.pos_ == "PROPN":
                removed.append({"word": word, "reason": f"pos_{token.pos_}"})
                continue
            if len(word) < MIN_WORD_LENGTH:
                removed.append({"word": word, "reason": "too_short"})
                continue
            filtered.append(word)
        else:
            # Multi-word term — check if any token is a proper noun or entity
            has_propn = any(t.pos_ == "PROPN" for t in doc)
            if has_propn:
                removed.append({"word": word, "reason": "contains_proper_noun"})
                continue
            filtered.append(word)

    return FilterWordsResponse(filtered=filtered, removed=removed)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(body: AnalyzeRequest, request: Request):
    """
    Full linguistic analysis of a Filipino/Tagalog text block.

    Returns per-token POS tags, NER entities, and a filtered list
    of content words suitable for vocabulary/flashcard use.

    Input:  { "text": "Ang alkansiya ay mabigat. Sina Zarah at Impong Sela ay nag-uusap." }
    Output: { tokens: [...], entities: [...], content_words: ["alkansiya", "mabigat", "nag-uusap"] }
    """
    verify_secret(request)

    text = body.text.strip()
    if not text:
        return AnalyzeResponse(tokens=[], entities=[], content_words=[], language="tl")

    doc = nlp(text)

    # Extract named entities to exclude
    entity_words = set()
    entities_out = []
    for ent in doc.ents:
        if ent.label_ in EXCLUDE_NER:
            # Add all individual words in a multi-word entity
            for word in ent.text.split():
                entity_words.add(word)
            entities_out.append({"text": ent.text, "label": ent.label_})

    # Build token list
    tokens_out = []
    content_words = []
    seen = set()

    for token in doc:
        word = token.text.strip()
        if not word or token.is_space:
            continue

        in_entity = word in entity_words
        content = is_content_word(token, entity_words)

        tokens_out.append(TokenInfo(
            word=word,
            pos=token.pos_,
            is_entity=in_entity,
            entity_type=next(
                (e["label"] for e in entities_out if word in e["text"]), None
            ),
            is_content_word=content
        ))

        # Add to content words (deduplicated, lowercase)
        if content and word.lower() not in seen:
            content_words.append(word)
            seen.add(word.lower())

    return AnalyzeResponse(
        tokens=tokens_out,
        entities=entities_out,
        content_words=content_words,
        language="tl"
    )
