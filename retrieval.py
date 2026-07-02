"""
Retrieval layer for the SHL assessment recommender.

Problems this fixes relative to the original main.py (see APPROACH.md):
1. Single blind query. The original code concatenated every user turn into
   one blob and ran one embedding search. A multi-skill JD ("Java, Spring,
   REST, Angular, SQL, AWS, Docker") gets averaged into a mushy vector that
   under-retrieves the less-dominant skills. We decompose into per-facet
   sub-queries and fuse results.
2. Pure dense search. Embeddings are bad at exact/abbreviated product names
   ("OPQ", "GSA", "G+"). We add a lexical BM25 index and fuse it with dense
   search via Reciprocal Rank Fusion, and on top of that we force-include
   any catalog item whose name (or a known alias) is explicitly mentioned
   in the conversation — critical for "compare X and Y" and "drop the OPQ"
   turns where the named item must be in context regardless of its
   embedding rank.
3. No recency weighting. A correction like "actually, drop OPQ and add AWS"
   should dominate retrieval for that turn, not get diluted by five turns
   of earlier context. We run the full-history query AND a latest-turn-only
   query and fuse both.
4. Fixed k=15 regardless of query complexity. Batteries covering many named
   skills need a wider net. We scale k with the number of decomposed facets
   (bounded so latency/tokens stay sane).
"""

import re
from catalog_utils import tokenize, reciprocal_rank_fusion, extract_alias_hits, core_name

MIN_K = 18
MAX_K = 40
PER_FACET_DENSE_K = 10
PER_FACET_LEXICAL_K = 10


def _split_facets(text: str):
    """Cheap heuristic JD/requirement decomposition: split on strong
    delimiters and keep phrases that look like discrete skills/requirements.
    Not a full NLP pipeline, but far better than treating the whole blob as
    one query when someone pastes a multi-skill job description."""
    if not text:
        return []
    # Normalize bullets/newlines to a common delimiter first.
    text = re.sub(r"[\u2022\-]{1}\s+", ", ", text)
    chunks = re.split(r",|;|\n|/|\. |\.$| and (?=[A-Za-z])", text)
    facets = []
    for c in chunks:
        c = c.strip(" .\"'")
        if 2 <= len(c) <= 60 and re.search(r"[a-zA-Z]", c):
            facets.append(c)
    # De-dupe while preserving order, cap to avoid runaway facet counts.
    seen = set()
    deduped = []
    for f in facets:
        key = f.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(f)
    return deduped[:10]


def decompose_query(messages):
    """Build the set of queries to search with, from the stateless
    conversation history."""
    user_turns = [m["content"] for m in messages if m.get("role") == "user"]
    assistant_turns = [m["content"] for m in messages if m.get("role") == "assistant"]
    full_context = " ".join(user_turns)
    # Full conversation text (both roles) is used only for anchoring items
    # already committed to in prior assistant replies — NOT for ranking,
    # since assistant prose would otherwise bias retrieval toward whatever
    # the model already said rather than what the user is actually asking.
    full_conversation_text = " ".join(user_turns + assistant_turns)
    latest_turn = user_turns[-1] if user_turns else ""

    queries = []
    if full_context:
        queries.append(full_context)
    if latest_turn and latest_turn != full_context:
        queries.append(latest_turn)

    # Facet decomposition runs on whichever user turn is longest (usually
    # the pasted JD or the most detail-rich requirement statement).
    richest_turn = max(user_turns, key=len, default="")
    facets = _split_facets(richest_turn) if len(richest_turn) > 40 else []

    return {
        "full_context": full_context,
        "full_conversation_text": full_conversation_text,
        "latest_turn": latest_turn,
        "queries": queries,
        "facets": facets,
    }


class HybridRetriever:
    def __init__(self, records, doc_texts, embedding_model, vector_index, bm25):
        self.records = records
        self.doc_texts = doc_texts
        self.embedding_model = embedding_model
        self.vector_index = vector_index
        self.bm25 = bm25

    def _dense_rank(self, query: str, k: int):
        import numpy as np

        vec = self.embedding_model.encode([query], normalize_embeddings=True)
        vec = np.array(vec).astype("float32")
        _, indices = self.vector_index.search(vec, k)
        return [int(i) for i in indices[0] if 0 <= i < len(self.records)]

    def _lexical_rank(self, query: str, k: int):
        tokens = tokenize(query)
        if not tokens:
            return []
        return [i for i, _ in self.bm25.top_k(tokens, k=k)]

    def retrieve(self, messages, base_k: int = MIN_K):
        decomposed = decompose_query(messages)
        facets = decomposed["facets"]

        # Scale candidate pool width with query complexity: a 7-skill JD
        # needs a wider net than a single "Rust engineer" ask.
        target_k = min(MAX_K, max(base_k, base_k + 2 * len(facets)))

        ranked_lists = []
        facet_top_hits = []
        for q in decomposed["queries"]:
            ranked_lists.append(self._dense_rank(q, PER_FACET_DENSE_K + 5))
            ranked_lists.append(self._lexical_rank(q, PER_FACET_LEXICAL_K + 5))
        for facet in facets:
            d = self._dense_rank(facet, PER_FACET_DENSE_K)
            l = self._lexical_rank(facet, PER_FACET_LEXICAL_K)
            ranked_lists.append(d)
            ranked_lists.append(l)
            # A single-skill facet like "Docker" or "AWS deployment" can get
            # outranked in the *global* fusion by generic documents that
            # weakly match several broader queries at once. Guarantee each
            # named facet still contributes its own best 1-2 hits so a
            # 7-skill JD doesn't silently lose the least-common skill.
            facet_fused = reciprocal_rank_fusion([r for r in (d, l) if r])
            facet_ordered = sorted(facet_fused.keys(), key=lambda i: facet_fused[i], reverse=True)
            facet_top_hits.extend(facet_ordered[:2])

        fused = reciprocal_rank_fusion([r for r in ranked_lists if r])
        ordered = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)

        # Force-include anything explicitly named/aliased in the whole
        # conversation, even if it fell outside the fused ranking, so
        # comparisons and "drop the OPQ" edits always have grounding.
        # Force-include anything explicitly named/aliased anywhere in the
        # conversation -- including the assistant's own prior replies, so
        # an item already committed to in an earlier turn stays anchored
        # in the candidate pool instead of losing a coin-flip to a
        # near-duplicate catalog entry (the catalog has several: e.g.
        # "SHL Verify Interactive G+" vs "Verify - G+" vs "Verify G+ -
        # Candidate Report" are six distinct entries for the same family).
        # Without this, a pure confirmation turn ("keep Verify G+,
        # locking it in") could resolve to a different G+ variant than the
        # one actually offered two turns earlier.
        alias_hits = extract_alias_hits(decomposed["full_conversation_text"])
        forced = []
        for hint in alias_hits:
            for i, rec in enumerate(self.records):
                if hint in rec["name"].lower() and i not in forced:
                    forced.append(i)
        for i in facet_top_hits:
            if i not in forced:
                forced.append(i)

        # Also force-include exact substring name mentions (user typed the
        # full product name rather than an abbreviation, or the assistant
        # already named it in a prior reply).
        lower_ctx = decomposed["full_conversation_text"].lower()
        for i, rec in enumerate(self.records):
            name_lower = rec["name"].lower()
            core_lower = core_name(name_lower)
            matched = (len(name_lower) > 5 and name_lower in lower_ctx) or (
                len(core_lower) > 5 and core_lower in lower_ctx
            )
            if matched and i not in forced:
                forced.append(i)

        final_order = forced + [i for i in ordered if i not in forced]
        final_order = final_order[:target_k]

        return [self.records[i] for i in final_order]
