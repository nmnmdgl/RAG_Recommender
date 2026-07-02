"""
Offline retrieval-recall evaluator.

This measures the retrieval layer in isolation from the LLM, using the
final recommendation table in each sample conversation trace as the gold
shortlist (the last turn's table is the trace's "converged" answer).

It compares two retrieval strategies over the SAME lexical (BM25) backend:
  OLD  = single query: naive concatenation of all user turns, fixed k=15,
         no alias forcing. (This mirrors what main.py used to do, minus
         the embedding model swap — the embedding model itself needs
         network access to download, which this evaluation environment
         doesn't have, so we isolate the *strategy* change instead.)
  NEW  = retrieval.HybridRetriever's query decomposition + alias forcing,
         same BM25 backend, scaled k.

Recall@K = (# gold items present in the candidate set) / (# gold items)

Run: python retrieval_eval.py sample_conversations.zip
"""

import sys
import re
import zipfile

import catalog_utils as cu
from retrieval import HybridRetriever


def parse_user_turns(md_content: str):
    turns = []
    blocks = md_content.split("**User**")
    for block in blocks[1:]:
        lines = block.split("\n")
        text = []
        for line in lines:
            if line.strip().startswith(">"):
                text.append(line.replace(">", "").strip())
            elif "**Agent**" in line:
                break
        if text:
            turns.append(" ".join(text))
    return turns


def parse_full_conversation(md_content: str):
    """Parse ordered (role, text) turns for BOTH roles, prose only (tables
    and metadata footers skipped). This matters because production's
    HybridRetriever scans assistant turns too (that's what the consistency/
    anchoring fix in retrieval.py relies on) -- an offline eval that only
    ever builds messages from user turns systematically understates
    anything the retriever can only find because the *assistant* named it
    first (e.g. introducing 'SVAR' in a reply before the user ever
    mentions it). This does not change what persists across real /chat
    calls: only the natural-language reply text is checked (matching the
    fact that the structured `recommendations` table never survives to
    the next stateless call), tables/footer lines are skipped."""
    turns = []
    # Split on either speaker marker while keeping track of who's talking.
    parts = re.split(r"\*\*(User|Agent)\*\*", md_content)
    # parts alternates: [preamble, 'User', block, 'Agent', block, 'User', block, ...]
    for i in range(1, len(parts) - 1, 2):
        speaker = parts[i]
        block = parts[i + 1]
        role = "user" if speaker == "User" else "assistant"
        text_lines = []
        for line in block.split("\n"):
            stripped = line.strip()
            if role == "user":
                if stripped.startswith(">"):
                    text_lines.append(stripped.lstrip(">").strip())
            else:
                # Assistant prose is the leading paragraph before any
                # markdown table / "RECS:" style footer begins.
                if stripped.startswith("|") or stripped.startswith("RECS:") or stripped.startswith("#"):
                    break
                if stripped.startswith(">"):
                    text_lines.append(stripped.lstrip(">").strip())
                elif stripped and not stripped.startswith("*"):
                    text_lines.append(stripped)
        text = " ".join(t for t in text_lines if t)
        if text:
            turns.append((role, text))
    return turns


def parse_gold_shortlist(md_content: str):
    """Take the LAST markdown table in the trace as the converged gold
    shortlist and extract test names from its 'Name' column."""
    tables = re.findall(r"\|\s*#\s*\|.*?\n(?:\|.*\n)+", md_content)
    if not tables:
        return []
    last_table = tables[-1]
    rows = [r for r in last_table.strip().split("\n") if r.startswith("|")][2:]  # skip header+separator
    names = []
    for row in rows:
        cols = [c.strip() for c in row.strip("|").split("|")]
        if len(cols) >= 2:
            names.append(cols[1])
    return names


def naive_old_retrieve(bm25, records, all_user_turns, k=15):
    """Old strategy: one blob query, plain BM25 top-k, no alias forcing."""
    full_query = " ".join(all_user_turns)
    tokens = cu.tokenize(full_query)
    hits = bm25.top_k(tokens, k=k)
    return [records[i]["name"] for i, _ in hits]


def recall_at(gold_names, candidate_names):
    if not gold_names:
        return None
    gold_set = {g.lower() for g in gold_names}
    cand_set = {c.lower() for c in candidate_names}
    hit = sum(1 for g in gold_set if g in cand_set)
    return hit / len(gold_set)


def main(zip_path):
    raw = cu.load_raw_catalog("shl_product_catalog.json")
    records = [cu.build_record(it) for it in raw if it.get("name") and it.get("description")]
    docs = [cu.build_document_text(r) for r in records]
    tokenized = [cu.tokenize(d) for d in docs]
    bm25 = cu.BM25(tokenized)

    # Dense search stubbed out (no network for the embedding model here) —
    # this isolates the effect of the retrieval *strategy* on the lexical
    # backend both approaches share.
    class StubIndex:
        def search(self, vec, k):
            return [[0.0] * k], [[-1] * k]

    class StubModel:
        def encode(self, x, normalize_embeddings=True):
            return [[0.0]]

    new_retriever = HybridRetriever(records, docs, StubModel(), StubIndex(), bm25)

    old_recalls, new_recalls = [], []
    with zipfile.ZipFile(zip_path) as z:
        md_files = sorted(n for n in z.namelist() if n.endswith(".md"))
        for name in md_files:
            content = z.read(name).decode("utf-8")
            user_turns = parse_user_turns(content)
            gold = parse_gold_shortlist(content)
            if not gold or not user_turns:
                continue

            old_names = naive_old_retrieve(bm25, records, user_turns, k=15)
            r_old = recall_at(gold, old_names)

            # Fair input for "what would the retriever hand the LLM right
            # before it generates the final answer": everything up to and
            # including the last user turn, but NOT the final assistant
            # reply itself -- that reply is what produces the gold table,
            # so including its prose would leak the answer into the query.
            full_turns = parse_full_conversation(content)
            if full_turns and full_turns[-1][0] == "assistant":
                full_turns = full_turns[:-1]
            messages = [{"role": role, "content": text} for role, text in full_turns]
            new_candidates = new_retriever.retrieve(messages)
            new_names = [c["name"] for c in new_candidates]
            r_new = recall_at(gold, new_names)

            old_recalls.append(r_old)
            new_recalls.append(r_new)
            print(f"{name:45s} gold={len(gold):2d}  OLD recall={r_old:.2f}  NEW recall={r_new:.2f}")

    if old_recalls:
        print("-" * 90)
        print(f"Mean recall  OLD (naive single-query, k=15): {sum(old_recalls)/len(old_recalls):.3f}")
        print(f"Mean recall  NEW (hybrid multi-facet+alias):  {sum(new_recalls)/len(new_recalls):.3f}")


if __name__ == "__main__":
    zip_path = sys.argv[1] if len(sys.argv) > 1 else "sample_conversations.zip"
    main(zip_path)
