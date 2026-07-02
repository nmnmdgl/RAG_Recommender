"""
Rebuilds the retrieval indices used by main.py.

Changes vs. the original build_index.py (see APPROACH.md for rationale):
  1. Stores FULL metadata per item (test_type, duration, languages,
     job_levels, entity_id) instead of just name/url/keys/description, so
     the LLM has real grounding at generation time instead of guessing.
  2. test_type is derived deterministically from `keys` via
     catalog_utils.derive_test_type (validated against the live SHL site),
     instead of being invented by the LLM per the old prompt's incomplete
     4-letter matrix.
  3. Embeddings are L2-normalized and the FAISS index is IndexFlatIP
     (inner product == cosine similarity on unit vectors). MiniLM is
     trained/evaluated with cosine similarity; the old code used raw
     un-normalized vectors in an L2 index, which is a mismatched metric.
  4. Persists a parallel BM25 lexical corpus (tokenized documents) so
     retrieval.py can hybrid-search dense + lexical, not dense-only.

Run: python build_index.py
Requires: sentence-transformers, faiss-cpu (see requirements.txt) and
network access to download the MiniLM weights the first time.
"""

import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import catalog_utils as cu

CATALOG_PATH = "shl_product_catalog.json"
FAISS_INDEX_PATH = "shl_vector_index.faiss"
METADATA_PATH = "shl_metadata_store.pkl"
BM25_CORPUS_PATH = "shl_bm25_corpus.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def ingest_catalog():
    print("🚀 Starting catalog ingestion...")

    try:
        raw_data = cu.load_raw_catalog(CATALOG_PATH)
    except FileNotFoundError:
        print(f"❌ Error: '{CATALOG_PATH}' not found in this directory.")
        return
    except Exception as e:
        print(f"❌ JSON parsing failed: {e}")
        return

    records = []
    documents = []
    for item in raw_data:
        name = (item.get("name") or "").strip()
        desc = (item.get("description") or "").strip()
        if not name or not desc:
            continue
        rec = cu.build_record(item)
        records.append(rec)
        documents.append(cu.build_document_text(rec))

    print(f"✅ Processed {len(records)} Individual Test Solutions.")

    print("🧠 Generating normalized embeddings (cosine metric)...")
    encoder = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = encoder.encode(
        documents, show_progress_bar=True, normalize_embeddings=True
    )
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # inner product on unit vectors = cosine
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(records, f)

    print("🔤 Building BM25 lexical corpus...")
    tokenized_docs = [cu.tokenize(d) for d in documents]
    with open(BM25_CORPUS_PATH, "wb") as f:
        pickle.dump(tokenized_docs, f)

    print("💾 Wrote FAISS index, metadata store, and BM25 corpus to disk.")


if __name__ == "__main__":
    ingest_catalog()
