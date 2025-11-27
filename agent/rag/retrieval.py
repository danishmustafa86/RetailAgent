import os
import glob
from rank_bm25 import BM25Okapi

class LocalRetriever:
    def __init__(self, docs_path="docs/"):
        self.chunks = []
        self.doc_ids = []
        self._load_docs(docs_path)

        # Tokenize for BM25
        tokenized_corpus = [doc.lower().split() for doc in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _load_docs(self, path):
        """Simple chunking by paragraph (double newline)"""
        files = glob.glob(os.path.join(path, "*.md"))
        for f in files:
            fname = os.path.basename(f).replace(".md", "")
            with open(f, 'r') as file:
                content = file.read()
                # Split by sections or paragraphs
                parts = content.split('\n\n')
                for i, part in enumerate(parts):
                    if part.strip():
                        self.chunks.append(part.strip())
                        self.doc_ids.append(f"{fname}::chunk{i}")

    def search(self, query, k=3):
        """Returns list of (content, doc_id, score) for top-k matches."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        # Get top k indices
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for i in top_n:
            if scores[i] > 0:  # Only return relevant
                results.append((self.chunks[i], self.doc_ids[i], float(scores[i])))
        return results