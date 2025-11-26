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
        """Improved chunking: split by sections (##) OR paragraphs (\\n\\n) OR bullet lists"""
        files = glob.glob(os.path.join(path, "*.md"))
        for f in files:
            fname = os.path.basename(f).replace(".md", "")
            with open(f, 'r') as file:
                content = file.read()
                
                # Strategy 1: Split by markdown headers (## )
                if '\n## ' in content:
                    parts = content.split('\n## ')
                    # Keep the title with the first part
                    for i, part in enumerate(parts):
                        if i > 0:
                            part = '## ' + part  # Re-add header marker
                        if part.strip():
                            self.chunks.append(part.strip())
                            self.doc_ids.append(f"{fname}::chunk{i}")
                
                # Strategy 2: Split by paragraphs (double newline)
                elif '\n\n' in content:
                    parts = content.split('\n\n')
                    for i, part in enumerate(parts):
                        if part.strip():
                            self.chunks.append(part.strip())
                            self.doc_ids.append(f"{fname}::chunk{i}")
                
                # Strategy 3: For bullet lists (like product_policy.md), split by lines
                else:
                    lines = content.split('\n')
                    # Group header + all bullet points as separate chunks
                    header = ""
                    for i, line in enumerate(lines):
                        if line.strip():
                            if line.startswith('#'):
                                header = line.strip()
                            elif line.startswith('-'):
                                # Each bullet becomes a chunk WITH the header for context
                                chunk_text = f"{header}\n{line.strip()}"
                                self.chunks.append(chunk_text)
                                self.doc_ids.append(f"{fname}::chunk{i}")
                            else:
                                # Regular paragraph
                                self.chunks.append(line.strip())
                                self.doc_ids.append(f"{fname}::chunk{i}")

    def search(self, query, k=3):
        """Returns list of (content, doc_id, score) for top-k matches with query expansion."""
        # Query expansion for better recall
        expanded_query = query.lower()
        
        # Expand key retail terms
        if 'return' in expanded_query or 'policy' in expanded_query:
            expanded_query += ' returns policy days window'
        if 'beverage' in expanded_query:
            expanded_query += ' beverages drinks unopened opened'
        if 'aov' in expanded_query or 'average order value' in expanded_query:
            expanded_query += ' aov order value revenue'
        if 'kpi' in expanded_query:
            expanded_query += ' kpi definition formula metric'
        if 'summer' in expanded_query or 'winter' in expanded_query:
            expanded_query += ' marketing calendar campaign dates'
        if 'category' in expanded_query or 'categories' in expanded_query:
            expanded_query += ' category categories product beverages dairy confections'
        
        tokenized_query = expanded_query.split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for i in top_n:
            if scores[i] > 0:  # Only return relevant
                results.append((self.chunks[i], self.doc_ids[i], float(scores[i])))
        return results