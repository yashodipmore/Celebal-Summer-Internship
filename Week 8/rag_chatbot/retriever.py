import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Retriever:
    def __init__(self, csv_path, chunk_size=100):
        self.df = pd.read_csv(csv_path)
        self.texts = self._chunk_texts(chunk_size)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def _chunk_texts(self, chunk_size):
        # Combine all text columns, chunk into pieces
        all_text = ' '.join(str(x) for x in self.df.values.flatten())
        return [all_text[i:i+chunk_size] for i in range(0, len(all_text), chunk_size)]

    def retrieve(self, query, top_k=3):
        query_emb = self.model.encode([query])[0]
        sims = cosine_similarity([query_emb], self.embeddings)[0]
        top_indices = np.argsort(sims)[-top_k:][::-1]
        return [self.texts[i] for i in top_indices]
