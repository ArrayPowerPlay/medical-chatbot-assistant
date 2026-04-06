import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List


class MedCPTEmbedder:
    def __init__(self, model_name: str | None = None):
        from config.settings import settings
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"Initializing MedCPT model {self.model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    @torch.no_grad()    # Turn off gradient checking
    def embed_texts(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        """Embed all chunks into vectors"""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            inputs = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)   # FAISS requires data to be in ndarray type
        
