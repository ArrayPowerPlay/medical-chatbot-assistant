import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Union
from config.logging_config import logger


class MedCPTEmbedder:
    """MedCPT Dual-Encoder wrapper.
    Supports Article-Encoder (indexing) and Query-Encoder (retrieval)
    """
    def __init__(self, mode: str = "article"):
        """
        Args:
            mode: "article" to load Article-Encoder, "query" to load Query-Encoder.
        """
        from config.settings import settings
        # Choose model based on 'mode'
        self.model_name = settings.EMBEDDING_MODEL if mode == 'article' else settings.QUERY_MODEL
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        logger.info(f"Initializing MedCPT model {self.model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    @torch.no_grad()    # Turn off gradient checking
    def embed_texts(self, texts: Union[str, List[str]], batch_size: int = 256) -> np.ndarray:
        """Embed all chunks into L2-normalized vectors."""
        if isinstance(texts, str):
            texts = [texts]

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
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings) 

    def close(self):
        """Free GPU memory"""
        if self.device.type == 'cuda':
            del self.model
            torch.cuda.empty_cache()
        
