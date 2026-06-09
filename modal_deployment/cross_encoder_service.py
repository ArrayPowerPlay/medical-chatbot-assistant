"""
Deploy ncbi/MedCPT-Cross-Encoder to Modal (Cloud GPU)
Takes query and a list of passages, returns reranking scores
"""

import modal
from typing import List

# Initialize Modal App
app = modal.App("medcpt-cross-encoder-v1")

# Define container image with Pytorch and Transformers
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers"
    )
)

@app.cls(gpu="T4", image=image)   # Request Modal to allocate an NVIDIA T4 GPU for this module
class CrossEncoderModel:
    """Class contains model which has been installed in GPU memory using Modal"""
    @modal.enter()                # Load the model into GPU memory as soon as the container is initialized
    def load_model(self):
        """Run once when container starts to load model into GPU"""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        print("Loading MedCPT-Cross-Encoder...")
        self.model_name = "ncbi/MedCPT-Cross-Encoder"
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully on", self.device)

    @modal.method()               # Convert the 'rerank' method into an API, runs on Modal
    def rerank(self, query: str, passages: List[str]) -> List[float]:
        """
        Inputs a query and a list of passages for the model to rank each passage.

        Args:
            query: User's question (rewritten query)
            passages: List of text chunks

        Returns:
            List of points (float)
        """
        import torch

        if not passages:
            return []
        
        pairs = [[query, p] for p in passages]

        features = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**features)
            scores = outputs.logits.squeeze(-1).tolist()

        if isinstance(scores, float):
            scores = [scores]

        return scores
    

@app.local_entrypoint()
def test_reranker():
    query = "What are the treatments for diabetes?"
    passages = [
        "Metformin is a first-line medication for the treatment of type 2 diabetes.",
        "Aspirin is commonly used for pain relief and reducing fever.",
        "Drug Metformin TREATS Disease Type 2 Diabetes" # Lấy từ KG
    ]
    
    # Khởi tạo và gọi hàm
    model = CrossEncoderModel()
    scores = model.rerank.remote(query, passages)
    
    print(f"Query: {query}")
    for passage, score in zip(passages, scores):
        print(f"Score: {score:.4f} | Passage: {passage}")


# if __name__ == "__main__":
    # test_reranker()