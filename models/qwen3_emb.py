# benchmark/models/qwen3_emb.py

import torch

from transformers import AutoTokenizer, AutoModel
from ..core.base_model import BaseModel

from tqdm import tqdm


class Qwen3Embedding(BaseModel):

    MAX_LENGTH = 32768
    MODEL_MAP = {
        "600m": ("Qwen/Qwen3-Embedding-0.6B",  "qwen3_emb_600"),
        "8b": ("Qwen/Qwen3-Embedding-8B", "qwen3_emb_8b")
    }

    def __init__(self, variant: str = "base", device: str = None):
        if variant not in self.MODEL_MAP:
            raise ValueError(f"Unsupported variant: {variant}")

        self.MODEL_ID, self.MODEL_NAME = self.MODEL_MAP[variant]

        super().__init__(self.MODEL_NAME, device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID, padding_side='left')
        self.model     = AutoModel.from_pretrained(self.MODEL_ID).to(self.device)
        self.model.eval()


    def encode(self, code: str, is_query: bool = False) -> torch.Tensor:
        return self.encode_batch([code])[0]

    def encode_batch(self, codes: list[str], batch_size: int = 16, is_query: bool = False) -> list[torch.Tensor]:
        embeddings = []

        for i in range(0, len(codes), batch_size):
            batch  = codes[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.MAX_LENGTH
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model(**inputs)

            batch_embeddings = self.last_token_pool(
                outputs.last_hidden_state, inputs["attention_mask"]
            )

            embeddings.append(batch_embeddings.cpu())         

        embedding_matrix = torch.cat(embeddings, dim=0)
        return embedding_matrix

    def last_token_pool(self, last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]