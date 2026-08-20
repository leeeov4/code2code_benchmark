# benchmark/models/codet5.py

import torch

from transformers import RobertaTokenizer, T5EncoderModel
from ..core.base_model import BaseModel

from tqdm import tqdm

class CodeT5(BaseModel):

    MAX_LENGTH = 512
    MODEL_MAP = {
        "base":  ("Salesforce/codet5-base",  "codet5_base"),
        "large": ("Salesforce/codet5-large", "codet5_large"),
    }

    def __init__(self, variant: str = "base", device: str = None):
        if variant not in self.MODEL_MAP:
            raise ValueError(f"Unsupported variant: {variant}")

        self.MODEL_ID, self.MODEL_NAME = self.MODEL_MAP[variant]

        super().__init__(self.MODEL_NAME, device)

        self.tokenizer = RobertaTokenizer.from_pretrained(self.MODEL_ID)
        self.model     = T5EncoderModel.from_pretrained(self.MODEL_ID).to(self.device)
        self.model.eval()


    def encode(self, code: str, is_query: bool = False) -> torch.Tensor:
        return self.encode_batch([code])[0]

    def encode_batch(self, codes: list[str], batch_size: int = 32, is_query: bool = False) -> list[torch.Tensor]:
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
                outputs = self.model.encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

            batch_embeddings = self._mean_pool(
                outputs.last_hidden_state, inputs["attention_mask"]
            )

            embeddings.append(batch_embeddings.cpu())             


        embedding_matrix = torch.cat(embeddings, dim=0)
        return embedding_matrix

    def _mean_pool(self, last_hidden_state: torch.Tensor,
                   attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_state.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]