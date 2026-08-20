# benchmark/models/starencoder.py

import torch
from transformers import AutoTokenizer, AutoModel

from ..core.base_model import BaseModel

from tqdm import tqdm


class StarEncoder(BaseModel):

    MODEL_ID   = "bigcode/starencoder"
    MODEL_NAME = "starencoder"
    MAX_LENGTH = 1024

    PAD_TOKEN  = "<pad>"
    SEP_TOKEN  = "<sep>"
    CLS_TOKEN  = "<cls>"
    MASK_TOKEN = "<mask>"

    def __init__(self, device: str = None):
        super().__init__(self.MODEL_NAME, device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.tokenizer.add_special_tokens({"pad_token":  self.PAD_TOKEN})
        self.tokenizer.add_special_tokens({"sep_token":  self.SEP_TOKEN})
        self.tokenizer.add_special_tokens({"cls_token":  self.CLS_TOKEN})
        self.tokenizer.add_special_tokens({"mask_token": self.MASK_TOKEN})

        self.model = AutoModel.from_pretrained(self.MODEL_ID)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = self.model.to(self.device)
        self.model.eval()

    def encode(self, code: str, is_query: bool = False) -> torch.Tensor:
        return self.encode_batch([code])[0]

    def encode_batch(self, codes: list[str], batch_size: int = 32, is_query: bool = False) -> list[torch.Tensor]:
        embeddings = []

        for i in tqdm(range(0, len(codes), batch_size)):
        #for i in range(0, len(codes), batch_size):
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