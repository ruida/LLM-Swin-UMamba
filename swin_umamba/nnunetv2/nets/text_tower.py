import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from .tokenizer import MyTokenizer


class Text_Tower(nn.Module):
    def __init__(self, biolord_checkpoint: str = None):
        super().__init__()

        self.biolord = AutoModel.from_pretrained(biolord_checkpoint)
        self.tokenizer = MyTokenizer(biolord_checkpoint, 256)

        # For weighted pooling (learned scalar score per token)
        hidden_size = self.biolord.config.hidden_size
        self.weight_proj = nn.Linear(hidden_size, 1)

    # ---------- pooling functions ----------

    def mean_pooling(self, model_output, attention_mask):
        """
        Simple masked mean pooling over tokens.
        """
        token_embeddings = model_output[0]  # (batch, seq, hidden)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask  # (batch, hidden)

    def max_pooling(self, model_output, attention_mask):
        """
        Masked max pooling over tokens.
        """
        token_embeddings = model_output[0]  # (batch, seq, hidden)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Use dtype-safe "negative infinity" for the current dtype (works with fp16, fp32, etc.)
        neg_inf = torch.finfo(token_embeddings.dtype).min
        masked_embeddings = token_embeddings.masked_fill(input_mask_expanded == 0, neg_inf)

        pooled = torch.max(masked_embeddings, dim=1).values  # (batch, hidden)
        return pooled

    def weighted_pooling(self, model_output, attention_mask):
        """
        Learned weighted average over tokens.
        Each token gets a scalar score from a linear layer, then softmax.
        """
        token_embeddings = model_output[0]  # (batch, seq, hidden)

        # (batch, seq, 1) -> (batch, seq)
        scores = self.weight_proj(token_embeddings).squeeze(-1)

        # Mask out padding positions before softmax, using dtype-safe min value
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(attention_mask == 0, neg_inf)

        # (batch, seq) -> (batch, seq, 1)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)

        # Weighted sum of token embeddings
        pooled = torch.sum(token_embeddings * weights, dim=1)  # (batch, hidden)
        return pooled



    # ---------- forward ----------

    def forward(self, text, pooling: str = "mean"):
        """
        pooling: "mean" | "max" | "weighted"
        """
        text = self.tokenizer.tokenize(text)  # (n, max_l)
        text["input_ids"] = text["input_ids"].to(device=torch.cuda.current_device())
        text["attention_mask"] = text["attention_mask"].to(device=torch.cuda.current_device())

        output = self.biolord(**text)

        if pooling == "mean":
            pooler_output = self.mean_pooling(output, text["attention_mask"])
        elif pooling == "max":
            pooler_output = self.max_pooling(output, text["attention_mask"])
        elif pooling == "weighted":
            pooler_output = self.weighted_pooling(output, text["attention_mask"])
        else:
            raise ValueError(f"Unknown pooling type: {pooling}")

        return pooler_output

