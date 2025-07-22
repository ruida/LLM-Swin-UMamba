import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel


class LLaVATextTower(nn.Module):
    def __init__(self, model_name='liuhaotian/llava-llama-2-7b-hf', max_length=256):
        """
        LLaVA-based text encoder for use in multimodal segmentation networks.
        Produces a fixed-length embedding from a list of text inputs.
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length

    def mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling over the token embeddings.
        """
        token_embeddings = model_output[0]  # shape: (B, T, C)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
            input_mask_expanded.sum(dim=1), min=1e-9
        )

    def forward(self, text_list):
        """
        Args:
            text_list (List[str]): A batch of text strings.

        Returns:
            torch.Tensor: A tensor of shape (B, C) containing pooled text features.
        """
        assert isinstance(text_list, list) and isinstance(text_list[0], str), \
            "Input to LLaVATextTower must be a list of strings."

        # Tokenize and move to model device
        encoding = self.tokenizer(
            text_list,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        device = next(self.model.parameters()).device
        encoding = {k: v.to(device) for k, v in encoding.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**encoding)

        return self.mean_pooling(outputs, encoding['attention_mask'])
