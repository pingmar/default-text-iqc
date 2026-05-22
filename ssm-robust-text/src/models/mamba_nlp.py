from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MAMBA_MODEL = "state-spaces/mamba-370m-hf"


def load_mamba(model_name: str = MAMBA_MODEL, device: str = "cuda"):
    dtype = torch.float16 if device == "cuda" else torch.float32
    return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()


def load_mamba_tokenizer(model_name: str = MAMBA_MODEL):
    tok = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


class MambaNLPWrapper(nn.Module):
    # Thin wrapper exposing .model for eval_ppl_vs_length
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def perplexity(self, text: str, device: str = "cuda", max_length: int = 1024) -> float:
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        return self.model(input_ids, labels=input_ids).loss.exp().item()
