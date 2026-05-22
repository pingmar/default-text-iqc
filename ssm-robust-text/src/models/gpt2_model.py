from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel


def load_gpt2(model_name: str = "gpt2", device: str = "cuda") -> GPT2LMHeadModel:
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
    return model.to(device).eval()


def load_gpt2_tokenizer(model_name: str = "gpt2"):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    return tok


class GPT2PPLWrapper(nn.Module):
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

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 50, device: str = "cuda") -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        out = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
