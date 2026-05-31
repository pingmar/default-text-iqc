"""
Three reference corpus loaders for feature extraction.

Each loader returns (DataLoader, list[str]) - a DataLoader of tokenised
batches and the corresponding raw text strings.  The raw strings are needed
to annotate top-k activating examples in the FeatureDictionary.

Tokenizer convention (identical to ssm_prefix_tuning.data):
    - padding_side = "left"   → last real token always at index -1
    - pad_token   = eos_token → Mamba checkpoints have no dedicated pad token
"""
from __future__ import annotations

import json
import os
from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_sentiment_corpus(
    model_name: str,
    max_length: int = 128,
    batch_size: int = 32,
    max_samples: Optional[int] = 1000,
    cache_dir: Optional[str] = None,
) -> tuple[DataLoader, list[str]]:
    """SST-2 validation split - binary sentiment (positive/negative reviews)."""
    tokenizer = get_tokenizer(model_name)
    raw = load_dataset("nyu-mll/glue", "sst2", cache_dir=cache_dir)
    texts = raw["validation"]["sentence"]
    if max_samples is not None:
        texts = texts[:max_samples]
    return _build_loader(texts, tokenizer, max_length, batch_size), list(texts)


def load_ner_corpus(
    model_name: str,
    max_length: int = 128,
    batch_size: int = 32,
    max_samples: Optional[int] = 1000,
    cache_dir: Optional[str] = None,
) -> tuple[DataLoader, list[str]]:
    """
    WikiText-2 train split - encyclopaedic prose with high named-entity
    density, no licence restrictions.  Filters empty lines before sampling.
    """
    tokenizer = get_tokenizer(model_name)
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=cache_dir)
    texts = [t.strip() for t in raw["train"]["text"] if t.strip()]
    if max_samples is not None:
        texts = texts[:max_samples]
    return _build_loader(texts, tokenizer, max_length, batch_size), list(texts)


def load_syntactic_corpus(
    model_name: str,
    max_length: int = 128,
    batch_size: int = 32,
    max_samples: Optional[int] = 1000,
    cache_dir: Optional[str] = None,
) -> tuple[DataLoader, list[str]]:
    """
    Syntactic diversity corpus. Uses Penn Treebank (ptb_text_only).
    """
    tokenizer = get_tokenizer(model_name)
    texts = _load_syntactic_texts(max_samples, cache_dir)
    return _build_loader(texts, tokenizer, max_length, batch_size), list(texts)


def _load_syntactic_texts(
    max_samples: Optional[int], cache_dir: Optional[str]
) -> list[str]:
    raw = load_dataset("ptb_text_only", "penn_treebank", cache_dir=cache_dir)
    texts = [s.strip() for s in raw["train"]["sentence"] if s.strip()]
    if max_samples is not None:
        texts = texts[:max_samples]
    return texts


def _build_loader(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    batch_size: int,
) -> DataLoader:
    dataset = _TokenisedDataset(texts, tokenizer, max_length)
    collator = _Collator(tokenizer.pad_token_id)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)


class _TokenisedDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        encoded = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self._input_ids = encoded["input_ids"]
        self._attention_mask = encoded["attention_mask"]

    def __len__(self) -> int:
        return len(self._input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self._input_ids[idx],
            "attention_mask": self._attention_mask[idx],
        }


class _Collator:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        }
