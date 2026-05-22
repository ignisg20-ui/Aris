"""Aris tokenizer wrapper.

We deliberately *don't* re-implement BPE training here — building a high-quality
production tokenizer is a multi-day effort that's well covered by
``sentencepiece`` and ``tokenizers``/``tiktoken``. Instead we wrap an existing
backend and add:

* Stable special-token IDs.
* A Claude/LLaMA-3 style chat template (system / user / assistant / tool roles).
* Streaming decoding with safe handling of multi-byte boundaries.

Backends:
* ``sentencepiece`` (default; matches LLaMA family).
* ``tiktoken`` (when ``model_path`` ends in ``.tiktoken``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ChatMessage:
    role: Role
    content: str
    name: str | None = None


@dataclass
class ChatTemplate:
    """Claude/Llama-3 inspired chat template.

    Special tokens (must be present in the tokenizer vocabulary):
        <|begin_of_text|>           BOS for the whole conversation
        <|start_header_id|>role<|end_header_id|>\n\n   role header
        <|eot|>                     end-of-turn
        <|eom|>                     end-of-message (tool / continuation)
    """

    bos: str = "<|begin_of_text|>"
    start_header: str = "<|start_header_id|>"
    end_header: str = "<|end_header_id|>"
    eot: str = "<|eot|>"
    eom: str = "<|eom|>"

    def render(self, messages: Sequence[ChatMessage], add_generation_prompt: bool = True) -> str:
        parts: list[str] = [self.bos]
        for m in messages:
            parts.append(f"{self.start_header}{m.role}{self.end_header}\n\n{m.content}{self.eot}")
        if add_generation_prompt:
            parts.append(f"{self.start_header}assistant{self.end_header}\n\n")
        return "".join(parts)

    @property
    def stop_tokens(self) -> tuple[str, ...]:
        return (self.eot, self.eom)


@dataclass
class ArisTokenizer:
    """Production tokenizer wrapper."""

    backend: str
    model_path: str
    bos_id: int = 1
    eos_id: int = 2
    pad_id: int = 0
    special_tokens: dict[str, int] = field(default_factory=dict)
    chat_template: ChatTemplate = field(default_factory=ChatTemplate)

    def __post_init__(self) -> None:
        if self.backend == "sentencepiece":
            import sentencepiece as spm

            self._sp = spm.SentencePieceProcessor()
            self._sp.Load(self.model_path)
            self._vocab_size = self._sp.GetPieceSize()
        elif self.backend == "tiktoken":
            import tiktoken

            self._enc = tiktoken.get_encoding(os.path.basename(self.model_path).split(".")[0])
            self._vocab_size = self._enc.n_vocab
        else:
            raise ValueError(f"unknown tokenizer backend: {self.backend}")

    # ------------------------------------------------------------------ #
    # Vocab / properties                                                  #
    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def bos_token_id(self) -> int:
        return self.bos_id

    @property
    def eos_token_id(self) -> int:
        return self.eos_id

    @property
    def pad_token_id(self) -> int:
        return self.pad_id

    # ------------------------------------------------------------------ #
    # Encode / decode                                                     #
    # ------------------------------------------------------------------ #
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        if self.backend == "sentencepiece":
            ids = self._sp.EncodeAsIds(text)
        else:
            ids = self._enc.encode(text, allowed_special="all")
        if add_bos:
            ids = [self.bos_id, *ids]
        if add_eos:
            ids = [*ids, self.eos_id]
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        ids = list(ids)
        if skip_special:
            special = {self.bos_id, self.eos_id, self.pad_id, *self.special_tokens.values()}
            ids = [i for i in ids if i not in special]
        if self.backend == "sentencepiece":
            return self._sp.DecodeIds(ids)
        return self._enc.decode(ids)

    def stream_decoder(self) -> StreamingDecoder:
        return StreamingDecoder(self)

    # ------------------------------------------------------------------ #
    # Chat                                                                 #
    # ------------------------------------------------------------------ #
    def apply_chat_template(
        self, messages: Sequence[ChatMessage], add_generation_prompt: bool = True
    ) -> list[int]:
        rendered = self.chat_template.render(messages, add_generation_prompt=add_generation_prompt)
        return self.encode(rendered, add_bos=False)

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #
    def save_metadata(self, path: str | Path) -> None:
        meta = {
            "backend": self.backend,
            "model_path": self.model_path,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "pad_id": self.pad_id,
            "special_tokens": self.special_tokens,
        }
        Path(path).write_text(json.dumps(meta, indent=2))

    @classmethod
    def from_metadata(cls, path: str | Path) -> ArisTokenizer:
        return cls(**json.loads(Path(path).read_text()))


class StreamingDecoder:
    """Incremental decoder that handles UTF-8 boundary issues."""

    def __init__(self, tokenizer: ArisTokenizer) -> None:
        self._tokenizer = tokenizer
        self._buffer_ids: list[int] = []
        self._emitted_chars = 0

    def step(self, token_id: int) -> str:
        self._buffer_ids.append(token_id)
        text = self._tokenizer.decode(self._buffer_ids, skip_special=True)
        # Only emit text that we are sure won't change with future tokens.
        # Heuristic: keep the last few bytes back to be safe across multi-byte chars.
        safe_len = max(0, len(text) - 4)
        new = text[self._emitted_chars : safe_len]
        self._emitted_chars = safe_len
        return new

    def flush(self) -> str:
        text = self._tokenizer.decode(self._buffer_ids, skip_special=True)
        out = text[self._emitted_chars :]
        self._emitted_chars = len(text)
        return out
