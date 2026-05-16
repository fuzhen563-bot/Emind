"""
Emind Tokenizer — Universal tokenizer with SentencePiece support
and a built-in fallback for dev environments without network.
"""
import json
import os
from pathlib import Path
from typing import List, Optional, Union, Dict


SPECIAL_TOKENS = {
    "pad": "<pad>",
    "unk": "<unk>",
    "bos": "<s>",
    "eos": "</s>",
}


class EmindTokenizer:
    def __init__(
        self,
        vocab_size: int = 32000,
        model_path: Optional[str] = None,
        special_tokens: Optional[Dict[str, str]] = None,
    ):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or SPECIAL_TOKENS
        self._sp = None
        self._fallback = None

        if model_path and os.path.exists(model_path):
            self._load_sp(model_path)
        else:
            try:
                import sentencepiece as spm
                self._sp = spm.SentencePieceProcessor()
                if model_path:
                    self._sp.Load(model_path)
            except ImportError:
                self._fallback = _FallbackTokenizer(vocab_size, self.special_tokens)

        self.pad_token_id = self.token_to_id(self.special_tokens["pad"])
        self.unk_token_id = self.token_to_id(self.special_tokens["unk"])
        self.bos_token_id = self.token_to_id(self.special_tokens["bos"])
        self.eos_token_id = self.token_to_id(self.special_tokens["eos"])

    def _load_sp(self, model_path: str):
        import sentencepiece as spm
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(model_path)

    @property
    def is_sentencepiece(self) -> bool:
        return self._sp is not None

    def train(self, corpus_path: str, model_prefix: str = "emind_tokenizer"):
        if self._sp is not None:
            import sentencepiece as spm
            spm.SentencePieceTrainer.train(
                input=corpus_path,
                model_prefix=model_prefix,
                vocab_size=self.vocab_size,
                character_coverage=0.9995,
                model_type="bpe",
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                pad_piece=self.special_tokens["pad"],
                unk_piece=self.special_tokens["unk"],
                bos_piece=self.special_tokens["bos"],
                eos_piece=self.special_tokens["eos"],
                user_defined_symbols="<|im_start|>,<|im_end|>,<|tool_call|>,<|tool_result|>",
            )
            self._sp = spm.SentencePieceProcessor()
            self._sp.Load(f"{model_prefix}.model")
        elif self._fallback:
            raise RuntimeError("Cannot train: install sentencepiece for training, or use pre-trained model")

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        if self._sp is not None:
            ids = self._sp.EncodeAsIds(text)
        elif self._fallback:
            ids = self._fallback.encode(text)
        else:
            ids = [ord(c) for c in text]
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        if self._sp is not None:
            if skip_special_tokens:
                ids = [i for i in ids if i not in (self.pad_token_id, self.bos_token_id, self.eos_token_id)]
            return self._sp.DecodeIds(ids)
        elif self._fallback:
            text = self._fallback.decode(ids)
            if skip_special_tokens:
                special_ids = {self.token_to_id(t) for t in self.special_tokens.values()}
                return "".join(c for i, c in enumerate(text) if ord(c) not in special_ids if hasattr(self._fallback, 'itos'))
            return text
        else:
            return "".join(chr(i) for i in ids if 32 <= i < 127 or i > 127)

    def token_to_id(self, token: str) -> int:
        if self._sp is not None:
            return self._sp.PieceToId(token)
        elif self._fallback:
            return self._fallback.stoi.get(token, self._fallback.stoi.get(self.special_tokens["unk"], 1))
        return hash(token) % self.vocab_size

    def id_to_token(self, idx: int) -> str:
        if self._sp is not None:
            return self._sp.IdToPiece(idx)
        elif self._fallback:
            return self._fallback.itos.get(idx, self.special_tokens["unk"])
        return chr(idx) if 32 <= idx < 127 else f"<{idx}>"

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, v: int):
        self._vocab_size = v

    def save(self, path: str):
        data = {"vocab_size": self.vocab_size, "special_tokens": self.special_tokens}
        if self._sp:
            sp_path = Path(path).with_suffix(".model")
            data["sp_model_path"] = str(sp_path)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.vocab_size = data.get("vocab_size", self.vocab_size)
        self.special_tokens = data.get("special_tokens", self.special_tokens)
        sp_path = data.get("sp_model_path")
        if sp_path and os.path.exists(sp_path):
            self._load_sp(sp_path)
        elif not self._sp and not self._fallback:
            self._fallback = _FallbackTokenizer(self.vocab_size, self.special_tokens)

    def __len__(self):
        return self.vocab_size


class _FallbackTokenizer:
    """Character-level fallback for when SentencePiece is unavailable."""

    def __init__(self, vocab_size: int = 32000, special_tokens: Dict[str, str] = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or SPECIAL_TOKENS
        self.stoi = {}
        self.itos = {}
        self._build_default_vocab()

    def _build_default_vocab(self):
        for name, token in self.special_tokens.items():
            idx = len(self.stoi)
            self.stoi[token] = idx
        for i in range(32, min(127, self.vocab_size - len(self.stoi)) + 1):
            self.stoi[chr(i)] = len(self.stoi)
        if len(self.stoi) < self.vocab_size:
            for cp in range(0x4E00, min(0x9FFF, self.vocab_size - len(self.stoi) + 0x4E00)):
                self.stoi[chr(cp)] = len(self.stoi)
        self.itos = {v: k for k, v in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, text: str) -> List[int]:
        ids = []
        for ch in text:
            ids.append(self.stoi.get(ch, self.stoi.get(self.special_tokens["unk"], 1)))
        return ids

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos.get(i, self.special_tokens["unk"]) for i in ids)


def create_tokenizer(vocab_size: int = 32000, model_path: Optional[str] = None) -> EmindTokenizer:
    return EmindTokenizer(vocab_size=vocab_size, model_path=model_path)


# Backward compatibility aliases
SimpleTokenizer = EmindTokenizer
BPETokenizer = EmindTokenizer


if __name__ == "__main__":
    tok = create_tokenizer(1000)
    ids = tok.encode("你好世界，Emind AI！")
    print(f"Encode: {ids}")
    text = tok.decode(ids)
    print(f"Decode: {text}")
    print(f"BOS={tok.bos_token_id}, EOS={tok.eos_token_id}, PAD={tok.pad_token_id}")
    print(f"Vocab size: {len(tok)}")
    print(f"Using SentencePiece: {tok.is_sentencepiece}")
