from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np


def completion_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, list) and item:
        value = item[-1]
        if isinstance(value, dict):
            return str(value.get("content", "")).strip()
    return str(item).strip()


def character_ngrams(text: str, n_min: int, n_max: int) -> list[str]:
    compact = re.sub(r"\s+", "", text)
    grams: list[str] = []
    for n in range(n_min, n_max + 1):
        grams.extend(compact[i : i + n] for i in range(max(0, len(compact) - n + 1)))
    return grams or [compact or "<空>"]


@dataclass
class HashEmbedder:
    dimension: int = 2048
    ngram_min: int = 2
    ngram_max: int = 4

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            counts = Counter(character_ngrams(text, self.ngram_min, self.ngram_max))
            for gram, count in counts.items():
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                column = value % self.dimension
                sign = 1.0 if value & 1 else -1.0
                matrix[row, column] += sign * (1.0 + math.log(count))
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-12, None)

class ModelEmbedder:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 16,
        max_length: int = 256,
    ):
        from sentence_transformers import SentenceTransformer

        self.batch_size = batch_size

        self.model = SentenceTransformer(
            model_path,
            device=device,
            trust_remote_code=True,
            local_files_only=True,
        )

        self.model.max_seq_length = max_length

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings

def build_embedder(config: dict[str, Any]):
    backend = config.get("backend", "hash")
    if backend == "hash":
        return HashEmbedder(
            dimension=int(config.get("dimension", 2048)),
            ngram_min=int(config.get("ngram_min", 2)),
            ngram_max=int(config.get("ngram_max", 4)),
        )
    if backend == "model":
        if not config.get("model_path"):
            raise ValueError("使用模型嵌入时必须配置 embedding.model_path。")
        return ModelEmbedder(
            model_path=str(config["model_path"]),
            device=str(config.get("device", "cpu")),
            batch_size=int(config.get("batch_size", 16)),
            max_length=int(config.get("max_length", 256)),
        )
    raise ValueError(f"未知嵌入方式：{backend}")


def lexical_f1(prediction: str, target: str) -> float:
    pred = Counter(character_ngrams(prediction, 1, 2))
    gold = Counter(character_ngrams(target, 1, 2))
    overlap = sum((pred & gold).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(sum(pred.values()), 1)
    recall = overlap / max(sum(gold.values()), 1)
    return 2 * precision * recall / (precision + recall)


def validity_score(text: str, cfg: dict[str, Any]) -> float:
    if not text:
        return -float(cfg.get("empty_penalty", 1.0))
    score = 1.0 if int(cfg.get("min_chars", 2)) <= len(text) <= int(cfg.get("max_chars", 180)) else 0.0
    if re.search(r"(?:作为|身为).{0,8}(?:助手|模型)|坐席[:：]|客服[:：]", text):
        score -= float(cfg.get("role_leak_penalty", 0.5))
    chunks = re.findall(r".{2,8}", text)
    if chunks and max(Counter(chunks).values()) >= 3:
        score -= float(cfg.get("repetition_penalty", 0.3))
    return score


class UserSimulatorRewards:
    def __init__(self, embedding_config: dict[str, Any], reward_config: dict[str, Any], group_size: int):
        self.embedder = build_embedder(embedding_config)
        self.cfg = reward_config
        self.group_size = group_size
        self._cache_key: tuple | None = None
        self._cache_embeddings: tuple[np.ndarray, np.ndarray] | None = None

    def _embeddings(self, completions: list[Any], ground_truth: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
        texts = [completion_text(item) for item in completions]
        key = tuple(texts) + tuple(map(str, ground_truth))
        if key != self._cache_key:
            combined = texts + list(map(str, ground_truth))
            encoded = self.embedder.encode(combined)
            self._cache_key = key
            self._cache_embeddings = (encoded[: len(texts)], encoded[len(texts) :])
        assert self._cache_embeddings is not None
        return texts, self._cache_embeddings[0], self._cache_embeddings[1]

    def base_reward(self, completions: list[Any], ground_truth: list[str], **_: Any) -> list[float]:
        texts, pred_vectors, gold_vectors = self._embeddings(completions, ground_truth)
        semantic = np.sum(pred_vectors * gold_vectors, axis=1)
        results = []
        for text, target, similarity in zip(texts, ground_truth, semantic):
            value = (
                float(self.cfg.get("semantic_weight", 0.8)) * float(similarity)
                + float(self.cfg.get("lexical_weight", 0.1)) * lexical_f1(text, str(target))
                + float(self.cfg.get("validity_weight", 0.1)) * validity_score(text, self.cfg)
            )
            results.append(value)
        return results

    def diversity_marginal_reward(
        self,
        completions: list[Any],
        ground_truth: list[str],
        **_: Any,
    ) -> list[float]:
        _, vectors, _ = self._embeddings(completions, ground_truth)

        if len(vectors) % self.group_size:
            raise ValueError(
                f"回复数量 {len(vectors)} 不能被组大小 {self.group_size} 整除。"
            )

        eps = float(self.cfg.get("renyi_eps", 1e-8))
        rewards: list[float] = []

        def renyi2_u_stat(group: np.ndarray) -> float:
            n = len(group)

            if n < 2:
                return 0.0

            norms = np.linalg.norm(group, axis=1, keepdims=True)
            group = group / np.maximum(norms, eps)
            # Cosine-similarity kernel.
            kernel = group @ group.T


            # U-statistic over off-diagonal pairs only.
            upper = np.triu_indices(n, k=1)
            pair_kernel = kernel[upper]

            collision = float(np.mean(pair_kernel))

            # Rényi-2 entropy proxy.
            collision = max(collision, eps)

            return -np.log(collision)

        for start in range(0, len(vectors), self.group_size):
            group = vectors[start : start + self.group_size]

            # H2(S).
            full_entropy = renyi2_u_stat(group)

            for index in range(self.group_size):
                # Leave one response out.
                reduced = np.delete(group, index, axis=0)

                # H2(S \ {i}).
                reduced_entropy = renyi2_u_stat(reduced)

                # Marginal contribution of response i to group diversity.
                marginal = full_entropy - reduced_entropy

                rewards.append(float(marginal))

        return rewards