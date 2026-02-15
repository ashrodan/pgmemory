"""Test utilities — fake embedding provider, helpers."""

from __future__ import annotations

import hashlib

from pgmemory.embeddings import EmbeddingProvider


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic hash-based embeddings for testing. No external calls."""

    def __init__(self, dims: int = 16):
        self._dims = dims

    @property
    def dimensions(self) -> int:
        return self._dims

    async def embed(self, texts):
        results = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            vec = [float(b) / 255.0 for b in h[: self._dims]]
            mag = sum(v ** 2 for v in vec) ** 0.5
            if mag > 0:
                vec = [v / mag for v in vec]
            results.append(vec)
        return results
