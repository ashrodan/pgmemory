# Embedding Providers

pgmemory needs vectors to power semantic search. How you generate them is your business — pick a built-in provider or write your own.

## Google Vertex AI

```bash
pip install pgmemory[vertex]
```

```python
from pgmemory import VertexEmbeddingProvider

embedder = VertexEmbeddingProvider(
    model="text-embedding-004",   # default
    task_type="RETRIEVAL_DOCUMENT",  # default
    dimensionality=768,              # default
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"text-embedding-004"` | Vertex AI embedding model name |
| `task_type` | `"RETRIEVAL_DOCUMENT"` | Task type hint for the model |
| `dimensionality` | `768` | Output vector dimensions |

!!! note
    Requires Google Cloud credentials configured (ADC, service account, etc).

## Ollama (local)

```bash
pip install pgmemory[ollama]
```

```python
from pgmemory import OllamaEmbeddingProvider

embedder = OllamaEmbeddingProvider(
    model="nomic-embed-text",  # default
    dimensionality=768,        # default
    host=None,                 # default (localhost:11434)
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"nomic-embed-text"` | Ollama model name |
| `dimensionality` | `768` | Output vector dimensions |
| `host` | `None` | Ollama server URL (defaults to `localhost:11434`) |

!!! tip
    Pull the model first: `ollama pull nomic-embed-text`

## OpenAI

```bash
pip install pgmemory[openai]
```

```python
from pgmemory import OpenAIEmbeddingProvider

embedder = OpenAIEmbeddingProvider(
    model="text-embedding-3-small",  # default
    dimensionality=1536,             # default
    api_key=None,                    # default (uses OPENAI_API_KEY env var)
    base_url=None,                   # default (override for Azure/proxies)
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"text-embedding-3-small"` | OpenAI embedding model name |
| `dimensionality` | `1536` | Output vector dimensions |
| `api_key` | `None` | API key (falls back to `OPENAI_API_KEY` env var) |
| `base_url` | `None` | Override for Azure OpenAI or compatible APIs |

## Custom provider

Implement the `EmbeddingProvider` abstract class:

```python
from pgmemory import EmbeddingProvider

class MyEmbedder(EmbeddingProvider):
    @property
    def dimensions(self) -> int:
        return 384

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # Return one vector per input text
        return [my_model.encode(t) for t in texts]
```

Two things to implement:

1. **`dimensions`** (property) — return the vector size your model produces
2. **`embed(texts)`** (async method) — take a list of strings, return a list of float vectors

Then pass it to `MemoryStore`:

```python
store = MemoryStore(
    "postgresql+asyncpg://...",
    MyEmbedder(),
)
```

!!! warning
    The `dimensions` value must match the actual vector size returned by `embed()`. pgvector will reject mismatched vectors.
