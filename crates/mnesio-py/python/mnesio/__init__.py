"""mnesio — self-improving long-term memory layer for AI agents.

Three primitives:

- ``Client(data_dir, embedder="mock")`` — open / create a mnesio store.
- ``client.write_memory(content, tenant, tags=None)`` — append a memory.
- ``client.search(query, tenant="default", k=5)`` — hybrid retrieval +
  synthesized answer.
- ``client.record_outcome(artifacts_used, success, ...)`` — record agent
  task outcome for the procedural compiler to learn from.

Synchronous: each call blocks the caller until complete. A future
release will add ``AsyncClient`` for native asyncio integration.
"""

from ._mnesio import Client, SearchResult, SearchHit, MnesioError

__all__ = ["Client", "SearchResult", "SearchHit", "MnesioError"]
