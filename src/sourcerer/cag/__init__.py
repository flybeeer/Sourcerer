"""CAG (Cache-Augmented Generation) PoC — governance-partitioned context caches.

RAG retrieves per query; CAG preloads the corpus into the model's context once and
answers from that cache (no per-query retrieval). The governance catch: a single
all-documents cache would put forbidden content in every principal's context,
breaking Phase 10's rule ("forbidden content never enters the context").

The fix here: run the Cerbos gate at **cache-build time**, not retrieval time. Each
principal's cache contains only the sources they may read (decided by the same
`governance.allowed_document_sources` used by the RAG path). Principals with the
same allowed-set share one cache (keyed by access profile), so it's caches per
*tier*, not per *user*.
"""
