# Corpus

Drop your source documents (PDF / Markdown / HTML) into `raw/`.

This directory is **gitignored** — the corpus is not committed. Keep it **small while iterating**
(especially before any GraphRAG indexing in Phase 6, which is expensive).

Pick a dataset that is hard to answer with plain ChatGPT — internal policy docs, a domain-specific
manual, or a specialized public corpus — so the value of RAG is obvious.
