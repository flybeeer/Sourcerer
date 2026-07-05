"""Confluence loader: turn wiki pages into documents for the RAG pipeline.

A CQL query defines what to ingest — the same idea as the SQL KB's user-written
SELECT ("a query defines what to ingest"); each page returned becomes one
document. Confluence stores page bodies as XHTML ("storage" format), so this
module also strips markup down to plain text before chunking.
"""

from __future__ import annotations

from collections.abc import Iterator
from html.parser import HTMLParser

import httpx

_SEARCH_TIMEOUT = 30.0
_PAGE_SIZE = 25

# Macro/script plumbing whose text content isn't real page content.
_SKIP_TAGS = {"script", "style"}
# Block-level tags become paragraph breaks so chunking sees natural boundaries
# instead of one run-on line.
_BLOCK_TAGS = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"}


class _StorageTextExtractor(HTMLParser):
    """Strips Confluence's XHTML storage format down to plain text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        lines = [line.strip() for line in raw.splitlines()]
        return "\n".join(line for line in lines if line).strip()


def storage_to_text(html: str) -> str:
    """Convert Confluence storage-format XHTML to plain text."""
    extractor = _StorageTextExtractor()
    extractor.feed(html)
    return extractor.text()


# The full expand set the search endpoint accepts is huge (avatars, ARIs,
# per-relation "_expandable" placeholders...); this pulls only the fields that
# are actually useful metadata for a citation/asset record.
_EXPAND = "body.storage,space,version,history,ancestors,metadata.labels"


def _page_metadata(page: dict, fallback_base: str) -> dict:
    """Shape one search result's raw JSON into a flat, useful metadata dict."""
    space = page.get("space") or {}
    version = page.get("version") or {}
    version_by = version.get("by") or {}
    history = page.get("history") or {}
    created_by = history.get("createdBy") or {}
    labels = (page.get("metadata") or {}).get("labels", {}).get("results", []) or []
    ancestors = page.get("ancestors") or []
    links = page.get("_links") or {}
    webui = links.get("webui")

    return {
        "id": page.get("id"),
        "title": page.get("title"),
        "type": page.get("type"),
        "status": page.get("status"),
        "space_key": space.get("key"),
        "space_name": space.get("name"),
        "url": f"{links.get('base', fallback_base)}{webui}" if webui else None,
        "version": version.get("number"),
        "last_modified": version.get("when"),
        "last_modified_by": version_by.get("displayName"),
        "created_at": history.get("createdDate"),
        "created_by": created_by.get("displayName"),
        "labels": [label["name"] for label in labels if label.get("name")],
        "ancestors": [
            {"id": a.get("id"), "title": a.get("title")} for a in ancestors if a.get("id")
        ],
    }


def iter_pages(
    cql: str,
    *,
    base_url: str,
    email: str,
    api_token: str,
    page_size: int = _PAGE_SIZE,
    max_pages: int = 1000,
    client: httpx.Client | None = None,
) -> Iterator[tuple[str, str, dict]]:
    """Yield `(source_label, text, metadata)` for every page a CQL query matches.

    `source_label` is `confluence:<page_id>`, a stable id that survives title
    edits/moves. `base_url` is the site's wiki root, e.g.
    `https://yoursite.atlassian.net/wiki`. Auth is Confluence Cloud's basic
    scheme: account email + API token. `max_pages` is a safety cap (like
    sqlkb's `max_rows`) on how many pages one ingest run pulls.

    `metadata` carries everything Confluence's API offers that's actually
    useful: space, version/last-modified(-by), created(-by), the page's
    ancestor breadcrumb, labels, and the clickable webui URL. It costs no
    extra request — `expand` just widens the same paginated response.

    `client` is an injection point for tests (pass an `httpx.Client` built on
    `httpx.MockTransport`); production callers leave it unset.

    Confluence Cloud's content search paginates via an opaque cursor in
    `_links.next`, not a numeric offset — `start` in the response is not a
    reliable page cursor (a naive start += len(results) loop re-fetches the
    same results forever). This follows `_links.next` until it's absent.
    """
    root = base_url.rstrip("/")
    owns_client = client is None
    http = client or httpx.Client(auth=httpx.BasicAuth(email, api_token), timeout=_SEARCH_TIMEOUT)
    try:
        seen = 0
        url = f"{root}/rest/api/content/search"
        params: dict | None = {
            "cql": cql,
            "limit": min(page_size, max_pages),
            "expand": _EXPAND,
        }
        while url and seen < max_pages:
            resp = http.get(url, params=params)
            resp.raise_for_status()
            resp_body = resp.json()
            results = resp_body.get("results", [])
            if not results:
                return
            for page in results:
                storage = page.get("body", {}).get("storage", {}).get("value", "")
                text = storage_to_text(storage)
                if text:
                    title = page.get("title", "")
                    doc = f"{title}\n\n{text}" if title else text
                    yield f"confluence:{page['id']}", doc, _page_metadata(page, root)
                seen += 1
                if seen >= max_pages:
                    return
            links = resp_body.get("_links", {})
            next_link = links.get("next")
            if not next_link:
                return
            url = links.get("base", root) + next_link
            params = None  # already encoded in next_link
    finally:
        if owns_client:
            http.close()
