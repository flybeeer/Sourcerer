"""Tests for the Confluence loader: HTML-to-text stripping and CQL pagination.

No network needed: pagination is exercised against httpx.MockTransport, the
same fake-transport seam httpx itself recommends for tests.
"""

from __future__ import annotations

import httpx

from sourcerer.ingestion.confluence import _page_metadata, iter_pages, storage_to_text

# --- storage_to_text ---------------------------------------------------------


def test_storage_to_text_strips_tags():
    html = "<p>Hello <strong>world</strong></p>"
    assert storage_to_text(html) == "Hello world"


def test_storage_to_text_breaks_on_block_tags():
    html = "<p>First paragraph</p><p>Second paragraph</p>"
    assert storage_to_text(html) == "First paragraph\nSecond paragraph"


def test_storage_to_text_decodes_entities():
    assert storage_to_text("<p>Tom &amp; Jerry</p>") == "Tom & Jerry"


def test_storage_to_text_drops_script_content():
    html = "<p>Visible</p><script>var x = 1;</script>"
    assert storage_to_text(html) == "Visible"


def test_storage_to_text_empty_input():
    assert storage_to_text("") == ""
    assert storage_to_text("<p>   </p>") == ""


# --- iter_pages ---------------------------------------------------------


def _page(page_id: str, title: str, html: str) -> dict:
    return {"id": page_id, "title": title, "body": {"storage": {"value": html}}}


def _client(pages: list[dict]) -> httpx.Client:
    """Mocks Confluence's real cursor-based pagination (`_links.next`), not a
    numeric offset — this is what the fixed loader actually has to follow."""

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        limit = int(params["limit"])
        offset = int(params.get("cursor", 0))
        batch = pages[offset : offset + limit]
        next_offset = offset + limit
        links = {"base": "https://co.atlassian.net/wiki"}
        if next_offset < len(pages):
            links["next"] = f"/rest/api/content/search?cursor={next_offset}&limit={limit}"
        return httpx.Response(200, json={"results": batch, "_links": links})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_iter_pages_yields_source_text_and_metadata():
    pages = [_page("101", "Onboarding", "<p>Welcome aboard</p>")]
    client = _client(pages)
    results = list(
        iter_pages(
            "space = ENG",
            base_url="https://co.atlassian.net/wiki",
            email="a@co.com",
            api_token="tok",
            client=client,
        )
    )
    assert len(results) == 1
    source, text, metadata = results[0]
    assert (source, text) == ("confluence:101", "Onboarding\n\nWelcome aboard")
    assert metadata["id"] == "101"
    assert metadata["title"] == "Onboarding"


def test_iter_pages_skips_empty_body():
    pages = [_page("1", "Empty", "<p></p>"), _page("2", "Real", "<p>content</p>")]
    client = _client(pages)
    results = list(
        iter_pages(
            "space = ENG",
            base_url="https://co.atlassian.net/wiki",
            email="a@co.com",
            api_token="tok",
            client=client,
        )
    )
    assert [s for s, _, _ in results] == ["confluence:2"]


def test_iter_pages_paginates_across_batches():
    pages = [_page(str(i), f"Page {i}", f"<p>body {i}</p>") for i in range(5)]
    client = _client(pages)
    results = list(
        iter_pages(
            "space = ENG",
            base_url="https://co.atlassian.net/wiki",
            email="a@co.com",
            api_token="tok",
            page_size=2,
            client=client,
        )
    )
    assert [s for s, _, _ in results] == [f"confluence:{i}" for i in range(5)]


def test_iter_pages_respects_max_pages_cap():
    pages = [_page(str(i), f"Page {i}", f"<p>body {i}</p>") for i in range(10)]
    client = _client(pages)
    results = list(
        iter_pages(
            "space = ENG",
            base_url="https://co.atlassian.net/wiki",
            email="a@co.com",
            api_token="tok",
            page_size=2,
            max_pages=3,
            client=client,
        )
    )
    assert len(results) == 3


def test_iter_pages_no_results_yields_nothing():
    client = _client([])
    results = list(
        iter_pages(
            "space = EMPTY",
            base_url="https://co.atlassian.net/wiki",
            email="a@co.com",
            api_token="tok",
            client=client,
        )
    )
    assert results == []


# --- _page_metadata ---------------------------------------------------------


def test_page_metadata_shapes_full_response():
    page = {
        "id": "9997914154",
        "title": "Test Case Format",
        "type": "page",
        "status": "current",
        "space": {"key": "QA", "name": "Quality Assurance"},
        "version": {
            "number": 3,
            "when": "2026-05-01T10:00:00.000Z",
            "by": {"displayName": "Alice"},
        },
        "history": {
            "createdDate": "2026-01-15T09:00:00.000Z",
            "createdBy": {"displayName": "Bob"},
        },
        "ancestors": [{"id": "111", "title": "QA Home"}, {"id": "222", "title": "Test Process"}],
        "metadata": {"labels": {"results": [{"name": "process"}, {"name": "qa"}]}},
        "_links": {
            "base": "https://co.atlassian.net/wiki",
            "webui": "/spaces/QA/pages/9997914154/Test+Case+Format",
        },
    }
    metadata = _page_metadata(page, fallback_base="https://fallback.example")
    assert metadata == {
        "id": "9997914154",
        "title": "Test Case Format",
        "type": "page",
        "status": "current",
        "space_key": "QA",
        "space_name": "Quality Assurance",
        "url": "https://co.atlassian.net/wiki/spaces/QA/pages/9997914154/Test+Case+Format",
        "version": 3,
        "last_modified": "2026-05-01T10:00:00.000Z",
        "last_modified_by": "Alice",
        "created_at": "2026-01-15T09:00:00.000Z",
        "created_by": "Bob",
        "labels": ["process", "qa"],
        "ancestors": [{"id": "111", "title": "QA Home"}, {"id": "222", "title": "Test Process"}],
    }


def test_page_metadata_handles_missing_optional_fields():
    page = {"id": "1", "title": "Bare", "type": "page", "status": "current"}
    metadata = _page_metadata(page, fallback_base="https://co.atlassian.net/wiki")
    assert metadata["space_key"] is None
    assert metadata["url"] is None
    assert metadata["labels"] == []
    assert metadata["ancestors"] == []
    assert metadata["last_modified_by"] is None
    assert metadata["created_by"] is None
