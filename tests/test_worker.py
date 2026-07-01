"""Worker-assembly and version-consistency tests."""

from __future__ import annotations

import json

from vgi_scikit_bio import __version__
from vgi_scikit_bio.worker import (
    _FUNCTIONS,
    _SCHEMA_FUNCTIONS,
    _SCHEMA_META,
    _SKBIO_CATALOG,
    DATA_VERSION,
    IMPLEMENTATION_VERSION,
)


def test_version_is_single_sourced() -> None:
    assert __version__ == IMPLEMENTATION_VERSION
    assert __version__ == DATA_VERSION


def test_catalog_identity() -> None:
    assert _SKBIO_CATALOG.name == "skbio"
    assert _SKBIO_CATALOG.default_schema == "sequence"


def test_every_schema_has_metadata() -> None:
    assert set(_SCHEMA_FUNCTIONS) == set(_SCHEMA_META)


def test_function_names_are_unique() -> None:
    names = [fn.Meta.name for fn in _FUNCTIONS]
    assert len(names) == len(set(names))


def test_every_function_has_docs() -> None:
    for fn in _FUNCTIONS:
        tags = getattr(fn.Meta, "tags", {})
        assert "vgi.doc_llm" in tags, f"{fn.Meta.name} missing vgi.doc_llm"
        assert "vgi.doc_md" in tags, f"{fn.Meta.name} missing vgi.doc_md"


def test_schema_example_queries_are_valid_json() -> None:
    for meta in _SCHEMA_META.values():
        assert isinstance(json.loads(meta["example_queries"]), list)
