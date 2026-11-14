"""Worker-assembly and version-consistency tests."""

from __future__ import annotations

import json

from vgi_scikit_bio import __version__
from vgi_scikit_bio.alignment import ALIGNMENT_IDENTITY_EXAMPLE
from vgi_scikit_bio.kmers import KMER_DISTANCE_EXAMPLE
from vgi_scikit_bio.phylo import RAREFIED_ALPHA_EXAMPLE
from vgi_scikit_bio.sequence import SYNONYMOUS_VARIANT_EXAMPLE
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


def test_schema_example_queries_are_ordered_projections() -> None:
    # Schema-level examples must yield stable, readable output: an ORDER BY when
    # they return more than one row's worth of grouping, and no bare SELECT *.
    for name, meta in _SCHEMA_META.items():
        for example in json.loads(meta["example_queries"]):
            sql = example["sql"]
            assert not sql.startswith("SELECT * FROM"), f"{name}: bare SELECT * in {sql!r}"
            if "GROUP BY" in sql or "FROM skbio." in sql:
                assert "ORDER BY" in sql, f"{name}: unordered multi-row example {sql!r}"


def test_composed_examples_are_published_where_agents_see_them() -> None:
    # Each composed example sits within the first five examples of the function
    # that drives it (the agent simulator shows only those) and is re-published,
    # verbatim, on that function's schema-level example list.
    placements = {
        "translate": ("sequence", SYNONYMOUS_VARIANT_EXAMPLE),
        "kmer_frequencies": ("sequence", KMER_DISTANCE_EXAMPLE),
        "pairwise_align_nucleotide": ("alignment", ALIGNMENT_IDENTITY_EXAMPLE),
        "subsample_counts": ("diversity", RAREFIED_ALPHA_EXAMPLE),
    }
    by_name = {fn.Meta.name: fn for fn in _FUNCTIONS}
    for fn_name, (schema, example) in placements.items():
        assert example in list(by_name[fn_name].Meta.examples)[:5], fn_name
        published = json.loads(_SCHEMA_META[schema]["example_queries"])
        assert {"description": example.description, "sql": example.sql} in published, fn_name
