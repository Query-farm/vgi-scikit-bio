"""VGI worker exposing scikit-bio to DuckDB/SQL.

Assembles the per-area implementation modules in ``vgi_scikit_bio`` into a single
``skbio`` catalog and provides the process entry points. The repo-root
``scikit_bio_worker.py`` / ``serve.py`` are thin shims over this module for
``uv run`` and the container; installed users get the ``vgi-scikit-bio`` and
``vgi-scikit-bio-http`` console scripts, which call ``main`` / ``main_http`` here.

    ATTACH 'skbio' (TYPE vgi, LOCATION 'vgi-scikit-bio');
    SELECT skbio.sequence.gc_content('ATGCGGATTACAGG');
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
from typing import Any

from vgi import Worker
from vgi.catalog import Catalog, ReadOnlyCatalogInterface, Schema
from vgi.catalog.catalog_interface import CatalogAttachResult, CatalogInfo

from vgi_scikit_bio import __version__
from vgi_scikit_bio.composition import COMPOSITION_FUNCTIONS
from vgi_scikit_bio.distance_stats import DISTANCE_STATS_FUNCTIONS
from vgi_scikit_bio.diversity import DIVERSITY_FUNCTIONS
from vgi_scikit_bio.kmers import KMER_FUNCTIONS
from vgi_scikit_bio.ordination import ORDINATION_FUNCTIONS
from vgi_scikit_bio.sequence import SEQUENCE_FUNCTIONS
from vgi_scikit_bio.tree import TREE_FUNCTIONS

log = logging.getLogger(__name__)

# The version the worker advertises over VGI. `implementation_version` is the
# worker *software* version (a semver per the VGI protocol), so it must be the
# released package version — not a build/commit id. Both it and the data version
# track __version__, which is the single source bumped per release.
IMPLEMENTATION_VERSION = __version__
DATA_VERSION = __version__
# data_version_spec is advertised as a SemVer *range* (a packaging SpecifierSet),
# not a bare version — an exact-match range, since the worker serves exactly the
# current data version.
DATA_VERSION_SPEC = f"=={DATA_VERSION}"
# Build provenance only (Sentry release / diagnostics) — NOT the advertised
# implementation version, which must stay a semver.
GIT_COMMIT = os.environ.get("VGI_SCIKIT_BIO_GIT_COMMIT") or "unknown"

# Functions are split across schemas by scikit-bio area. The default `sequence`
# schema holds the per-sequence functions, so `skbio.sequence.gc_content(...)`
# also resolves unqualified.
_DEFAULT_SCHEMA = "sequence"
_SCHEMA_FUNCTIONS: dict[str, list[type]] = {
    "sequence": [*SEQUENCE_FUNCTIONS, *KMER_FUNCTIONS],
    "diversity": [*DIVERSITY_FUNCTIONS],
    "stats": [*ORDINATION_FUNCTIONS, *DISTANCE_STATS_FUNCTIONS, *COMPOSITION_FUNCTIONS],
    "tree": [*TREE_FUNCTIONS],
}
_FUNCTIONS: list[type] = [fn for fns in _SCHEMA_FUNCTIONS.values() for fn in fns]

# Provenance / about link advertised on the catalog (VGI source_url).
SOURCE_URL = "https://github.com/query-farm/vgi-scikit-bio"

# Catalog-level metadata surfaced through duckdb_databases() (comment + tags).
_CATALOG_COMMENT = "scikit-bio for SQL: sequence analysis, community diversity, ordination, and phylogenetics in DuckDB"
_CATALOG_DESCRIPTION_LLM = (
    "scikit-bio for SQL. Analyze biological sequences (GC content, reverse complement, translation, "
    "k-mer and residue composition); compute alpha diversity as aggregates and beta-diversity distance "
    "matrices; run PCoA ordination, PERMANOVA/ANOSIM/Mantel distance tests, and CLR/ILR compositional "
    "transforms; and build neighbour-joining trees — all as DuckDB scalar, aggregate, and table functions."
)
_CATALOG_DESCRIPTION_MD = (
    "# scikit-bio for SQL\n\n"
    "Exposes [scikit-bio](https://scikit.bio) to DuckDB/SQL as VGI functions:\n\n"
    "- **Sequence** — GC content, reverse complement, translation, k-mer & residue composition\n"
    "- **Diversity** — alpha-diversity aggregates (`shannon`, `chao1`, ...) and beta-diversity matrices\n"
    "- **Stats** — PCoA ordination, PERMANOVA / ANOSIM / Mantel tests, CLR/ILR transforms\n"
    "- **Tree** — neighbour-joining construction and Newick inspection"
)
# Guaranteed-runnable, self-contained examples advertised on the catalog
# (VGI509): each is fully schema-qualified and executes as written against a
# freshly attached worker.
_CATALOG_EXECUTABLE_EXAMPLES = json.dumps(
    [
        {
            "description": "GC content of a DNA sequence.",
            "sql": "SELECT skbio.sequence.gc_content('ATGCGGATTACAGG') AS gc",
        },
        {
            "description": "Reverse complement of a DNA sequence.",
            "sql": "SELECT skbio.sequence.reverse_complement('ATGCGGATTACAGG') AS rc",
        },
        {
            "description": "3-mer composition of two reads (long format).",
            "sql": (
                "SELECT * FROM skbio.sequence.kmer_frequencies((SELECT * FROM "
                "(VALUES (1, 'ATGCGGATTACAGG'), (2, 'TTGCACGT')) AS reads(id, seq)), id := 'id', k := 3)"
            ),
        },
        {
            "description": "Shannon alpha diversity per sample over a long feature table.",
            "sql": (
                "SELECT sample_id, skbio.diversity.shannon(count) AS shannon FROM "
                "(VALUES (1,'a',4),(1,'b',2),(1,'c',1),(2,'a',1),(2,'b',9)) AS t(sample_id, feature_id, count) "
                "GROUP BY sample_id ORDER BY sample_id"
            ),
        },
        {
            "description": "Bray-Curtis distance matrix, then a 2-axis PCoA embedding.",
            "sql": (
                "SELECT * FROM skbio.stats.pcoa((SELECT * FROM skbio.diversity.beta_diversity((SELECT * FROM "
                "(VALUES ('s1','a',4),('s1','b',1),('s2','a',3),('s2','b',2),('s3','a',0),('s3','b',9)) "
                "AS t(sample_id, feature_id, count)))), n_components := 2)"
            ),
        },
    ]
)
_CATALOG_TAGS = {
    "vgi.doc_llm": _CATALOG_DESCRIPTION_LLM,
    "vgi.doc_md": _CATALOG_DESCRIPTION_MD,
    "vgi.author": "Query Farm <hello@query.farm>",
    "vgi.copyright": "Copyright 2026 Query Farm LLC - https://query.farm",
    "vgi.license": "MIT",
    "vgi.support_contact": f"{SOURCE_URL}/issues",
    "vgi.support_policy_url": f"{SOURCE_URL}/blob/main/SUPPORT.md",
    "vgi.title": "scikit-bio for SQL",
    "vgi.keywords": json.dumps(
        [
            "scikit-bio",
            "bioinformatics",
            "sequence",
            "dna",
            "diversity",
            "ordination",
            "phylogenetics",
            "microbiome",
            "distance matrix",
            "composition",
        ]
    ),
    "vgi.executable_examples": _CATALOG_EXECUTABLE_EXAMPLES,
    # Analyst tasks for `vgi-lint simulate` — natural-language prompts an agent
    # should be able to satisfy using this worker, each with a reference query.
    "vgi.agent_test_tasks": json.dumps(
        [
            {
                "name": "gc_content",
                "prompt": "What is the GC content of the DNA sequence ATGCGGATTACAGG?",
                "reference_sql": "SELECT skbio.sequence.gc_content('ATGCGGATTACAGG')",
            },
            {
                "name": "reverse_complement",
                "prompt": "Give the reverse complement of the DNA sequence ATGCGGATTACAGG.",
                "reference_sql": "SELECT skbio.sequence.reverse_complement('ATGCGGATTACAGG')",
            },
            {
                "name": "alpha_diversity",
                "prompt": (
                    "Given a long feature table of (sample_id, feature_id, count), compute the Shannon "
                    "alpha diversity of each sample."
                ),
                "reference_sql": (
                    "SELECT sample_id, skbio.diversity.shannon(count) FROM "
                    "(VALUES (1,'a',4),(1,'b',2),(2,'a',1),(2,'b',9)) AS t(sample_id, feature_id, count) "
                    "GROUP BY sample_id"
                ),
            },
            {
                "name": "beta_diversity_ordination",
                "prompt": (
                    "From a long (sample_id, feature_id, count) table, compute a Bray-Curtis distance "
                    "matrix and embed the samples in two dimensions with PCoA."
                ),
                "reference_sql": (
                    "SELECT * FROM skbio.stats.pcoa((SELECT * FROM skbio.diversity.beta_diversity((SELECT * FROM "
                    "(VALUES ('s1','a',4),('s1','b',1),('s2','a',3),('s2','b',2),('s3','a',0),('s3','b',9)) "
                    "AS t(sample_id, feature_id, count)))), n_components := 2)"
                ),
            },
        ]
    ),
}

# Per-schema metadata. Each schema carries its own description/title/keywords and
# a runnable, schema-qualified example query (VGI112/113/124/126/506).
_SCHEMA_META: dict[str, dict[str, str]] = {
    "sequence": {
        "comment": "Nucleotide/protein sequence functions plus k-mer and residue composition.",
        "title": "Sequence",
        "keywords": json.dumps(["sequence", "dna", "rna", "protein", "kmer"]),
        "doc_llm": (
            "Per-sequence functions over VARCHAR sequence columns: scalar transforms (`gc_content`, "
            "`reverse_complement`, `complement`, `transcribe`, `translate`), validation (`is_valid_dna`, "
            "`is_valid_protein`), pairwise `hamming_distance`, and composition table functions "
            "(`kmer_frequencies`, `residue_frequencies`) that emit long token-count matrices. DNA, RNA, "
            "and protein are all supported; malformed rows yield NULL rather than failing the query."
        ),
        "doc_md": (
            "### Sequence\n\n"
            "Analyze biological sequences directly in SQL:\n\n"
            "- **Transforms** — `gc_content`, `reverse_complement`, `complement`, `transcribe`, `translate`\n"
            "- **Validate** — `is_valid_dna`, `is_valid_protein`; **compare** — `hamming_distance`\n"
            "- **Composition** — `kmer_frequencies`, `residue_frequencies` (long token-count matrices)"
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "GC content of a DNA sequence",
                    "sql": "SELECT skbio.sequence.gc_content('ATGCGGATTACAGG')",
                }
            ]
        ),
    },
    "diversity": {
        "comment": "Alpha-diversity aggregates and beta-diversity distance matrices.",
        "title": "Diversity",
        "keywords": json.dumps(["diversity", "alpha", "beta", "ecology", "microbiome"]),
        "doc_llm": (
            "Community-ecology diversity over a long `(sample, feature, count)` table: alpha-diversity "
            "aggregates (`shannon`, `simpson`, `inv_simpson`, `observed_features`, `chao1`, "
            "`pielou_evenness`, `dominance`) that give one value per sample under `GROUP BY`, and the "
            "`beta_diversity` table function that emits the between-sample distance matrix long, ready for "
            "`skbio.stats.pcoa`/`permanova`."
        ),
        "doc_md": (
            "### Diversity\n\n"
            "Community diversity from a long `(sample, feature, count)` table:\n\n"
            "- **Alpha** (aggregates, `GROUP BY sample`) — `shannon`, `simpson`, `inv_simpson`, "
            "`observed_features`, `chao1`, `pielou_evenness`, `dominance`\n"
            "- **Beta** — `beta_diversity` emits the distance matrix long for `pcoa` / `permanova`"
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "Shannon diversity per sample",
                    "sql": (
                        "SELECT sample_id, skbio.diversity.shannon(count) FROM "
                        "(VALUES (1,'a',4),(1,'b',2),(2,'a',1),(2,'b',9)) AS t(sample_id, feature_id, count) "
                        "GROUP BY sample_id"
                    ),
                }
            ]
        ),
    },
    "stats": {
        "comment": "Ordination (PCoA), distance-matrix tests, and compositional transforms.",
        "title": "Stats",
        "keywords": json.dumps(["ordination", "pcoa", "permanova", "mantel", "composition"]),
        "doc_llm": (
            "Multivariate statistics over distance matrices and compositions: `pcoa` embeds samples from a "
            "distance matrix; `permanova`/`anosim` test whether a grouping explains between-sample "
            "distances; `mantel` correlates two distance matrices; and `clr`/`ilr` are log-ratio "
            "transforms of compositional feature tables. Distance-matrix inputs use the long "
            "`(id_1, id_2, distance)` shape produced by `skbio.diversity.beta_diversity`."
        ),
        "doc_md": (
            "### Stats\n\n"
            "Multivariate analysis over distance matrices and compositions:\n\n"
            "- **Ordination** — `pcoa` (principal coordinates)\n"
            "- **Distance tests** — `permanova`, `anosim`, `mantel`\n"
            "- **Composition** — `clr`, `ilr` log-ratio transforms\n\n"
            "Distance inputs use the long `(id_1, id_2, distance)` shape from `beta_diversity`."
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "PCoA of a distance matrix",
                    "sql": (
                        "SELECT * FROM skbio.stats.pcoa((SELECT * FROM "
                        "(VALUES ('a','a',0.0),('a','b',0.5),('a','c',0.7),('b','a',0.5),('b','b',0.0),"
                        "('b','c',0.6),('c','a',0.7),('c','b',0.6),('c','c',0.0)) AS d(id_1, id_2, distance)), "
                        "n_components := 2)"
                    ),
                }
            ]
        ),
    },
    "tree": {
        "comment": "Neighbour-joining tree construction and Newick inspection.",
        "title": "Tree",
        "keywords": json.dumps(["tree", "phylogenetics", "newick", "neighbor joining", "distance"]),
        "doc_llm": (
            "Phylogenetics: `neighbor_joining` builds an unrooted tree from a long distance matrix and "
            "returns it as a Newick string, while the `tip_count` and `total_branch_length` scalars "
            "inspect Newick strings stored per row. Pairs naturally with `skbio.diversity.beta_diversity` "
            "as the distance source."
        ),
        "doc_md": (
            "### Tree\n\n"
            "Build and inspect phylogenetic trees:\n\n"
            "- **Construct** — `neighbor_joining` (distance matrix → Newick)\n"
            "- **Inspect** — `tip_count`, `total_branch_length` (scalars over Newick strings)"
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "Tips of a Newick tree",
                    "sql": "SELECT skbio.tree.tip_count('((a:2,b:3):3,d:4,c:4);')",
                }
            ]
        ),
    },
}


def _humanize(name: str) -> str:
    """Title-case a snake_case function name for a display title."""
    return name.replace("_", " ").title()


def _apply_discovery_tags(functions: list[type]) -> None:
    """Inject the per-function discovery tags the catalog-quality linter expects.

    ``vgi.title`` and ``vgi.keywords`` (a JSON array of strings) are derived
    mechanically from each function's existing Meta (display name, categories).
    The richer ``vgi.doc_llm`` / ``vgi.doc_md`` tags are authored per function in
    the implementation modules and are left untouched here.
    """
    for fn in functions:
        meta = getattr(fn, "Meta", None)
        if meta is None:
            continue
        name = getattr(meta, "name", fn.__name__)
        cats = list(getattr(meta, "categories", []) or [])
        tags = dict(getattr(meta, "tags", {}) or {})
        tags.setdefault("vgi.title", _humanize(name))
        keywords = list(dict.fromkeys(cats or name.split("_")))
        tags.setdefault("vgi.keywords", json.dumps(keywords))
        meta.tags = tags


_apply_discovery_tags(_FUNCTIONS)


def _build_schema(name: str, functions: list[type]) -> Schema:
    """Build a ``Schema`` from its function list and the ``_SCHEMA_META`` entry."""
    meta = _SCHEMA_META[name]
    return Schema(
        name=name,
        comment=meta["comment"],
        tags={
            "provider": "scikit-bio",
            "domain": "bioinformatics",
            "vgi.title": meta["title"],
            "vgi.keywords": meta["keywords"],
            "vgi.doc_llm": meta["doc_llm"],
            "vgi.doc_md": meta["doc_md"],
            "vgi.example_queries": meta["example_queries"],
        },
        functions=functions,
    )


_SKBIO_CATALOG = Catalog(
    name="skbio",
    default_schema=_DEFAULT_SCHEMA,
    comment=_CATALOG_COMMENT,
    tags=_CATALOG_TAGS,
    schemas=[_build_schema(name, functions) for name, functions in _SCHEMA_FUNCTIONS.items()],
)


class SkbioCatalog(ReadOnlyCatalogInterface):
    """Advertises the worker's data + implementation version on ATTACH."""

    catalog = _SKBIO_CATALOG
    catalog_name = _SKBIO_CATALOG.name

    def catalogs(self) -> list[CatalogInfo]:
        """Advertise the catalog with its implementation and data versions."""
        return [
            CatalogInfo(
                name=self._effective_catalog_name,
                implementation_version=IMPLEMENTATION_VERSION,
                data_version_spec=DATA_VERSION_SPEC,
                source_url=SOURCE_URL,
                attach_option_specs=[spec.serialize() for spec in self.attach_option_specs],
            )
        ]

    def catalog_attach(self, **kwargs: Any) -> CatalogAttachResult:
        """Resolve the data and implementation versions reported on ATTACH."""
        result = super().catalog_attach(**kwargs)
        return dataclasses.replace(
            result,
            resolved_data_version=DATA_VERSION,
            resolved_implementation_version=IMPLEMENTATION_VERSION,
        )


class ScikitBioWorker(Worker):
    """Worker process hosting the scikit-bio catalog."""

    catalog = _SKBIO_CATALOG
    catalog_interface = SkbioCatalog


def _warn_if_ephemeral_state() -> None:
    """Warn when the worker's shared-state dir looks container-local (no volume mounted).

    The published image declares a ``/data`` volume (advertised via the
    ``farm.query.vgi.volumes`` image label) holding the shared ``BoundStorage``
    SQLite. If the worker runs with that default but ``/data`` is not actually a
    mounted volume, framework state lives on the container's writable layer and
    vanishes on ``docker run --rm``. Surface that instead of silently losing it.

    A no-op outside that container shape: it only fires when the state dir is
    rooted under ``/data`` and ``/proc/mounts`` is readable (a Linux container).
    Never raises — an unmounted run is still valid for ephemeral use.
    """
    sqlite_dir = os.path.dirname(os.environ.get("VGI_WORKER_SQLITE_PATH", ""))
    if not sqlite_dir.startswith("/data"):
        return
    try:
        with open("/proc/mounts", encoding="utf-8") as fh:  # Linux container only
            mountpoints = {parts[1] for line in fh if len(parts := line.split()) > 1}
    except OSError:
        return
    if "/data" not in mountpoints and sqlite_dir not in mountpoints:
        log.warning(
            "state directory /data is not a mounted volume: the shared BoundStorage is container-local "
            "and will NOT persist across restarts or be shared across worker instances. Mount a volume "
            "at /data (the image advertises this via the 'farm.query.vgi.volumes' label)."
        )


def main() -> None:
    """Run the worker (stdio by default; pass ``--http`` for the HTTP server)."""
    _warn_if_ephemeral_state()
    ScikitBioWorker.main()


def main_http() -> None:
    """Run the worker over HTTP (injects ``--http`` into the worker CLI)."""
    _warn_if_ephemeral_state()
    argv = sys.argv[1:]
    if "--http" not in argv:
        argv = ["--http", *argv]
    sys.argv = [sys.argv[0], *argv]
    ScikitBioWorker.main()
