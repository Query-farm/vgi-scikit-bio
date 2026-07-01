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
    # should satisfy using this worker. Each pins the exact output column
    # name(s) and supplies any data inline, so the analyst must run a query and
    # its result set is deterministically comparable to the reference query.
    "vgi.agent_test_tasks": json.dumps(
        [
            {
                "name": "gc_content",
                "prompt": (
                    "Compute the GC content of the DNA sequence 'ATGCGGATTACAGG'. "
                    "Return a single row with one column named gc."
                ),
                "reference_sql": "SELECT skbio.sequence.gc_content('ATGCGGATTACAGG') AS gc",
            },
            {
                "name": "reverse_complement",
                "prompt": (
                    "Return the reverse complement of the DNA sequence 'ATGCGGATTACAGG' "
                    "as a single row with one column named rc."
                ),
                "reference_sql": "SELECT skbio.sequence.reverse_complement('ATGCGGATTACAGG') AS rc",
            },
            {
                "name": "translate",
                "prompt": (
                    "Translate the DNA sequence 'ATGCGGATTACAGGT' to its amino-acid sequence using the "
                    "standard genetic code. Return a single row with one column named protein."
                ),
                "reference_sql": "SELECT skbio.sequence.translate('ATGCGGATTACAGGT') AS protein",
            },
            {
                "name": "kmer_count",
                "prompt": (
                    "For the DNA read 'ATGCGGATTACAGG', how many distinct overlapping 3-mers occur? "
                    "Return a single row with one column named n."
                ),
                "reference_sql": (
                    "SELECT count(*) AS n FROM skbio.sequence.kmer_frequencies("
                    "(SELECT * FROM (VALUES (1, 'ATGCGGATTACAGG')) AS r(id, seq)), id := 'id', k := 3)"
                ),
            },
            {
                "name": "alpha_diversity",
                "prompt": (
                    "Here is a long feature table with columns (sample_id, feature_id, count): "
                    "(1,'a',4), (1,'b',2), (2,'a',1), (2,'b',9). For each sample, compute its Shannon "
                    "alpha diversity rounded to 4 decimal places. Return columns sample_id and shannon, "
                    "ordered by sample_id."
                ),
                "reference_sql": (
                    "SELECT sample_id, round(skbio.diversity.shannon(count), 4) AS shannon FROM "
                    "(VALUES (1,'a',4),(1,'b',2),(2,'a',1),(2,'b',9)) AS t(sample_id, feature_id, count) "
                    "GROUP BY sample_id ORDER BY sample_id"
                ),
            },
            {
                "name": "observed_richness",
                "prompt": (
                    "Given the feature counts (1,'a',4), (1,'b',0), (1,'c',3) as (sample_id, feature_id, "
                    "count) for sample 1, how many features are observed (have a non-zero count)? Return a "
                    "single row with one integer column named richness."
                ),
                "reference_sql": (
                    "SELECT skbio.diversity.observed_features(count)::BIGINT AS richness FROM "
                    "(VALUES (1,'a',4),(1,'b',0),(1,'c',3)) AS t(sample_id, feature_id, count)"
                ),
            },
            {
                "name": "beta_matrix_size",
                "prompt": (
                    "From the feature table (s1,'a',4), (s1,'b',1), (s2,'a',3), (s2,'b',2), (s3,'a',0), "
                    "(s3,'b',9) as (sample_id, feature_id, count), build the Bray-Curtis between-sample "
                    "distance matrix. How many rows does the full matrix have? Return a single row with "
                    "one column named n."
                ),
                "reference_sql": (
                    "SELECT count(*) AS n FROM skbio.diversity.beta_diversity((SELECT * FROM "
                    "(VALUES ('s1','a',4),('s1','b',1),('s2','a',3),('s2','b',2),('s3','a',0),('s3','b',9)) "
                    "AS t(sample_id, feature_id, count)))"
                ),
            },
            {
                "name": "tree_tip_count",
                "prompt": (
                    "How many tips (leaves) are in the Newick tree '((a:2,b:3):3,d:4,c:4);'? "
                    "Return a single row with one column named tips."
                ),
                "reference_sql": "SELECT skbio.tree.tip_count('((a:2,b:3):3,d:4,c:4);') AS tips",
            },
        ]
    ),
}

# Per-schema category registries (VGI413/408/409/410/411/412). Each schema
# declares an ordered list of {name, description} sections; every function is
# assigned to one via `_FUNCTION_CATEGORY` below. Categories drive the worker's
# navigation and listing sections.
_SCHEMA_CATEGORIES: dict[str, list[dict[str, str]]] = {
    "sequence": [
        {
            "name": "transforms",
            "description": "Derive a new sequence from a DNA input (complement, transcribe, translate).",
        },
        {"name": "validation", "description": "Check whether a string is a valid sequence of a given type."},
        {"name": "distance", "description": "Compare two sequences position by position."},
        {"name": "composition", "description": "Break a sequence into per-token counts (k-mers, residues)."},
    ],
    "diversity": [
        {"name": "alpha", "description": "Per-sample diversity of one community, as aggregates."},
        {"name": "beta", "description": "Between-sample community distances, as a matrix."},
    ],
    "stats": [
        {"name": "ordination", "description": "Embed samples in a low-dimensional space from a distance matrix."},
        {"name": "hypothesis-tests", "description": "Test associations and correlations over distance matrices."},
        {"name": "composition", "description": "Log-ratio transforms that move compositional data into real space."},
    ],
    "tree": [
        {"name": "construction", "description": "Build a phylogenetic tree from a distance matrix."},
        {"name": "inspection", "description": "Read properties of a tree given as a Newick string."},
    ],
}

# Function machine-name -> its schema category name (VGI411: every categorizable
# object in a schema that declares categories carries a vgi.category).
_FUNCTION_CATEGORY: dict[str, str] = {
    # sequence
    "gc_content": "transforms",
    "reverse_complement": "transforms",
    "complement": "transforms",
    "transcribe": "transforms",
    "translate": "transforms",
    "is_valid_dna": "validation",
    "is_valid_protein": "validation",
    "hamming_distance": "distance",
    "kmer_frequencies": "composition",
    "residue_frequencies": "composition",
    # diversity — the alpha metrics self-declare vgi.category = "alpha" in their
    # generated Meta (see diversity._make_alpha); only beta_diversity is manual.
    "beta_diversity": "beta",
    # stats
    "pcoa": "ordination",
    "permanova": "hypothesis-tests",
    "anosim": "hypothesis-tests",
    "mantel": "hypothesis-tests",
    "clr": "composition",
    "ilr": "composition",
    # tree
    "neighbor_joining": "construction",
    "tip_count": "inspection",
    "total_branch_length": "inspection",
}

# Per-schema metadata. Each schema carries a concept-focused description (VGI173:
# describe what the area is for, not an inventory of its objects), a descriptive
# display title (VGI124/125), keywords, and a runnable example query.
_SCHEMA_META: dict[str, dict[str, str]] = {
    "sequence": {
        "comment": "Analyze DNA, RNA, and protein sequences held in VARCHAR columns.",
        "title": "Biological Sequences",
        "keywords": json.dumps(["sequence", "dna", "rna", "protein", "kmer"]),
        "doc_llm": (
            "Analyze biological sequences — DNA, RNA, or protein — stored one per row in ordinary VARCHAR "
            "columns. Reach for this area to derive new sequences (base complementation, transcription, "
            "codon translation), measure composition (GC fraction, k-mer and single-residue profiles as "
            "long token-count tables), compare reads to a reference, or validate that a string really is a "
            "sequence of a given alphabet. Inputs are case-insensitive and whitespace-tolerant; a NULL or "
            "malformed sequence yields NULL rather than failing the query, so the functions are safe over "
            "messy real-world reads."
        ),
        "doc_md": (
            "### Biological sequences\n\n"
            "Work with DNA, RNA, and protein sequences directly in SQL — one sequence per VARCHAR cell.\n\n"
            "- **Derive** new sequences (complementation, transcription, codon translation)\n"
            "- **Measure** composition (GC fraction, k-mer and residue profiles)\n"
            "- **Compare** reads and **validate** alphabets\n\n"
            "Case-insensitive; malformed or NULL input degrades to NULL instead of erroring."
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
        "comment": "Community-ecology diversity from a long (sample, feature, count) table.",
        "title": "Community Diversity",
        "keywords": json.dumps(["diversity", "alpha", "beta", "ecology", "microbiome"]),
        "doc_llm": (
            "Quantify community diversity from a long feature table — one row per (sample, feature, count), "
            "the shape a microbiome or OTU/ASV study naturally produces. Reach here for two questions: how "
            "diverse is each individual sample (alpha diversity, computed as aggregates so a GROUP BY over "
            "the sample id yields one value per sample), and how different are samples from one another "
            "(beta diversity, computed as a between-sample distance matrix emitted in long form). The "
            "distance matrix is the input other areas consume for ordination, group tests, and tree "
            "building."
        ),
        "doc_md": (
            "### Community diversity\n\n"
            "From a long `(sample, feature, count)` feature table:\n\n"
            "- **Alpha** — per-sample diversity, as aggregates you `GROUP BY sample_id`\n"
            "- **Beta** — a between-sample distance matrix, emitted long for downstream ordination, "
            "group tests, and tree building"
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
        "comment": "Ordination, distance-matrix hypothesis tests, and compositional transforms.",
        "title": "Multivariate Statistics",
        "keywords": json.dumps(["ordination", "distance", "hypothesis test", "composition", "microbiome"]),
        "doc_llm": (
            "Multivariate analysis of distance matrices and compositional data. Reach here to embed samples "
            "in a few interpretable dimensions from a distance matrix (principal-coordinates ordination), "
            "to test whether a grouping or a second matrix explains the distances (permutational and "
            "rank-based group tests, and matrix correlation), and to move compositional feature data into "
            "ordinary real space where standard statistics apply (log-ratio transforms). Distance-matrix "
            "inputs use the long (id_1, id_2, distance) shape that the diversity area's beta-diversity "
            "matrix produces."
        ),
        "doc_md": (
            "### Multivariate statistics\n\n"
            "Over distance matrices and compositions:\n\n"
            "- **Ordination** — embed samples in low dimensions from a distance matrix\n"
            "- **Hypothesis tests** — do groups or a second matrix explain the distances?\n"
            "- **Composition** — log-ratio transforms into real space\n\n"
            "Distance inputs use the long `(id_1, id_2, distance)` shape from the beta-diversity matrix."
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
        "comment": "Build phylogenetic trees from distances and inspect Newick strings.",
        "title": "Phylogenetics",
        "keywords": json.dumps(["tree", "phylogenetics", "newick", "distance", "microbiome"]),
        "doc_llm": (
            "Build and inspect phylogenetic trees. Reach here to reconstruct an unrooted tree from a long "
            "distance matrix (returned as a Newick string), and to read properties of trees stored per row "
            "— how many tips they have, or their total branch length — with scalar functions over a Newick "
            "column. The distance-matrix input is the long (id_1, id_2, distance) shape produced by the "
            "diversity area's beta-diversity matrix."
        ),
        "doc_md": (
            "### Phylogenetics\n\n"
            "Build and inspect trees:\n\n"
            "- **Construct** an unrooted tree from a distance matrix, returned as a Newick string\n"
            "- **Inspect** Newick strings stored per row (tip counts, total branch length)\n\n"
            "Distance input uses the long `(id_1, id_2, distance)` shape from the beta-diversity matrix."
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


def _apply_discovery_tags(functions: list[type]) -> None:
    """Inject the per-function discovery tags the catalog-quality linter expects.

    ``vgi.keywords`` (a JSON array of strings) is derived mechanically from each
    function's existing Meta (display name, categories); ``vgi.category`` assigns
    the function to one of its schema's declared category sections (see
    ``_FUNCTION_CATEGORY``). The richer ``vgi.doc_llm`` / ``vgi.doc_md`` tags are
    authored per function in the implementation modules and left untouched here.
    A per-function ``vgi.title`` is deliberately not set: a mechanical title would
    just restate the machine name, and only the catalog and schemas need one.
    """
    for fn in functions:
        meta = getattr(fn, "Meta", None)
        if meta is None:
            continue
        name = getattr(meta, "name", fn.__name__)
        cats = list(getattr(meta, "categories", []) or [])
        tags = dict(getattr(meta, "tags", {}) or {})
        keywords = list(dict.fromkeys(cats or name.split("_")))
        tags.setdefault("vgi.keywords", json.dumps(keywords))
        if name in _FUNCTION_CATEGORY:
            tags.setdefault("vgi.category", _FUNCTION_CATEGORY[name])
        meta.tags = tags


_apply_discovery_tags(_FUNCTIONS)


def _build_schema(name: str, functions: list[type]) -> Schema:
    """Build a ``Schema`` from its function list, ``_SCHEMA_META``, and its categories."""
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
            "vgi.categories": json.dumps(_SCHEMA_CATEGORIES[name]),
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
