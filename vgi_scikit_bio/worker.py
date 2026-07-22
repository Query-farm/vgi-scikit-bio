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
from vgi_scikit_bio.alignment import ALIGNMENT_FUNCTIONS
from vgi_scikit_bio.composition import COMPOSITION_FUNCTIONS
from vgi_scikit_bio.distance_stats import DISTANCE_STATS_FUNCTIONS
from vgi_scikit_bio.diversity import DIVERSITY_FUNCTIONS
from vgi_scikit_bio.kmers import KMER_FUNCTIONS
from vgi_scikit_bio.ordination import ORDINATION_FUNCTIONS
from vgi_scikit_bio.phylo import PHYLO_FUNCTIONS
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
    "alignment": [*ALIGNMENT_FUNCTIONS],
    "diversity": [*DIVERSITY_FUNCTIONS, *PHYLO_FUNCTIONS],
    "stats": [*ORDINATION_FUNCTIONS, *DISTANCE_STATS_FUNCTIONS, *COMPOSITION_FUNCTIONS],
    "tree": [*TREE_FUNCTIONS],
}
_FUNCTIONS: list[type] = [fn for fns in _SCHEMA_FUNCTIONS.values() for fn in fns]

# Provenance / about link advertised on the catalog (VGI source_url).
SOURCE_URL = "https://github.com/query-farm/vgi-scikit-bio"

# Catalog-level metadata surfaced through duckdb_databases() (comment + tags).
_CATALOG_COMMENT = "scikit-bio for SQL: sequence analysis, community diversity, ordination, and phylogenetics in DuckDB"
_CATALOG_DESCRIPTION_LLM = (
    "scikit-bio for SQL — comprehensive bioinformatics in DuckDB. Analyze biological sequences "
    "(GC content, reverse complement, translation, six-frame translation, k-mer/residue composition, "
    "validation, and sequence distances); align sequence pairs (global/local, with scores and aligned "
    "strings); compute the full family of alpha-diversity metrics as aggregates and beta-diversity "
    "distance matrices (including phylogenetic Faith's PD and UniFrac); rarefy feature tables; run PCA/CA/"
    "PCoA ordination, PERMANOVA/ANOSIM/Mantel distance tests, CLR/ILR/ALR and other compositional "
    "transforms, and ANCOM/Dirichlet-multinomial differential abundance; and build and compare "
    "phylogenetic trees (neighbour joining, UPGMA, minimum evolution; Robinson-Foulds and cophenetic "
    "distances) — all as DuckDB scalar, aggregate, and table functions."
)
_CATALOG_DESCRIPTION_MD = (
    "# scikit-bio for SQL\n\n"
    "Exposes [scikit-bio](https://scikit.bio) to DuckDB/SQL as VGI functions:\n\n"
    "- **Sequence** — GC content, reverse complement, translation (incl. six-frame), composition, "
    "validation, sequence distances\n"
    "- **Alignment** — global/local pairwise alignment (scores and aligned strings)\n"
    "- **Diversity** — the full alpha-diversity metric family (aggregates), beta-diversity matrices, "
    "phylogenetic Faith's PD & UniFrac, and rarefaction\n"
    "- **Stats** — PCA/CA/PCoA ordination, PERMANOVA / ANOSIM / Mantel tests, CLR/ILR/ALR transforms, "
    "ANCOM & Dirichlet-multinomial differential abundance\n"
    "- **Tree** — neighbour joining / UPGMA / minimum evolution, Newick inspection, and tree comparison"
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
            "description": "Optimal global alignment score between two DNA sequences.",
            "sql": "SELECT skbio.alignment.align_score_nucleotide('ACTGGT', 'ACTGT') AS score",
        },
        {
            "description": "Faith's phylogenetic diversity per sample given a tree.",
            "sql": (
                "SELECT * FROM skbio.diversity.faith_pd((SELECT * FROM "
                "(VALUES ('s1','f1',1),('s1','f2',1),('s2','f3',1),('s2','f4',1)) AS t(sample_id, feature_id, count)), "
                "tree := '((f1:0.1,f2:0.2):0.3,(f3:0.15,f4:0.25):0.35);')"
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
# Analyst tasks for `vgi-lint simulate` — natural-language prompts an agent
# should satisfy using this worker, and the gate that every published object is
# actually exercised (VGI520). Each pins the exact output column name(s) and
# supplies its data inline, so the analyst must run a query and its result set is
# deterministically comparable to the reference query. Permutation p-values and
# raw ordination axis signs are deliberately never asserted — only the
# deterministic parts of those results are.
#
# The inline datasets below are shared by several tasks; each prompt restates its
# data in prose, so nothing the grader knows leaks into what the analyst is told.
_D_COUNTS = "(VALUES ('a',4),('b',2),('c',1),('d',1),('e',3),('f',7)) AS t(feature_id, count)"
_P_COUNTS = "(a,4), (b,2), (c,1), (d,1), (e,3), (f,7)"

_D_FEATURES = (
    "(VALUES ('s1','a',4),('s1','b',2),('s1','c',1),('s2','a',1),('s2','b',9),('s2','c',2)) "
    "AS t(sample_id, feature_id, count)"
)
_P_FEATURES = "(s1,a,4), (s1,b,2), (s1,c,1), (s2,a,1), (s2,b,9), (s2,c,2)"

_D_COMPOSITION = (
    "(VALUES ('s1','a',1),('s1','b',2),('s1','c',3),('s2','a',4),('s2','b',5),('s2','c',6)) "
    "AS t(sample_id, feature_id, value)"
)
_P_COMPOSITION = "(s1,a,1), (s1,b,2), (s1,c,3), (s2,a,4), (s2,b,5), (s2,c,6)"

_D_DISTANCES = (
    "(VALUES ('a','b',5),('a','c',9),('a','d',9),('b','c',10),('b','d',10),('c','d',8)) AS d(id_1, id_2, distance)"
)
_P_DISTANCES = "(a,b,5), (a,c,9), (a,d,9), (b,c,10), (b,d,10), (c,d,8)"

_D_TREE = "((f1:0.1,f2:0.2):0.3,(f3:0.15,f4:0.25):0.35);"
_D_TREE_TABLE = (
    "(VALUES ('s1','f1',1),('s1','f2',1),('s2','f3',1),('s2','f4',1),('s3','f1',1),('s3','f3',1)) "
    "AS t(sample_id, feature_id, count)"
)
_P_TREE_TABLE = "(s1,f1,1), (s1,f2,1), (s2,f3,1), (s2,f4,1), (s3,f1,1), (s3,f3,1)"

_D_GROUPED = (
    "(VALUES ('s1','a',4),('s1','b',1),('s2','a',3),('s2','b',2),('s3','a',1),('s3','b',8),"
    "('s4','a',0),('s4','b',9)) AS t(sample_id, feature_id, count)"
)
_P_GROUPED = "(s1,a,4), (s1,b,1), (s2,a,3), (s2,b,2), (s3,a,1), (s3,b,8), (s4,a,0), (s4,b,9)"
_D_GROUPS = "(VALUES ('s1','x'),('s2','x'),('s3','y'),('s4','y')) AS g(sample, grp)"
_P_GROUPS = "s1 and s2 are in group x; s3 and s4 are in group y"

_D_DIFF_COUNTS = (
    "(VALUES ('s1','b1',12),('s1','b2',11),('s2','b1',9),('s2','b2',11),('s3','b1',1),"
    "('s3','b2',11),('s4','b1',22),('s4','b2',21),('s5','b1',20),('s5','b2',22),"
    "('s6','b1',23),('s6','b2',21)) AS s(sample_id, feature_id, count)"
)
_P_DIFF_COUNTS = (
    "(s1,b1,12), (s1,b2,11), (s2,b1,9), (s2,b2,11), (s3,b1,1), (s3,b2,11), "
    "(s4,b1,22), (s4,b2,21), (s5,b1,20), (s5,b2,22), (s6,b1,23), (s6,b2,21)"
)
_D_DIFF_GROUPS = "(VALUES ('s1','x'),('s2','x'),('s3','x'),('s4','y'),('s5','y'),('s6','y')) AS g(sample, grp)"
_P_DIFF_GROUPS = "s1, s2 and s3 are in group x; s4, s5 and s6 are in group y"

_D_DIFF_INPUT = (
    f"(SELECT s.sample_id, s.feature_id, s.count, g.grp FROM {_D_DIFF_COUNTS} "
    f"JOIN {_D_DIFF_GROUPS} ON s.sample_id = g.sample)"
)
_D_GROUPED_INPUT = (
    "(SELECT b.id_1, b.id_2, b.distance, g.grp FROM skbio.diversity.beta_diversity("
    f"(SELECT * FROM {_D_GROUPED})) AS b JOIN {_D_GROUPS} ON b.id_1 = g.sample)"
)

_AGENT_TEST_TASKS: list[dict[str, str]] = [
    # --- sequence scalars -------------------------------------------------
    {
        "name": "gc_content",
        "prompt": (
            "Compute the GC content of the DNA sequence 'ATGCGGATTACAGG'. Return a single row with one column named gc."
        ),
        "reference_sql": "SELECT skbio.sequence.gc_content('ATGCGGATTACAGG') AS gc",
    },
    {
        "name": "gc_and_motif_counts",
        "prompt": (
            "For the DNA sequence 'ATGCGATGCATG', return a single row with two integer columns: "
            "gc_bases, the number of G or C bases in it, and start_codons, the number of times "
            "the subsequence 'ATG' occurs in it."
        ),
        "reference_sql": (
            "SELECT skbio.sequence.gc_frequency('ATGCGATGCATG') AS gc_bases, "
            "skbio.sequence.count_subsequence('ATGCGATGCATG', 'ATG') AS start_codons"
        ),
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
        "name": "strand_relationship",
        "prompt": (
            "Return a single row with two boolean columns for the DNA pair 'ATGC' and 'GCAT': "
            "same_strand, true if the two strings are identical, and opposite_strand, true if "
            "the second is the reverse complement of the first."
        ),
        "reference_sql": (
            "SELECT ('ATGC' = 'GCAT') AS same_strand, "
            "skbio.sequence.is_reverse_complement('ATGC', 'GCAT') AS opposite_strand"
        ),
    },
    {
        "name": "transcription_round_trip",
        "prompt": (
            "Starting from the DNA sequence 'ATGCGGATTACAGG', return a single row with three "
            "columns: comp, its base complement (not reversed); rna, its RNA transcript; and "
            "back_to_dna, the DNA recovered by reverse-transcribing that RNA transcript."
        ),
        "reference_sql": (
            "SELECT skbio.sequence.complement('ATGCGGATTACAGG') AS comp, "
            "skbio.sequence.transcribe('ATGCGGATTACAGG') AS rna, "
            "skbio.sequence.reverse_transcribe(skbio.sequence.transcribe('ATGCGGATTACAGG')) AS back_to_dna"
        ),
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
        "name": "sequence_quality_flags",
        "prompt": (
            "Quality-check the string 'AC-GTN'. Return a single row with five columns: valid_dna "
            "(is it a valid IUPAC DNA sequence), valid_protein (is it a valid IUPAC protein "
            "sequence), gapped (does it contain gap characters), ambiguous (does it contain "
            "degenerate/ambiguity codes), and ungapped (the string with gap characters removed)."
        ),
        "reference_sql": (
            "SELECT skbio.sequence.is_valid_dna('AC-GTN') AS valid_dna, "
            "skbio.sequence.is_valid_protein('AC-GTN') AS valid_protein, "
            "skbio.sequence.has_gaps('AC-GTN') AS gapped, "
            "skbio.sequence.has_degenerates('AC-GTN') AS ambiguous, "
            "skbio.sequence.degap('AC-GTN') AS ungapped"
        ),
    },
    {
        "name": "read_vs_reference",
        "prompt": (
            "Compare the read 'ACGTACGT' against the reference 'ACGAACGT' (same length). Return "
            "a single row with three columns: matches, the number of positions where they agree; "
            "mismatches, the number of positions where they differ; and hamming, the Hamming "
            "distance between them rounded to 4 decimal places."
        ),
        "reference_sql": (
            "SELECT skbio.sequence.match_count('ACGTACGT', 'ACGAACGT') AS matches, "
            "skbio.sequence.mismatch_count('ACGTACGT', 'ACGAACGT') AS mismatches, "
            "round(skbio.sequence.hamming_distance('ACGTACGT', 'ACGAACGT'), 4) AS hamming"
        ),
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
        "name": "base_composition",
        "prompt": (
            "Break the DNA read 'ATGCGGATTACAGG' down into its per-base counts. Return one row "
            "per distinct base with columns residue and count, ordered by residue."
        ),
        "reference_sql": (
            "SELECT residue, count FROM skbio.sequence.residue_frequencies("
            "(SELECT * FROM (VALUES ('ATGCGGATTACAGG')) AS r(seq))) ORDER BY residue"
        ),
    },
    {
        "name": "six_frame_translation",
        "prompt": (
            "Translate the DNA read 'ATGCGGATTACAGG' in all six reading frames. Return one row "
            "per frame with columns frame and protein, ordered by frame."
        ),
        "reference_sql": (
            "SELECT frame, protein FROM skbio.sequence.translate_six_frames("
            "(SELECT * FROM (VALUES ('ATGCGGATTACAGG')) AS r(seq))) ORDER BY frame"
        ),
    },
    # --- alignment --------------------------------------------------------
    {
        "name": "alignment_score",
        "prompt": (
            "Compute the optimal global alignment score between the DNA sequences 'ACTGGT' and "
            "'ACTGT'. Return a single row with one column named score."
        ),
        "reference_sql": "SELECT skbio.alignment.align_score_nucleotide('ACTGGT', 'ACTGT') AS score",
    },
    {
        "name": "protein_alignment_score",
        "prompt": (
            "Compute the optimal global alignment score between the protein sequences 'MRITMK' "
            "and 'MRIMK'. Return a single row with one column named score."
        ),
        "reference_sql": "SELECT skbio.alignment.align_score_protein('MRITMK', 'MRIMK') AS score",
    },
    {
        "name": "pairwise_dna_alignment",
        "prompt": (
            "Globally align the DNA read 'ACTGT' against the reference 'ACTGGT' and show the "
            "actual alignment. Return a single row with three columns: aligned_1 (the reference "
            "with alignment gaps), aligned_2 (the read with alignment gaps), and score."
        ),
        "reference_sql": (
            "SELECT aligned_1, aligned_2, score FROM skbio.alignment.pairwise_align_nucleotide("
            "(SELECT * FROM (VALUES ('ACTGGT', 'ACTGT')) AS p(ref, read)))"
        ),
    },
    {
        "name": "pairwise_protein_alignment",
        "prompt": (
            "Globally align the protein sequence 'MRIMK' against the reference 'MRITMK' and show "
            "the actual alignment. Return a single row with three columns: aligned_1 (the "
            "reference with alignment gaps), aligned_2 (the query with alignment gaps), and score."
        ),
        "reference_sql": (
            "SELECT aligned_1, aligned_2, score FROM skbio.alignment.pairwise_align_protein("
            "(SELECT * FROM (VALUES ('MRITMK', 'MRIMK')) AS p(ref, query)))"
        ),
    },
    # --- alpha diversity --------------------------------------------------
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
        "name": "richness_estimators",
        "prompt": (
            f"One sample has the feature counts {_P_COUNTS} as (feature_id, count). Estimate its "
            "total richness, including features it may have missed. Return a single row with six "
            "columns rounded to 4 decimal places: chao1, ace, singletons (features seen exactly "
            "once), doubletons (features seen exactly twice), chao1_lower and chao1_upper (the "
            "lower and upper bounds of the Chao1 confidence interval)."
        ),
        "reference_sql": (
            "SELECT round(skbio.diversity.chao1(count), 4) AS chao1, "
            "round(skbio.diversity.ace(count), 4) AS ace, "
            "round(skbio.diversity.singles(count), 4) AS singletons, "
            "round(skbio.diversity.doubles(count), 4) AS doubletons, "
            "round(skbio.diversity.chao1_ci(count)[1], 4) AS chao1_lower, "
            "round(skbio.diversity.chao1_ci(count)[2], 4) AS chao1_upper "
            f"FROM {_D_COUNTS}"
        ),
    },
    {
        "name": "observed_singles_doubles_triple",
        "prompt": (
            f"One sample has the feature counts {_P_COUNTS} as (feature_id, count). Return the "
            "OSD triple — observed richness, singletons, doubletons — as a single row with three "
            "columns named observed, singletons and doubletons."
        ),
        "reference_sql": (
            "SELECT skbio.diversity.osd(count)[1] AS observed, "
            "skbio.diversity.osd(count)[2] AS singletons, "
            "skbio.diversity.osd(count)[3] AS doubletons "
            f"FROM {_D_COUNTS}"
        ),
    },
    {
        "name": "evenness_metrics",
        "prompt": (
            f"One sample has the feature counts {_P_COUNTS} as (feature_id, count). How evenly is "
            "abundance spread across its features? Return a single row with five columns rounded "
            "to 4 decimal places: pielou, heip, simpson_e, mcintosh_e and gini."
        ),
        "reference_sql": (
            "SELECT round(skbio.diversity.pielou_evenness(count), 4) AS pielou, "
            "round(skbio.diversity.heip_evenness(count), 4) AS heip, "
            "round(skbio.diversity.simpson_e(count), 4) AS simpson_e, "
            "round(skbio.diversity.mcintosh_e(count), 4) AS mcintosh_e, "
            "round(skbio.diversity.gini_index(count), 4) AS gini "
            f"FROM {_D_COUNTS}"
        ),
    },
    {
        "name": "dominance_metrics",
        "prompt": (
            f"One sample has the feature counts {_P_COUNTS} as (feature_id, count). Measure how "
            "much the community is dominated by its commonest features. Return a single row with "
            "six columns rounded to 4 decimal places: dominance, simpson, simpson_d, inv_simpson, "
            "enspie and berger_parker."
        ),
        "reference_sql": (
            "SELECT round(skbio.diversity.dominance(count), 4) AS dominance, "
            "round(skbio.diversity.simpson(count), 4) AS simpson, "
            "round(skbio.diversity.simpson_d(count), 4) AS simpson_d, "
            "round(skbio.diversity.inv_simpson(count), 4) AS inv_simpson, "
            "round(skbio.diversity.enspie(count), 4) AS enspie, "
            "round(skbio.diversity.berger_parker_d(count), 4) AS berger_parker "
            f"FROM {_D_COUNTS}"
        ),
    },
    {
        "name": "mcintosh_and_strong_dominance",
        "prompt": (
            f"One sample has the feature counts {_P_COUNTS} as (feature_id, count). Return a "
            "single row with two columns rounded to 4 decimal places: mcintosh_d (McIntosh's "
            "dominance index) and strong (Strong's dominance index)."
        ),
        "reference_sql": (
            "SELECT round(skbio.diversity.mcintosh_d(count), 4) AS mcintosh_d, "
            "round(skbio.diversity.strong(count), 4) AS strong "
            f"FROM {_D_COUNTS}"
        ),
    },
    {
        "name": "entropy_metrics",
        "prompt": (
            f"One sample has the feature counts {_P_COUNTS} as (feature_id, count). Return its "
            "entropy-family diversity as a single row with two columns rounded to 4 decimal "
            "places: shannon and brillouin."
        ),
        "reference_sql": (
            "SELECT round(skbio.diversity.shannon(count), 4) AS shannon, "
            "round(skbio.diversity.brillouin_d(count), 4) AS brillouin "
            f"FROM {_D_COUNTS}"
        ),
    },
    {
        "name": "diversity_orders",
        "prompt": (
            f"One sample has the feature counts {_P_COUNTS} as (feature_id, count). Compute the "
            "three diversity metrics that take a diversity order q, all at q = 2. Return a single "
            "row with three columns rounded to 4 decimal places: hill, renyi and tsallis."
        ),
        "reference_sql": (
            "SELECT round(skbio.diversity.hill(count, q := 2), 4) AS hill, "
            "round(skbio.diversity.renyi(count, q := 2), 4) AS renyi, "
            "round(skbio.diversity.tsallis(count, q := 2), 4) AS tsallis "
            f"FROM {_D_COUNTS}"
        ),
    },
    {
        "name": "richness_indices",
        "prompt": (
            f"One sample has the feature counts {_P_COUNTS} as (feature_id, count). Return the "
            "sample-size-corrected richness indices as a single row with four columns rounded to "
            "4 decimal places: margalef, menhinick, fisher_alpha and kempton_taylor."
        ),
        "reference_sql": (
            "SELECT round(skbio.diversity.margalef(count), 4) AS margalef, "
            "round(skbio.diversity.menhinick(count), 4) AS menhinick, "
            "round(skbio.diversity.fisher_alpha(count), 4) AS fisher_alpha, "
            "round(skbio.diversity.kempton_taylor_q(count), 4) AS kempton_taylor "
            f"FROM {_D_COUNTS}"
        ),
    },
    {
        "name": "coverage_metrics",
        "prompt": (
            f"One sample has the feature counts {_P_COUNTS} as (feature_id, count). How much of "
            "the community has been observed? Return a single row with four columns rounded to 4 "
            "decimal places: goods_coverage, robbins, esty_lower and esty_upper (the lower and "
            "upper bounds of Esty's coverage confidence interval)."
        ),
        "reference_sql": (
            "SELECT round(skbio.diversity.goods_coverage(count), 4) AS goods_coverage, "
            "round(skbio.diversity.robbins(count), 4) AS robbins, "
            "round(skbio.diversity.esty_ci(count)[1], 4) AS esty_lower, "
            "round(skbio.diversity.esty_ci(count)[2], 4) AS esty_upper "
            f"FROM {_D_COUNTS}"
        ),
    },
    # --- beta / phylogenetic diversity ------------------------------------
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
        "name": "faith_pd",
        "prompt": (
            f"Given the feature table {_P_TREE_TABLE} as (sample_id, feature_id, count) and the "
            f"tree '{_D_TREE}', compute Faith's phylogenetic diversity of each sample, rounded to "
            "4 decimals. Return columns sample_id and faith_pd, ordered by sample_id."
        ),
        "reference_sql": (
            "SELECT sample_id, round(faith_pd, 4) AS faith_pd FROM skbio.diversity.faith_pd("
            f"(SELECT * FROM {_D_TREE_TABLE}), tree := '{_D_TREE}') ORDER BY sample_id"
        ),
    },
    {
        "name": "unifrac_distances",
        "prompt": (
            f"Given the feature table {_P_TREE_TABLE} as (sample_id, feature_id, count) and the "
            f"tree '{_D_TREE}', compute the unweighted UniFrac distance between every distinct "
            "pair of samples (exclude the diagonal and report each unordered pair once, with the "
            "alphabetically smaller id first). Return columns id_1, id_2 and distance rounded to "
            "4 decimals, ordered by id_1 then id_2."
        ),
        "reference_sql": (
            "SELECT id_1, id_2, round(distance, 4) AS distance FROM skbio.diversity.unifrac("
            f"(SELECT * FROM {_D_TREE_TABLE}), tree := '{_D_TREE}') "
            "WHERE id_1 < id_2 ORDER BY id_1, id_2"
        ),
    },
    {
        "name": "rarefaction",
        "prompt": (
            "Rarefy the feature table (s1,'a',4), (s1,'b',2), (s1,'c',6), (s2,'a',10), (s2,'b',5), "
            "(s2,'c',5) as (sample_id, feature_id, count) to a common depth of 8 counts per "
            "sample, using the default seed. Return the rarefied table with columns sample_id, "
            "feature_id and count, ordered by sample_id then feature_id."
        ),
        "reference_sql": (
            "SELECT sample_id, feature_id, count FROM skbio.diversity.subsample_counts((SELECT * FROM "
            "(VALUES ('s1','a',4),('s1','b',2),('s1','c',6),('s2','a',10),('s2','b',5),('s2','c',5)) "
            "AS t(sample_id, feature_id, count)), depth := 8) ORDER BY sample_id, feature_id"
        ),
    },
    # --- ordination -------------------------------------------------------
    {
        "name": "pcoa_embedding",
        "prompt": (
            f"Given the feature table {_P_FEATURES} as (sample_id, feature_id, count), build the "
            "Bray-Curtis distance matrix and embed the samples on 2 principal coordinate axes. "
            "Return columns sample_id and pc_1 rounded to 4 decimals, ordered by sample_id."
        ),
        "reference_sql": (
            "SELECT sample_id, round(pc_1, 4) AS pc_1 FROM skbio.stats.pcoa("
            f"(SELECT * FROM skbio.diversity.beta_diversity((SELECT * FROM {_D_FEATURES}))), "
            "n_components := 2) ORDER BY sample_id"
        ),
    },
    {
        "name": "pca_scores",
        "prompt": (
            f"Given the long feature table {_P_FEATURES} as (sample_id, feature_id, value), run a "
            "principal components analysis directly on the feature values (no distance matrix) "
            "with 2 components. Return columns sample_id and pc_1 rounded to 4 decimals, ordered "
            "by sample_id."
        ),
        "reference_sql": (
            "SELECT sample_id, round(pc_1, 4) AS pc_1 FROM skbio.stats.pca("
            f"(SELECT * FROM {_D_FEATURES}), n_components := 2) ORDER BY sample_id"
        ),
    },
    {
        "name": "correspondence_analysis",
        "prompt": (
            f"Given the long count table {_P_FEATURES} as (sample_id, feature_id, value), run a "
            "correspondence analysis with 2 axes. Return columns sample_id and ca_1 rounded to 4 "
            "decimals, ordered by sample_id."
        ),
        "reference_sql": (
            "SELECT sample_id, round(ca_1, 4) AS ca_1 FROM skbio.stats.ca("
            f"(SELECT * FROM {_D_FEATURES}), n_components := 2) ORDER BY sample_id"
        ),
    },
    # --- distance-matrix hypothesis tests ---------------------------------
    {
        "name": "permanova_statistic",
        "prompt": (
            f"Given the feature table {_P_GROUPED} as (sample_id, feature_id, count), where "
            f"{_P_GROUPS}, test with PERMANOVA whether the grouping explains the Bray-Curtis "
            "between-sample distances. Return a single row with two columns: pseudo_f, the test "
            "statistic rounded to 4 decimals, and n_groups, the number of distinct groups. Do not "
            "report the p-value (it is permutation-based and not reproducible)."
        ),
        "reference_sql": (
            "SELECT round(test_statistic, 4) AS pseudo_f, number_of_groups AS n_groups "
            f"FROM skbio.stats.permanova({_D_GROUPED_INPUT})"
        ),
    },
    {
        "name": "anosim_statistic",
        "prompt": (
            f"Given the feature table {_P_GROUPED} as (sample_id, feature_id, count), where "
            f"{_P_GROUPS}, run an ANOSIM test on the Bray-Curtis between-sample distances. Return "
            "a single row with two columns: r_statistic, the ANOSIM R rounded to 4 decimals, and "
            "sample_size, the number of samples. Do not report the p-value (it is "
            "permutation-based and not reproducible)."
        ),
        "reference_sql": (
            f"SELECT round(test_statistic, 4) AS r_statistic, sample_size FROM skbio.stats.anosim({_D_GROUPED_INPUT})"
        ),
    },
    {
        "name": "mantel_correlation",
        "prompt": (
            "Two distance matrices measured over the same three samples are given as "
            "(id_1, id_2, distance_x, distance_y): (a,b,0.5,0.4), (a,c,0.7,0.9), (b,c,0.6,0.5). "
            "Run a Mantel test between them. Return a single row with two columns: correlation, "
            "the Mantel correlation coefficient rounded to 4 decimals, and n, the number of "
            "samples compared. Do not report the p-value (it is permutation-based and not "
            "reproducible)."
        ),
        "reference_sql": (
            "SELECT round(correlation, 4) AS correlation, n FROM skbio.stats.mantel((SELECT * FROM "
            "(VALUES ('a','b',0.5,0.4),('a','c',0.7,0.9),('b','c',0.6,0.5)) "
            "AS d(id_1, id_2, distance_x, distance_y)))"
        ),
    },
    # --- compositional transforms -----------------------------------------
    {
        "name": "clr_round_trip",
        "prompt": (
            f"Given the long feature table {_P_COMPOSITION} as (sample_id, feature_id, value), "
            "apply the centred log-ratio transform and then invert it, so the result is back in "
            "proportions. Return columns sample_id, feature and value rounded to 4 decimals, "
            "ordered by sample_id then feature."
        ),
        "reference_sql": (
            "SELECT sample_id, feature, round(value, 4) AS value FROM skbio.stats.clr_inv("
            f"(SELECT * FROM skbio.stats.clr((SELECT * FROM {_D_COMPOSITION})))) "
            "ORDER BY sample_id, feature"
        ),
    },
    {
        "name": "ilr_round_trip",
        "prompt": (
            f"Given the long feature table {_P_COMPOSITION} as (sample_id, feature_id, value), "
            "apply the isometric log-ratio transform and then invert it back to a composition. "
            "Return columns sample_id, feature and value rounded to 4 decimals, ordered by "
            "sample_id then feature."
        ),
        "reference_sql": (
            "SELECT sample_id, feature, round(value, 4) AS value FROM skbio.stats.ilr_inv("
            f"(SELECT * FROM skbio.stats.ilr((SELECT * FROM {_D_COMPOSITION})))) "
            "ORDER BY sample_id, feature"
        ),
    },
    {
        "name": "alr_round_trip",
        "prompt": (
            f"Given the long feature table {_P_COMPOSITION} as (sample_id, feature_id, value), "
            "apply the additive log-ratio transform against the default reference part and then "
            "invert it back to a composition. Return columns sample_id, feature and value rounded "
            "to 4 decimals, ordered by sample_id then feature."
        ),
        "reference_sql": (
            "SELECT sample_id, feature, round(value, 4) AS value FROM skbio.stats.alr_inv("
            f"(SELECT * FROM skbio.stats.alr((SELECT * FROM {_D_COMPOSITION})))) "
            "ORDER BY sample_id, feature"
        ),
    },
    {
        "name": "normalise_composition",
        "prompt": (
            f"Given the long feature table {_P_COMPOSITION} as (sample_id, feature_id, value), "
            "rescale each sample's parts so they sum to 1. Return columns sample_id, feature_id "
            "and proportion rounded to 4 decimals, ordered by sample_id then feature_id."
        ),
        "reference_sql": (
            "SELECT sample_id, feature_id, round(proportion, 4) AS proportion "
            f"FROM skbio.stats.closure((SELECT * FROM {_D_COMPOSITION})) "
            "ORDER BY sample_id, feature_id"
        ),
    },
    {
        "name": "centre_composition",
        "prompt": (
            f"Given the long feature table {_P_COMPOSITION} as (sample_id, feature_id, value), "
            "centre each sample's composition on the dataset's geometric mean. Return columns "
            "sample_id, feature_id and centered rounded to 4 decimals, ordered by sample_id then "
            "feature_id."
        ),
        "reference_sql": (
            "SELECT sample_id, feature_id, round(centered, 4) AS centered "
            f"FROM skbio.stats.centralize((SELECT * FROM {_D_COMPOSITION})) "
            "ORDER BY sample_id, feature_id"
        ),
    },
    {
        "name": "sparse_composition_handling",
        "prompt": (
            "A sparse feature table is given as (sample_id, feature_id, value): (s1,a,0), "
            "(s1,b,2), (s1,c,3), (s2,a,4), (s2,b,0), (s2,c,6). Replace its zeros with small "
            "positive values by multiplicative replacement, keeping every non-zero ratio intact. "
            "Return columns sample_id, feature_id and value rounded to 4 decimals, ordered by "
            "sample_id then feature_id."
        ),
        "reference_sql": (
            "SELECT sample_id, feature_id, round(value, 4) AS value "
            "FROM skbio.stats.multi_replace((SELECT * FROM "
            "(VALUES ('s1','a',0),('s1','b',2),('s1','c',3),('s2','a',4),('s2','b',0),('s2','c',6)) "
            "AS t(sample_id, feature_id, value))) ORDER BY sample_id, feature_id"
        ),
    },
    {
        "name": "robust_clr",
        "prompt": (
            "A sparse feature table is given as (sample_id, feature_id, value): (s1,a,0), "
            "(s1,b,2), (s1,c,3), (s2,a,4), (s2,b,0), (s2,c,6). Apply the robust centred log-ratio "
            "transform, which handles the zeros without a pseudocount. Return columns sample_id, "
            "feature_id and rclr rounded to 4 decimals, ordered by sample_id then feature_id."
        ),
        "reference_sql": (
            "SELECT sample_id, feature_id, round(rclr, 4) AS rclr "
            "FROM skbio.stats.rclr((SELECT * FROM "
            "(VALUES ('s1','a',0),('s1','b',2),('s1','c',3),('s2','a',4),('s2','b',0),('s2','c',6)) "
            "AS t(sample_id, feature_id, value))) ORDER BY sample_id, feature_id"
        ),
    },
    {
        "name": "power_transform",
        "prompt": (
            f"Given the long feature table {_P_COMPOSITION} as (sample_id, feature_id, value), "
            "apply the compositional power transform with an exponent of 2 (square each part of "
            "the closed composition and re-close it). Return columns sample_id, feature_id and "
            "value rounded to 4 decimals, ordered by sample_id then feature_id."
        ),
        "reference_sql": (
            "SELECT sample_id, feature_id, round(value, 4) AS value "
            f"FROM skbio.stats.power((SELECT * FROM {_D_COMPOSITION}), power := 2.0) "
            "ORDER BY sample_id, feature_id"
        ),
    },
    {
        "name": "feature_association",
        "prompt": (
            "Given the long feature table (s1,a,1), (s1,b,2), (s1,c,3), (s2,a,4), (s2,b,5), "
            "(s2,c,6), (s3,a,2), (s3,b,1), (s3,c,7) as (sample_id, feature_id, value), compute "
            "the variance of the log-ratio between every distinct pair of features (report each "
            "unordered pair once, with the alphabetically smaller feature first). Return columns "
            "feature_1, feature_2 and vlr rounded to 4 decimals, ordered by feature_1 then "
            "feature_2."
        ),
        "reference_sql": (
            "SELECT feature_1, feature_2, round(vlr, 4) AS vlr FROM skbio.stats.pairwise_vlr("
            "(SELECT * FROM "
            "(VALUES ('s1','a',1),('s1','b',2),('s1','c',3),('s2','a',4),('s2','b',5),('s2','c',6),"
            "('s3','a',2),('s3','b',1),('s3','c',7)) AS t(sample_id, feature_id, value))) "
            "WHERE feature_1 < feature_2 ORDER BY feature_1, feature_2"
        ),
    },
    {
        "name": "ancom_differential_abundance",
        "prompt": (
            f"A two-group count table is given as (sample_id, feature_id, count): {_P_DIFF_COUNTS}, "
            f"where {_P_DIFF_GROUPS}. Run the ANCOM differential-abundance test. Return one row "
            "per feature with columns feature_id, w and significant, ordered by feature_id."
        ),
        "reference_sql": (
            f"SELECT feature_id, w, significant FROM skbio.stats.ancom({_D_DIFF_INPUT}) ORDER BY feature_id"
        ),
    },
    {
        "name": "dirmult_differential_abundance",
        "prompt": (
            f"A two-group count table is given as (sample_id, feature_id, count): {_P_DIFF_COUNTS}, "
            f"where {_P_DIFF_GROUPS}. Run the Dirichlet-multinomial t-test for differential "
            "abundance. Return one row per feature with columns feature_id, log2_fold_change "
            "rounded to 4 decimals, and significant, ordered by feature_id."
        ),
        "reference_sql": (
            "SELECT feature_id, round(log2_fold_change, 4) AS log2_fold_change, significant "
            f"FROM skbio.stats.dirmult_ttest({_D_DIFF_INPUT}) ORDER BY feature_id"
        ),
    },
    # --- trees ------------------------------------------------------------
    {
        "name": "tree_tip_count",
        "prompt": (
            "How many tips (leaves) are in the Newick tree '((a:2,b:3):3,d:4,c:4);'? "
            "Return a single row with one column named tips."
        ),
        "reference_sql": "SELECT skbio.tree.tip_count('((a:2,b:3):3,d:4,c:4);') AS tips",
    },
    {
        "name": "neighbor_joining_tree",
        "prompt": (
            f"Build a neighbour-joining tree from the distance matrix {_P_DISTANCES} given as "
            "(id_1, id_2, distance). Return a single row with two columns: newick, the tree in "
            "Newick format, and tips, the number of tips in it."
        ),
        "reference_sql": (
            "SELECT newick, skbio.tree.tip_count(newick) AS tips "
            f"FROM skbio.tree.neighbor_joining((SELECT * FROM {_D_DISTANCES}))"
        ),
    },
    {
        "name": "upgma_tree_shape",
        "prompt": (
            f"Build a UPGMA tree from the distance matrix {_P_DISTANCES} given as "
            "(id_1, id_2, distance), then measure it. Return a single row with two columns "
            "rounded to 4 decimal places: total_length, the sum of all its branch lengths, and "
            "height, its maximum root-to-tip distance."
        ),
        "reference_sql": (
            "SELECT round(skbio.tree.total_branch_length(newick), 4) AS total_length, "
            "round(skbio.tree.tree_height(newick), 4) AS height "
            f"FROM skbio.tree.upgma((SELECT * FROM {_D_DISTANCES}))"
        ),
    },
    {
        "name": "minimum_evolution_topologies_agree",
        "prompt": (
            f"From the distance matrix {_P_DISTANCES} given as (id_1, id_2, distance), build one "
            "tree with greedy minimum evolution and another with balanced minimum evolution, then "
            "compare them. Return a single row with one column named rf_distance holding the "
            "Robinson-Foulds distance between the two topologies."
        ),
        "reference_sql": (
            "SELECT skbio.tree.robinson_foulds(g.newick, b.newick) AS rf_distance FROM "
            f"skbio.tree.gme((SELECT * FROM {_D_DISTANCES})) AS g, "
            f"skbio.tree.bme((SELECT * FROM {_D_DISTANCES})) AS b"
        ),
    },
    {
        "name": "branch_length_aware_tree_comparison",
        "prompt": (
            "Compare the two Newick trees '((a:1,b:1):1,(c:1,d:1):1);' and "
            "'((a:1,c:1):5,(b:1,d:1):5);' in the two branch-length-aware ways. Return a single "
            "row with two columns rounded to 4 decimal places: wrf, their weighted "
            "Robinson-Foulds distance, and cophenetic, their cophenetic distance."
        ),
        "reference_sql": (
            "SELECT round(skbio.tree.weighted_robinson_foulds("
            "'((a:1,b:1):1,(c:1,d:1):1);', '((a:1,c:1):5,(b:1,d:1):5);'), 4) AS wrf, "
            "round(skbio.tree.cophenetic_distance("
            "'((a:1,b:1):1,(c:1,d:1):1);', '((a:1,c:1):5,(b:1,d:1):5);'), 4) AS cophenetic"
        ),
    },
]


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
            "alignment",
            "diversity",
            "unifrac",
            "ordination",
            "phylogenetics",
            "microbiome",
            "differential abundance",
            "composition",
        ]
    ),
    "vgi.executable_examples": _CATALOG_EXECUTABLE_EXAMPLES,
    # Analyst tasks for `vgi-lint simulate` -- see _AGENT_TEST_TASKS above.
    "vgi.agent_test_tasks": json.dumps(_AGENT_TEST_TASKS),
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
    "alignment": [
        {"name": "score", "description": "Optimal alignment score between two sequences."},
        {"name": "pairwise", "description": "Aligned sequence pair (aligned strings + score)."},
    ],
    "diversity": [
        {"name": "alpha", "description": "Per-sample diversity of one community, as aggregates."},
        {"name": "beta", "description": "Between-sample community distances, as a matrix."},
        {"name": "phylogenetic", "description": "Tree-aware diversity (Faith's PD, UniFrac)."},
        {"name": "preprocessing", "description": "Prepare a feature table (rarefaction)."},
    ],
    "stats": [
        {"name": "ordination", "description": "Embed samples in a low-dimensional space from a distance matrix."},
        {"name": "hypothesis-tests", "description": "Test associations and correlations over distance matrices."},
        {"name": "composition", "description": "Log-ratio transforms that move compositional data into real space."},
    ],
    "tree": [
        {"name": "construction", "description": "Build a phylogenetic tree from a distance matrix."},
        {"name": "inspection", "description": "Read properties of a tree given as a Newick string."},
        {"name": "comparison", "description": "Compare two trees given as Newick strings."},
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
    # tree functions and the alignment schema self-declare vgi.category in their Meta.
}

# Per-schema metadata. Each schema carries a concept-focused description (VGI173:
# describe what the area is for, not an inventory of its objects), a descriptive
# display title (VGI124/125), keywords, and a runnable example query.
_SCHEMA_META: dict[str, dict[str, str]] = {
    "sequence": {
        "comment": "Analyze DNA, RNA, and protein sequences held in `VARCHAR` columns.",
        "title": "Biological Sequences",
        "keywords": json.dumps(["sequence", "dna", "rna", "protein", "kmer"]),
        "doc_llm": (
            "Analyze biological sequences — DNA, RNA, or protein — stored one per row in ordinary `VARCHAR` "
            "columns. Reach for this area to derive new sequences (base complementation, transcription, "
            "codon translation), measure composition (GC fraction, k-mer and single-residue profiles as "
            "long token-count tables), compare reads to a reference, or validate that a string really is a "
            "sequence of a given alphabet. Inputs are case-insensitive and whitespace-tolerant; a NULL or "
            "malformed sequence yields NULL rather than failing the query, so the functions are safe over "
            "messy real-world reads."
        ),
        "doc_md": (
            "### Biological sequences\n\n"
            "Work with DNA, RNA, and protein sequences directly in SQL — one sequence per `VARCHAR` cell.\n\n"
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
    "alignment": {
        "comment": "Pairwise sequence alignment: scores and aligned strings.",
        "title": "Sequence Alignment",
        "keywords": json.dumps(["alignment", "pairwise", "needleman-wunsch", "smith-waterman", "dna"]),
        "doc_llm": (
            "Pairwise alignment of biological sequences. Reach here to score how similar two DNA or protein "
            "sequences are (optimal global-alignment score, per row) or to produce the actual alignment — "
            "the two sequences padded with gaps plus the score and aligned length — in global "
            "(Needleman-Wunsch) or local (Smith-Waterman) mode. Sequences are plain `VARCHAR` columns; a NULL "
            "or unparseable pair yields NULL rather than failing the query."
        ),
        "doc_md": (
            "### Sequence alignment\n\n"
            "Align pairs of DNA or protein sequences:\n\n"
            "- **Score** — optimal global-alignment score per row (a similarity measure)\n"
            "- **Pairwise** — the aligned strings (gaps as `-`) plus score and length, global or local\n\n"
            "Inputs are plain `VARCHAR` columns; malformed pairs degrade to NULL."
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "Global alignment score of two DNA sequences",
                    "sql": "SELECT skbio.alignment.align_score_nucleotide('ACTGGT', 'ACTGT')",
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
            "the shape a microbiome or OTU/ASV study naturally produces. Reach here to measure how diverse "
            "each individual sample is (alpha diversity, the full scikit-bio metric family computed as "
            "aggregates so a GROUP BY over the sample id yields one value per sample), how different samples "
            "are from one another (beta diversity, a between-sample distance matrix emitted in long form), "
            "and their tree-aware counterparts (Faith's phylogenetic diversity and UniFrac, which take a "
            "Newick tree argument). It also rarefies feature tables to a common depth. The distance matrix "
            "is the input other areas consume for ordination, group tests, and tree building."
        ),
        "doc_md": (
            "### Community diversity\n\n"
            "From a long `(sample, feature, count)` feature table:\n\n"
            "- **Alpha** — per-sample diversity (the full metric family), as aggregates you "
            "`GROUP BY sample_id`\n"
            "- **Beta** — a between-sample distance matrix, emitted long for downstream ordination, "
            "group tests, and tree building\n"
            "- **Phylogenetic** — Faith's PD and UniFrac (given a Newick tree)\n"
            "- **Preprocessing** — rarefy each sample to a common depth"
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
            "in a few interpretable dimensions — either from a distance matrix (principal-coordinates "
            "ordination) or directly from a feature table (principal components and correspondence "
            "analysis); to test whether a grouping or a second matrix explains between-sample distances "
            "(permutational and rank-based group tests, and matrix correlation); to move compositional "
            "feature data into ordinary real space with log-ratio transforms and their inverses; and to "
            "find differentially abundant features between groups (ANCOM and a Dirichlet-multinomial "
            "t-test). Distance-matrix inputs use the long (id_1, id_2, distance) shape the diversity area's "
            "beta-diversity matrix produces."
        ),
        "doc_md": (
            "### Multivariate statistics\n\n"
            "Over distance matrices, feature tables, and compositions:\n\n"
            "- **Ordination** — embed samples in low dimensions (from a distance matrix or feature table)\n"
            "- **Hypothesis tests** — do groups or a second matrix explain the distances?\n"
            "- **Composition** — log-ratio transforms (and inverses) into real space\n"
            "- **Differential abundance** — which features differ between groups (ANCOM, Dirichlet-"
            "multinomial)\n\n"
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

    ``vgi.example_queries`` re-publishes each function's ``Meta.examples`` **with
    their descriptions**. Both are carriers of the same examples, but they are not
    equivalent: the native ``duckdb_functions().examples`` column that
    ``Meta.examples`` lands in is a bare ``VARCHAR[]`` of SQL strings, so the
    per-example prose is lost in transit. The tag is JSON, so it survives -- and
    an example's description is the part that says *why* you would run the query.
    Derived here rather than hand-written so the two carriers cannot drift.
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
        examples = list(getattr(meta, "examples", []) or [])
        if examples:
            tags.setdefault(
                "vgi.example_queries",
                json.dumps([{"description": ex.description, "sql": ex.sql} for ex in examples]),
            )
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
