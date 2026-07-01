"""scikit-bio as a VGI worker: sequences, diversity, ordination, and phylogeny for DuckDB/SQL.

The implementation is split by scikit-bio area so each module stays focused:

- ``sequence``    -- nucleotide/protein scalar functions (GC content, translation, ...)
- ``kmers``       -- k-mer and residue composition as table functions
- ``diversity``   -- alpha-diversity aggregates and beta-diversity distance matrices
- ``ordination``  -- principal coordinates analysis (PCoA) over a distance matrix
- ``distance_stats`` -- PERMANOVA / ANOSIM / Mantel tests over distance matrices
- ``composition`` -- compositional transforms (CLR, ILR) over feature tables
- ``tree``        -- neighbour-joining tree construction and Newick inspection

``scikit_bio_worker.py`` at the repo root assembles these into the ``skbio``
catalog and runs the worker.
"""

from __future__ import annotations

__version__ = "0.1.1"
