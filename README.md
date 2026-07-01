# vgi-scikit-bio

**[scikit-bio](https://scikit.bio) for SQL** — a [VGI](https://query.farm)
worker that exposes biological sequence analysis, community diversity,
ordination, distance-matrix statistics, compositional transforms, and
phylogenetics to **DuckDB** as ordinary scalar, aggregate, and table functions.

```sql
-- Attach the worker (installed console script, or `uv run scikit_bio_worker.py`)
ATTACH 'skbio' (TYPE vgi, LOCATION 'vgi-scikit-bio');

SELECT skbio.sequence.gc_content('ATGCGGATTACAGG');            -- 0.5
SELECT skbio.sequence.reverse_complement('ATGCGGATTACAGG');    -- CCTGTAATCCGCAT
SELECT skbio.sequence.translate('ATGCGGATTACAGGT');            -- MRITG
```

Everything runs in a separate Python process and streams Apache Arrow between
DuckDB and scikit-bio, so you get scikit-bio's algorithms without leaving SQL.

---

## Contents

- [Install & attach](#install--attach)
- [What's inside](#whats-inside)
  - [`sequence` — biological sequences](#sequence--biological-sequences)
  - [`alignment` — pairwise alignment](#alignment--pairwise-alignment)
  - [`diversity` — alpha, beta, phylogenetic, rarefaction](#diversity--alpha-beta-phylogenetic-rarefaction)
  - [`stats` — ordination, distance tests, composition, differential abundance](#stats--ordination-distance-tests-composition-differential-abundance)
  - [`tree` — phylogenetics](#tree--phylogenetics)
- [A worked microbiome example](#a-worked-microbiome-example)
- [Data-shape conventions](#data-shape-conventions)
- [Development](#development)
- [Deployment](#deployment)
- [License & support](#license--support)

---

## Install & attach

The worker is a normal Python package that ships two console scripts:
`vgi-scikit-bio` (stdio, spawned by DuckDB) and `vgi-scikit-bio-http` (HTTP
server).

```sh
pip install vgi-scikit-bio          # or: uv pip install vgi-scikit-bio
```

Attach it from DuckDB (the DuckDB `vgi` community extension must be installed):

```sql
INSTALL vgi FROM community;
LOAD vgi;

-- stdio: DuckDB spawns the worker as a subprocess
ATTACH 'skbio' (TYPE vgi, LOCATION 'vgi-scikit-bio');

-- or run straight from a source checkout without installing:
ATTACH 'skbio' (TYPE vgi, LOCATION 'uv run scikit_bio_worker.py');
```

Functions are organised into **five schemas** — `sequence` (the default),
`alignment`, `diversity`, `stats`, and `tree` — so `skbio.sequence.gc_content(...)`
also resolves as `skbio.gc_content(...)`. There are **~90 functions**; the tables
below list them by area (grouped by the schema's category sections, which the
worker also exposes for navigation).

---

## What's inside

### `sequence` — biological sequences

Per-sequence functions over `VARCHAR` columns of DNA/RNA/protein. Input is
upper-cased and stripped; a NULL or invalid sequence yields NULL rather than
failing the whole query.

- **Transforms** — `gc_content`, `gc_frequency`, `reverse_complement`,
  `complement`, `transcribe`, `reverse_transcribe`, `translate`, `degap`
- **Validation** — `is_valid_dna`, `is_valid_protein`, `has_gaps`,
  `has_degenerates`, `is_reverse_complement`
- **Distance** — `hamming_distance`, `mismatch_count`, `match_count`
- **Composition** (table functions, long output) — `kmer_frequencies`,
  `residue_frequencies`, `count_subsequence`, `translate_six_frames`

```sql
-- Profile reads by GC content, keeping only valid DNA
SELECT id, skbio.sequence.gc_content(seq) AS gc
FROM reads WHERE skbio.sequence.is_valid_dna(seq);

-- 4-mer feature matrix (long → pivot for a wide matrix)
SELECT id, kmer, count
FROM skbio.sequence.kmer_frequencies((SELECT id, seq FROM reads), id := 'id', k := 4);
```

### `alignment` — pairwise alignment

- **Score** — `align_score_nucleotide(a, b)`, `align_score_protein(a, b)` →
  optimal global-alignment score
- **Pairwise** — `pairwise_align_nucleotide` / `pairwise_align_protein` → aligned
  strings + score + length; `mode := 'global'` (Needleman–Wunsch) or `'local'`
  (Smith–Waterman)

```sql
SELECT aligned_1, aligned_2, score
FROM skbio.alignment.pairwise_align_nucleotide((SELECT id, ref, read FROM pairs),
     seq1 := 'ref', seq2 := 'read');
```

### `diversity` — alpha, beta, phylogenetic, rarefaction

Community-ecology diversity over a **long feature table** — one row per
`(sample, feature, count)`.

- **Alpha** (aggregates over `count`, `GROUP BY sample`) — the full scikit-bio
  metric family: `shannon`, `simpson`, `inv_simpson`, `observed_features`,
  `chao1`, `pielou_evenness`, `dominance`, `ace`, `berger_parker_d`,
  `brillouin_d`, `fisher_alpha`, `gini_index`, `goods_coverage`, `heip_evenness`,
  `kempton_taylor_q`, `margalef`, `mcintosh_d`, `mcintosh_e`, `menhinick`,
  `robbins`, `simpson_d`, `simpson_e`, `strong`, `singles`, `doubles`, `enspie`;
  parameterized `hill(count, q := …)` / `renyi` / `tsallis`; and interval/triple
  metrics `chao1_ci`, `esty_ci`, `osd` (return `DOUBLE[]`)
- **Beta** — `beta_diversity(tbl, metric := …)` → the full distance matrix long
  (`braycurtis`, `jaccard`, `euclidean`, `canberra`, `cosine`, `jensenshannon`, …)
- **Phylogenetic** (given a `tree := '<newick>'`) — `faith_pd` (per-sample PD),
  `unifrac` (weighted/unweighted distance matrix)
- **Preprocessing** — `subsample_counts(tbl, depth := N)` (rarefaction)

```sql
SELECT sample_id,
       skbio.diversity.shannon(count) AS shannon,
       skbio.diversity.chao1(count)   AS chao1
FROM feature_table GROUP BY sample_id;

SELECT * FROM skbio.diversity.beta_diversity(
  (SELECT sample_id, feature_id, count FROM feature_table), metric := 'braycurtis');
```

### `stats` — ordination, distance tests, composition, differential abundance

- **Ordination** — `pcoa(dm)` (distance matrix), `pca(tbl)` /
  `ca(tbl)` (feature table) → `(sample_id, <axis>_1 … k)`
- **Hypothesis tests** — `permanova` / `anosim` (grouping in a 4th column),
  `mantel` (two distance columns)
- **Composition** — transforms `clr`, `ilr`, `alr`, `closure`, `centralize`,
  `rclr`, `multi_replace`, `power`; inverses `clr_inv`, `ilr_inv`, `alr_inv`;
  association `pairwise_vlr`
- **Differential abundance** — `ancom`, `dirmult_ttest` (grouping in a 4th column)

`pcoa`/`permanova`/`anosim`/`mantel` read the long `(id_1, id_2, distance)` matrix
from `beta_diversity`. `pca`/`ca` and the composition/ANCOM functions read a long
feature table.

```sql
-- Embed samples in 2-D from a Bray–Curtis matrix
SELECT * FROM skbio.stats.pcoa(
  (SELECT * FROM skbio.diversity.beta_diversity((SELECT * FROM feature_table))),
  n_components := 2);
```

### `tree` — phylogenetics

- **Construction** (distance matrix → one `newick` row) — `neighbor_joining`,
  `upgma`, `gme`, `bme`
- **Inspection** (scalars over a Newick string) — `tip_count`,
  `total_branch_length`, `tree_height`
- **Comparison** (two Newick trees → distance) — `robinson_foulds`,
  `weighted_robinson_foulds`, `cophenetic_distance`

```sql
SELECT newick
FROM skbio.tree.neighbor_joining(
  (SELECT * FROM skbio.diversity.beta_diversity((SELECT * FROM feature_table))));
```

> **Not exposed** (they don't map to a single SQL function): constrained
> ordination `cca`/`rda` and `bioenv`/`pwmantel` (need a second matrix beyond the
> one subquery slot); `permdisp` (an upstream bug in scikit-bio 0.7.3); and
> stateful/IO surface (`TabularMSA`, BIOM `Table`, FASTA/FASTQ readers — DuckDB
> already reads files) and randomness-seeded estimators (`lladser_*`,
> `michaelis_menten_fit`).

---

## A worked microbiome example

Starting from a long OTU/feature table `feature_table(sample_id, feature_id, count)`
and a `metadata(sample_id, group)` table:

```sql
-- 1. Alpha diversity per sample
SELECT sample_id, skbio.diversity.shannon(count) AS shannon
FROM feature_table GROUP BY sample_id;

-- 2. Bray–Curtis distances → PCoA coordinates for an ordination plot
CREATE TABLE bc AS
  SELECT * FROM skbio.diversity.beta_diversity((SELECT * FROM feature_table));

SELECT * FROM skbio.stats.pcoa((SELECT * FROM bc), n_components := 2);

-- 3. Does the grouping explain the community differences? (PERMANOVA)
SELECT * FROM skbio.stats.permanova((
  SELECT bc.id_1, bc.id_2, bc.distance, m.group
  FROM bc JOIN metadata m ON bc.id_1 = m.sample_id));

-- 4. Neighbour-joining tree of the samples
SELECT newick FROM skbio.tree.neighbor_joining((SELECT * FROM bc));
```

---

## Data-shape conventions

- **Feature tables are long**: `(sample_id, feature_id, count)` (or `value`).
  This is the natural SQL shape and avoids a fixed, data-dependent column width.
  `PIVOT` if you need a wide matrix.
- **Distance matrices are long**: `(id_1, id_2, distance)` — exactly what
  `beta_diversity` emits, and what `pcoa`/`permanova`/`anosim`/`mantel`/
  `neighbor_joining` consume. The matrix is symmetrised on read, so a full
  square or a triangle both work (the grouped tests need every sample to appear
  as `id_1`, which the full square from `beta_diversity` guarantees).
- **Column names default positionally** and can be overridden with named args
  (`sample :=`, `feature :=`, `count :=`, `id_1 :=`, …).
- **Bad rows degrade to NULL** in the sequence scalars rather than raising.

---

## Development

The repo is an installable package (`pyproject.toml`, hatchling, `uv.lock`).

```sh
uv sync                    # resolve deps (scikit-bio, numpy, pandas, scipy, vgi-python[http])
uv run pytest tests/ -q    # unit tests
uv run ruff check .        # lint
uv run ruff format --check .
uv run mypy vgi_scikit_bio/

make test-stdio            # DuckDB sqllogictest suite, worker as a subprocess (authoritative)
make test-http             # same suite against a local HTTP server
```

The unit tests exercise each function's logic in-process; the **SQL suite
(`test/sql/*.test`) is authoritative** because it drives the real DuckDB → VGI →
worker path. It needs a sqllogictest runner with the `vgi` extension — CI uses a
prebuilt standalone `haybarn-unittest` plus `INSTALL vgi FROM community` (no C++
build). See `ci/README.md`.

To add a function: implement it in the relevant `vgi_scikit_bio/*.py`, export it
from that module's `*_FUNCTIONS` list, and splice the list into
`_SCHEMA_FUNCTIONS` in `vgi_scikit_bio/worker.py`. See `CLAUDE.md` for the
framework patterns and sharp edges.

## Deployment

One multi-arch container image (`ghcr.io/query-farm/vgi-scikit-bio`) serves both
transports — HTTP (default) and stdio — via `docker-entrypoint.sh`. It is built,
tested against the signed `vgi` extension, and signed by CI on every tag. The
image mounts a `/data` volume for the shared framework `BoundStorage`. `make
deploy` points Fly.io at the published image.

## License & support

MIT — see [LICENSE](LICENSE). Copyright 2026 Query Farm LLC.

Developed and maintained by **[Query.Farm](https://query.farm)**. For bug
reports and feature requests use the
[issue tracker](https://github.com/query-farm/vgi-scikit-bio/issues); for
commercial support and SLAs see [SUPPORT.md](SUPPORT.md) or email
[hello@query.farm](mailto:hello@query.farm).

Built on [scikit-bio](https://scikit.bio) — if you use it in research, please
also cite scikit-bio.
