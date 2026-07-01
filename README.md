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
  - [`diversity` — alpha & beta diversity](#diversity--alpha--beta-diversity)
  - [`stats` — ordination, distance tests, composition](#stats--ordination-distance-tests-composition)
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

Functions are organised into four schemas — `sequence` (the default), `diversity`,
`stats`, and `tree` — so `skbio.sequence.gc_content(...)` also resolves as
`skbio.gc_content(...)`.

---

## What's inside

### `sequence` — biological sequences

Per-sequence functions over `VARCHAR` columns of DNA/RNA/protein. Input is
upper-cased and stripped; a NULL or invalid sequence yields NULL rather than
failing the whole query.

| Function | Kind | Returns | Description |
| --- | --- | --- | --- |
| `gc_content(seq)` | scalar | `DOUBLE` | GC fraction of DNA in `[0, 1]` |
| `reverse_complement(seq)` | scalar | `VARCHAR` | opposite-strand sequence |
| `complement(seq)` | scalar | `VARCHAR` | base complement (order preserved) |
| `transcribe(seq)` | scalar | `VARCHAR` | DNA → RNA (T → U) |
| `translate(seq)` | scalar | `VARCHAR` | DNA → protein (standard code) |
| `is_valid_dna(seq)` | scalar | `BOOLEAN` | is the string valid IUPAC DNA? |
| `is_valid_protein(seq)` | scalar | `BOOLEAN` | is the string valid IUPAC protein? |
| `hamming_distance(a, b)` | scalar | `DOUBLE` | mismatch fraction of two equal-length sequences |
| `kmer_frequencies(tbl, k := …)` | table | long | overlapping k-mer counts per sequence |
| `residue_frequencies(tbl)` | table | long | single base/amino-acid counts per sequence |

```sql
-- Profile reads by GC content
SELECT id, skbio.sequence.gc_content(seq) AS gc
FROM reads
WHERE skbio.sequence.is_valid_dna(seq);

-- 4-mer feature matrix (long → pivot for a wide matrix)
SELECT id, kmer, count
FROM skbio.sequence.kmer_frequencies((SELECT id, seq FROM reads), id := 'id', k := 4);
```

### `diversity` — alpha & beta diversity

Community-ecology diversity over a **long feature table** — one row per
`(sample, feature, count)`.

**Alpha diversity** is a set of SQL aggregates over the `count` column, so
`GROUP BY sample` gives one value per sample:

| Aggregate | Description |
| --- | --- |
| `shannon(count)` | Shannon entropy (richness + evenness) |
| `simpson(count)` | Simpson index (`1 − dominance`) |
| `inv_simpson(count)` | inverse Simpson (effective number of features) |
| `observed_features(count)` | richness (non-zero features) |
| `chao1(count)` | Chao1 estimated richness (corrects for unseen) |
| `pielou_evenness(count)` | Pielou's evenness in `[0, 1]` |
| `dominance(count)` | Simpson's dominance |

```sql
SELECT sample_id,
       skbio.diversity.shannon(count)  AS shannon,
       skbio.diversity.chao1(count)    AS chao1
FROM feature_table
GROUP BY sample_id;
```

**Beta diversity** is a table function that emits the full between-sample
distance matrix in long form, ready for `pcoa` / `permanova`:

```sql
SELECT * FROM skbio.diversity.beta_diversity(
  (SELECT sample_id, feature_id, count FROM feature_table),
  metric := 'braycurtis');   -- also: jaccard, euclidean, cityblock, canberra, cosine, ...
-- → (id_1, id_2, distance)
```

### `stats` — ordination, distance tests, composition

| Function | Description |
| --- | --- |
| `pcoa(dm, n_components := k)` | Principal Coordinates Analysis → `(sample_id, pc_1 … pc_k)` |
| `permanova(dm_with_group)` | does a grouping explain between-sample distances? (pseudo-F) |
| `anosim(dm_with_group)` | are within-group distances smaller than between-group? (R) |
| `mantel(two_dms)` | correlation between two distance matrices |
| `clr(feature_table)` | centred log-ratio transform (same long shape) |
| `ilr(feature_table)` | isometric log-ratio transform (`D−1` components) |

`pcoa`, `permanova`, `anosim`, and `mantel` all read the long
`(id_1, id_2, distance)` distance matrix produced by `beta_diversity`.
`permanova`/`anosim` take the sample grouping as a fourth column (the group of
`id_1`); `mantel` takes two distance columns.

```sql
-- Embed samples in 2-D from a Bray–Curtis matrix
SELECT * FROM skbio.stats.pcoa(
  (SELECT * FROM skbio.diversity.beta_diversity((SELECT * FROM feature_table))),
  n_components := 2);
```

### `tree` — phylogenetics

| Function | Kind | Description |
| --- | --- | --- |
| `neighbor_joining(dm)` | table | build an unrooted tree from a distance matrix → one `newick` row |
| `tip_count(newick)` | scalar | number of tips (leaves) in a Newick tree |
| `total_branch_length(newick)` | scalar | sum of all branch lengths (Faith's PD over a feature tree) |

```sql
SELECT newick
FROM skbio.tree.neighbor_joining(
  (SELECT * FROM skbio.diversity.beta_diversity((SELECT * FROM feature_table))));
```

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
