# CLAUDE.md — vgi-scikit-bio

Contributor/agent notes for this repo. User-facing docs live in `README.md`;
this file is the "how it's built and where the sharp edges are" companion.

## What this is

A [VGI](https://github.com/query-farm/vgi-python) worker exposing
[scikit-bio](https://scikit.bio) to DuckDB/SQL. `vgi_scikit_bio/worker.py`
assembles every function into one `skbio` catalog (four schemas) and runs it over
stdio (local) or HTTP (container/Fly.io). Modeled on `~/Development/vgi-scikit-learn`.

## Layout

```
vgi_scikit_bio/
  worker.py          builds the `skbio` Catalog + ScikitBioWorker; main()/main_http() entry points
  sequence.py        DNA/RNA/protein scalar functions (gc_content, reverse_complement, translate, ...)
  kmers.py           kmer_frequencies / residue_frequencies (buffering, long-format composition)
  diversity.py       alpha-diversity aggregates + beta_diversity distance matrix (buffering)
  ordination.py      pcoa — principal coordinates over a long distance matrix (buffering)
  distance_stats.py  permanova / anosim / mantel — single-row distance-matrix tests (buffering)
  composition.py     clr / ilr — compositional log-ratio transforms (buffering)
  tree.py            neighbor_joining (buffering) + tip_count / total_branch_length (scalars)
  distance_utils.py  reconstruct a skbio DistanceMatrix from a long (id_1, id_2, distance) table
  buffering.py       shared SinkBuffer sink/combine/serialize helpers for whole-input functions
  schema_utils.py    pa.Field comment helper, name sanitisation, NoArgs
scikit_bio_worker.py repo-root stdio shim over vgi_scikit_bio.worker (uv run / container / tests)
serve.py             repo-root HTTP shim over vgi_scikit_bio.worker (container / tests)
tests/               pytest (function-level unit tests; aggregates via tests/harness.py)
test/sql/*.test      DuckDB sqllogictest — the authoritative integration tests
```

To add functions: implement in the relevant `vgi_scikit_bio/*.py`, export a
`*_FUNCTIONS` list, and splice it into `_SCHEMA_FUNCTIONS` in
`vgi_scikit_bio/worker.py`.

**Entry points / packaging.** Console scripts (`vgi-scikit-bio`,
`vgi-scikit-bio-http`) point at `vgi_scikit_bio.worker:main` / `:main_http` —
*inside the package*, so they ship in the wheel. The repo-root
`scikit_bio_worker.py` / `serve.py` are thin shims for `uv run` / the container /
`import` in tests; they are deliberately NOT in the wheel. Don't point entry
points at the root modules — that breaks console scripts on `pip install`.

## Which VGI primitive for which job

| Need | Primitive | Example here |
| --- | --- | --- |
| One value per row from one/more columns | `ScalarFunction` (`Param`/`Returns` on `compute`) | `sequence.py`, `tree.TipCount` |
| Scalar-per-group over one column | `AggregateFunction[State]` | `diversity._AlphaMetric` |
| Needs the whole input before emitting | `SinkBuffer` (a `TableBufferingFunction`) | `kmers`, `beta_diversity`, `pcoa`, `permanova`, `clr`, `neighbor_joining` |

- **Scalar functions output a single column named `result`, 1:1 with input
  rows.** Declare inputs with `Annotated[pa.XArray, Param(doc=...)]` and the
  output with `Annotated[pa.XArray, Returns(pa.type())]`.
- **Buffering functions** subclass `SinkBuffer[Args, DrainState]`, set
  `FunctionArguments`, implement `on_bind` (fix the output schema),
  `initial_finalize_state`, and `finalize`. `finalize` is a single-shot drain:
  guard on `state.done`, then emit one batch. The heavy logic lives in an
  `encode(table, args) -> dict[str, list]` classmethod so it is unit-testable
  without the RPC lifecycle (see `tests/`).

## Data-shape conventions (the core design)

- **Feature tables are long** — `(sample_id, feature_id, count/value)`. Columns
  default positionally and are overridable by named args (`sample :=`,
  `feature :=`, `count :=`). `beta_diversity`/`clr`/`ilr` pivot this to a dense
  matrix internally (missing cells = 0).
- **Distance matrices are long** — `(id_1, id_2, distance)`, exactly what
  `beta_diversity` emits. `distance_utils.distance_matrix_from_long` rebuilds a
  `skbio.DistanceMatrix`, **symmetrizing and zero-filling the diagonal**, so a
  full square or a triangle both parse.
- **Grouped tests take the grouping as a fourth column** (`permanova`/`anosim`):
  `(id_1, id_2, distance, group)` where `group` is the label of `id_1`. Because
  a table function gets only one subquery slot, this is how the per-sample
  grouping rides in — build it by joining a grouping table onto a
  `beta_diversity` result on `id_1`. **Every sample must appear as `id_1`**, which
  the full-square `beta_diversity` output guarantees; a condensed triangle would
  leave the last sample ungrouped (a clear error is raised).
- **`mantel` takes two distance columns** — `(id_1, id_2, distance_x, distance_y)`
  — since it correlates two matrices over the same id pairs.
- **Bad rows degrade to NULL** in the sequence scalars: each cell is upper-cased
  and parsed in a `try/except`, so one malformed read never fails the query.

## Sharp edges (read before debugging)

1. **Aggregate state: reassign, don't mutate.** `_AlphaMetric.update` does
   `states[g] = CountState(counts=...)` — the framework only persists groups you
   *assign* this batch. An in-place `.counts.append` on a group first seen in the
   batch is silently dropped (every result NULL). Single-group/whole-table
   aggregates always hit this. Unit tests call classmethods directly and can pass
   while the real RPC path is broken — **the SQL suite is authoritative**.
2. **Never name an aggregate `update()` parameter `params`** — the framework
   injects `ProcessParams` into any arg named `params`. The count column is named
   `count` here; fine.
3. **Scalar output is one column named `result`, same row count as input.**
   Returning a different length or multiple columns fails validation.
4. **`query <letters>` in `.test` files: the letter count must equal the output
   column count** (`query I` = 1 col, `query II` = 2, `query IIII` = 4). A
   mismatch fails with a "Wrong column count" error, not a value diff.
5. **Table argument syntax is `(SELECT ...)`, not `TABLE(...)`.** A table
   function gets at most one subquery parameter — that slot is the input relation.
6. **Output schema is fixed at bind.** Fine when width comes from args (`pcoa`
   `n_components`) or is a fixed long shape. Data-dependent width (k-mer
   vocabulary, distance-matrix size) is emitted **long** to dodge this.
7. **PCoA cannot return more axes than samples.** `ordination` caps the
   `number_of_dimensions` it requests at the sample count and NULL-pads any extra
   requested `pc_k` columns.
8. **scikit-bio returns pandas Series/DataFrames** — index by `.iloc`/`.to_numpy()`,
   not positional `[0]` (pandas 3.x raises `KeyError 0` on integer-label lookup).
9. **Alpha metrics operate on integer abundances.** Counts are rounded to int in
   `finalize` (chao1 in particular needs integer singletons/doubletons).
10. **HTTP entry point:** current vgi-python has no `main_http`; `serve.py` /
    `main_http()` inject `--http` into the worker CLI.

## Packaging & CI

Installable package (`pyproject.toml`, hatchling, `uv.lock`): `uv sync` resolves
PyPI `vgi-python[http]` + `scikit-bio` + numpy/pandas/scipy and exposes the
`vgi-scikit-bio` (stdio) and `vgi-scikit-bio-http` console scripts. Lint/format is
ruff; types mypy; docstrings pydoclint (config in `pyproject.toml`).

- **Version is single-sourced** from `__version__` in `vgi_scikit_bio/__init__.py`
  (`pyproject.toml` is `dynamic = ["version"]`). It is advertised over VGI as
  `implementation_version` (a semver — *not* the git sha) and `data_version`.
  Bump it before tagging; `ci/check-version.sh` fails a release if the tag
  doesn't match.
- **GitHub Actions** (`.github/workflows/ci.yml` + `ci/`) runs the unit + SQL
  suites on Linux/macOS/Windows against the **signed community `vgi` extension**
  via a prebuilt `haybarn-unittest` — no C++ build (see `ci/README.md`). A
  metadata-quality gate runs `vgi-lint-check` (static, fail-on warning; execute,
  fail-on error).
- **Container image → ghcr.io** (`docker-publish.yml`): one multi-arch image
  serving both transports (`http` default, `stdio`/`tcp` args via
  `docker-entrypoint.sh`). Tag-driven; installs from PyPI (`pip install .[serve]`).
  The `/data` volume holds only the shared `BoundStorage` SQLite (there is no
  model registry — scikit-bio has no fit/predict state to persist).
- **PyPI publish** is `publish.yml` (GitHub Release → CI → `uv build && uv publish`).

## Testing

```sh
uv sync && uv run pytest tests/ -q   # unit tests against PyPI deps (CI's unit job)
make venv && make pytest             # unit tests against local vgi checkouts
make test-stdio                      # SQL tests, worker as subprocess (authoritative)
make test-http                       # SQL tests against a local HTTP server
```

- **SQL tests are authoritative** — they drive the real DuckDB → VGI → worker
  path. Locally they need a sqllogictest runner with the vgi extension: either a
  DuckDB `unittest` built with vgi at `$(VGI_BUILD_DIR)/test/unittest`, or a
  standalone `haybarn-unittest` + `INSTALL vgi FROM community` (what CI uses).
- Unit tests exercise each function's `compute`/`encode` logic in-process; the
  full buffering/aggregate lifecycle is covered by the SQL suite.

## Deployment (Fly.io)

```sh
make deploy        # fly deploy --image ghcr.io/query-farm/vgi-scikit-bio:<version>
fly volumes create scikit_bio_state --size 1 --region iad   # one-time, shared state
```

Fly pulls the CI-published ghcr image (no local build). `fly.toml` sets VM memory
to 1gb (scikit-bio/scipy are heavy) and mounts a volume at `/data` for the shared
`BoundStorage` SQLite (`VGI_WORKER_SQLITE_PATH=/data/state/...`).
