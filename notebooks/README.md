# scikit-bio notebooks

[01 · Same proportions, different total counts](01_same_proportions_different_counts.ipynb)
compares within-sample diversity and Bray–Curtis distance before and after closing
each sample to proportions. It uses six explicitly synthetic communities and SQL
reconstructions of the real worker's results. The experiment multiplies one
community's counts by eight instead of three, keeping its proportions fixed.

## Run from this directory alone

Install [uv](https://docs.astral.sh/uv/) and Git, then:

```sh
cd notebooks
uv sync --locked
uv run --locked python execute.py
uv run --locked python -m ipykernel install --user --name vgi-scikit-bio-notebooks --display-name 'Python (vgi-scikit-bio notebooks)'
```

Select that kernel in Jupyter and start it in this directory. The notebook supports
Python 3.13/3.14; the worker command requests Python 3.13 through uv. Initial setup
downloads the locked notebook packages, VGI extension and tagged worker. No worker
checkout, root environment or locally compiled binary is required. Tested on macOS
ARM64 with Haybarn 2.0.0rc11. No credentials, API calls or live/replay mode are used.

The notebook shows this literal launch:

```sql
INSTALL vgi FROM community;
LOAD vgi;
ATTACH 'skbio' (
 TYPE vgi,
 LOCATION 'uvx --python 3.13 --from git+https://github.com/Query-farm/vgi-scikit-bio.git@v0.3.0 --constraints worker-constraints.txt vgi-scikit-bio'
);
```

The published Git release v0.3.0 resolves to
`eea42cf85f63a79b06be0e08880fe3e9c3824daa`. It is not a PyPI package or a native
GitHub binary. `uvx` builds and caches the tagged package. The companion constraints
retain the source's tested numerical versions (scikit-bio 0.7.3, NumPy 2.5.0,
SciPy 1.18.0, pandas 3.0.3, Arrow 24.0.0) and SDK 0.51.0/RPC 0.53.1. Root dependencies
and locks are unchanged. To upgrade, change the visible tag and constraints, then
rerun defaults, the experiment, and the authoritative source SQL suites.

## Input and interpretation

The SQL `VALUES` block contains the complete synthetic input: six samples, five
arbitrary features, nonnegative integer counts including explicit zeros. Only
B_scaled receives the visible multiplier. There are no external observations,
biological sample identifiers or estimated absolute abundances. The notebook records
a fingerprint of the unscaled table in its outputs and ignored audit JSON.

Missing/NULL values and duplicate sample-feature keys are rejected before worker
calls. The worker otherwise treats missing/NULL cells as zero and sums duplicate
keys when pivoting; that behavior is unsuitable for missing measurements without an
explicit domain decision. Every synthetic sample has a positive total.

Alpha aggregates receive integer counts because the worker rounds those inputs.
Shannon entropy uses natural logarithms (nats), and the order-1 Hill number equals
its exponential. These summaries do not retain feature identity. The beta-diversity
worker accepts fractional proportions: both raw-count and closed matrices are
reconstructed independently in SQL and checked for symmetry, bounds, zero diagonal
and full pair coverage.

Bray–Curtis on counts answers a different question from Bray–Curtis on proportions.
Closing a sample divides by its total, with no pseudocount; it removes total-size
information. It does not rarefy, recover absolute abundance or correct sampling
uncertainty. Exact rescaling of a fixed vector is not a model of real deeper sampling.
The notebook makes no inferential or biological claim.

References: [Bray–Curtis formula](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.braycurtis.html)
and [Shannon and effective diversity](https://scikit.bio/docs/latest/generated/skbio.diversity.alpha.shannon.html).

## Separate one-parameter experiment

```sh
uv run --locked python execute.py --parameters '{"DEPTH_MULTIPLIER":8}' --output-dir validation/multiplier-8
```

A_base/B_scaled raw Bray–Curtis should change from 1/2 to 7/9, while their closed
distance stays zero. All alpha summaries, all closed distances, and raw pairs that
exclude B_scaled should stay unchanged. Restore the default multiplier 3 afterwards.
Feature colors and distance scales are fixed across runs; the count axis adapts.

The runner uses a fresh kernel and atomically replaces the saved notebook only after
success. Overrides require a separate output destination; unknown names are rejected.
Experiment copies include assets, constraints, environment lock and documentation.
Generated `artifacts/` and `validation/` are ignored; successful default charts and
outputs are embedded in the committed notebook. See [STANDARD.md](STANDARD.md).

## Validation

Default and notebooks-only multiplier-8 executions pass, with two inspected charts
each, six SQL matrix checks, alpha/closure checks and cross-run invariants. The
runner's three invalid-override guards preserve the saved default.

Source validation passes 100 unit tests; the docstring gate skips once because
`docstring-parser` and `docstring-parser-fork` overlap (`DocstringYields` import
error). Isolated `uvx --from pydoclint==0.8.7 pydoclint --config pyproject.toml
vgi_scikit_bio/ scikit_bio_worker.py serve.py` passes. Ruff, formatting and strict
mypy pass. The authoritative suites pass 143 stdio and 143 Unix-launch assertions
in six cases, plus 155 HTTP assertions including explicit httpfs install/load
prerequisites in staged tests. The HTTP test process is scoped and stopped after use.

Source metadata checks (`vgi-lint-check==2.5.0`, link checking disabled) score
100/100: static fail-on-warning and executable fail-on-error gates both pass with
zero findings; executable assurance is L2. No worker source or root lock changes.
