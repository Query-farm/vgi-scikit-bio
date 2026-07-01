# CI test harness

The `test/sql/*.test` sqllogictest suite is the **authoritative** check: it
drives the real DuckDB → `vgi` extension → worker path, which unit tests cannot.
CI runs it without building the VGI C++ extension from source, using two pieces:

- **`haybarn-unittest`** — a prebuilt, standalone DuckDB sqllogictest runner
  published by [`Query-farm-haybarn/haybarn`](https://github.com/Query-farm-haybarn/haybarn).
  CI downloads the latest release asset for the runner OS.
- **the signed community `vgi` extension** — installed at runtime with
  `INSTALL vgi FROM community`, so no local build.

`haybarn-unittest` links none of the extensions the `.test` files `require`, so
`ci/preprocess-require.awk` rewrites each `require <ext>` line into an explicit
signed `INSTALL … / LOAD …` (vgi from community, core extensions from core)
before the suite runs. `ci/run-integration.sh` stages the preprocessed tree,
warms the extension cache once, and runs the suite against the worker LOCATION in
`$VGI_SCIKIT_BIO_WORKER`.

## Transports

`$VGI_SCIKIT_BIO_WORKER` selects how the `.test` files reach the worker:

- **`launch:<command>`** (Linux/macOS in CI) — the warm AF_UNIX launcher: the
  worker is spawned once and reused for every `ATTACH`, so scikit-bio/numpy
  import cost is paid once, not per attach.
- **`http://host:port`** (Windows in CI, where AF_UNIX is unreliable) — one HTTP
  server started for the whole suite.
- a plain **stdio command** (e.g. `vgi-scikit-bio`) — DuckDB cold-spawns a worker
  per `ATTACH`; simplest, but slowest.

## Running locally

```sh
# against a source checkout, worker as a subprocess (see the repo Makefile)
make test-stdio
make test-http

# or directly, mirroring CI:
HAYBARN_UNITTEST=/path/to/haybarn-unittest \
VGI_SCIKIT_BIO_WORKER="launch:$(pwd)/.venv/bin/vgi-scikit-bio" \
  ci/run-integration.sh
```
