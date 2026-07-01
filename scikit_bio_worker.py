# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http,oauth]",
#     "vgi-rpc[sentry]",
#     "scikit-bio>=0.6",
#     "numpy",
#     "pandas",
#     "scipy",
# ]
#
# [tool.uv.sources]
# vgi-python = { path = "../vgi-python" }
# vgi-rpc = { path = "../vgi-rpc" }
#
# [tool.uv]
# # Use the local vgi-rpc checkout even if it lags vgi-python's pinned lower bound.
# override-dependencies = ["vgi-rpc>=0.21.0"]
# ///
"""Stdio entry shim for the scikit-bio VGI worker.

Lets the worker run straight from a source checkout (``uv run
scikit_bio_worker.py``) and from the container, and keeps ``import
scikit_bio_worker`` working for tests. The implementation lives in
``vgi_scikit_bio.worker``; installed users invoke the ``vgi-scikit-bio`` console
script (which points at ``vgi_scikit_bio.worker:main``) instead.

    ATTACH 'skbio' (TYPE vgi, LOCATION 'uv run scikit_bio_worker.py');
    SELECT skbio.sequence.gc_content('ATGCGGATTACAGG');
"""

from vgi_scikit_bio.worker import ScikitBioWorker, main

__all__ = ["ScikitBioWorker", "main"]

if __name__ == "__main__":
    main()
