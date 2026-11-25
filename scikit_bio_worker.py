# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "vgi-python[http,oauth]>=0.55.2,<0.56",
#     "vgi-rpc[sentry]",
#     "scikit-bio>=0.6",
#     "numpy",
#     "pandas",
#     "scipy",
# ]
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
