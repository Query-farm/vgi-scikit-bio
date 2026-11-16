# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "vgi-python[http,oauth]>=0.52.2,<0.53",
#     "vgi-rpc[sentry]",
#     "scikit-bio>=0.6",
#     "numpy",
#     "pandas",
#     "scipy",
# ]
# ///
"""HTTP entry shim for the scikit-bio VGI worker (used by the container).

Forces the worker CLI into HTTP mode. The implementation lives in
``vgi_scikit_bio.worker``; installed users invoke the ``vgi-scikit-bio-http``
console script (which points at ``vgi_scikit_bio.worker:main_http``) instead.
"""

from vgi_scikit_bio.worker import ScikitBioWorker, main_http

__all__ = ["ScikitBioWorker", "main_http"]


def main() -> None:
    """Run the worker over HTTP."""
    main_http()


if __name__ == "__main__":
    main()
