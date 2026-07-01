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
