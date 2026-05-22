"""Aris: production-grade Mixture-of-Experts decoder-only LLM framework."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aris")
except PackageNotFoundError:  # editable / source install
    __version__ = "0.1.0"

__all__ = ["__version__"]
