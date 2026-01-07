"""
DKSplit - High-performance string segmentation using BiLSTM-CRF
"""

__version__ = "0.1.0"

from .split import Splitter, split, split_batch

__all__ = ["Splitter", "split", "split_batch", "__version__"]