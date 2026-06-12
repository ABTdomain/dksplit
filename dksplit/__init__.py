"""
DKSplit - High-performance string segmentation using BiLSTM-CRF
"""

__version__ = "1.0.2"

from .split import Splitter, split, split_batch, split_topk, split3, split5

__all__ = ["Splitter", "split", "split_batch", "split_topk", "split3", "split5", "__version__"]