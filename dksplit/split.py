"""
DKSplit - High-performance string segmentation using BiLSTM-CRF
"""
import os
os.environ['ORT_LOG_LEVEL'] = '3'
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import List, Optional, Union
from collections import defaultdict


# ============ Character Mapping (Array for fast lookup) ============
CHAR_VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789"
PAD_IDX = 0
UNK_IDX = 1
MAX_LEN = 64

# Build lookup table
CHAR_MAP = np.zeros(128, dtype=np.int64)
CHAR_MAP[:] = UNK_IDX
for i, c in enumerate(CHAR_VOCAB):
    CHAR_MAP[ord(c)] = i + 2


def _get_model_dir() -> Path:
    """Get model directory"""
    return Path(__file__).parent / "models"


def _text_to_ids_fast(text: str) -> np.ndarray:
    """Convert text to character IDs using array lookup"""
    arr = np.frombuffer(text.encode('ascii', errors='replace'), dtype=np.uint8)
    return CHAR_MAP[np.clip(arr, 0, 127)]


def _crf_decode(
    emissions: np.ndarray,
    transitions: np.ndarray,
    start_transitions: np.ndarray,
    end_transitions: np.ndarray
) -> np.ndarray:
    """
    CRF Viterbi decoding (no padding, all sequences same length)
    """
    batch_size, seq_len, num_tags = emissions.shape
    
    # Initialize
    score = start_transitions + emissions[:, 0]
    history = []
    
    # Forward pass
    for t in range(1, seq_len):
        broadcast_score = score[:, :, None]
        broadcast_emissions = emissions[:, t, None, :]
        next_score = broadcast_score + transitions + broadcast_emissions
        
        history.append(np.argmax(next_score, axis=1))
        score = np.max(next_score, axis=1)
    
    # Add end transitions
    score = score + end_transitions
    best_last_tags = np.argmax(score, axis=1)
    
    # Backtrack
    batch_idx = np.arange(batch_size)
    best_paths = np.zeros((batch_size, seq_len), dtype=np.int32)
    best_paths[:, seq_len - 1] = best_last_tags
    
    for t in range(seq_len - 2, -1, -1):
        best_paths[:, t] = history[t][batch_idx, best_paths[:, t + 1]]
    
    return best_paths


def _decode_predictions_batch(texts: List[str], preds: np.ndarray) -> List[List[str]]:
    """Batch decode predictions to words"""
    results = []
    for i, text in enumerate(texts):
        pred = preds[i]
        words = []
        current = []
        
        for char, label in zip(text, pred):
            if label == 1 and current:
                words.append(''.join(current))
                current = [char]
            else:
                current.append(char)
        
        if current:
            words.append(''.join(current))
        
        results.append(words)
    
    return results


class Splitter:
    """
    High-performance string splitter
    
    Example:
        >>> splitter = Splitter()
        >>> splitter.split("chatgptlogin")
        ['chatgpt', 'login']
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        crf_path: Optional[str] = None,
        num_threads: int = 4,
        use_gpu: bool = False
    ):
        model_dir = _get_model_dir()
        
        if model_path is None:
            model_path = model_dir / "dksplit-int8.onnx"
        if crf_path is None:
            crf_path = model_dir / "dksplit.npz"
        
        # ONNX Runtime configuration
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers
        )
        
        # Load CRF parameters
        crf_data = np.load(crf_path)
        self.transitions = crf_data['transitions'].astype(np.float32)
        self.start_transitions = crf_data['start_transitions'].astype(np.float32)
        self.end_transitions = crf_data['end_transitions'].astype(np.float32)
    
    def split(self, text: str) -> List[str]:
        """Split single text"""
        if not text:
            return []
        
        text = text.lower()[:MAX_LEN]
        char_ids = _text_to_ids_fast(text)
        
        chars = char_ids.reshape(1, -1)
        emissions = self.session.run(None, {'chars': chars})[0]
        
        preds = _crf_decode(
            emissions,
            self.transitions,
            self.start_transitions,
            self.end_transitions
        )
        
        return _decode_predictions_batch([text], preds)[0]
    
    def split_batch(self, texts: List[str], batch_size: int = 256) -> List[List[str]]:
        """Batch split (groups by length for accuracy)"""
        if not texts:
            return []
        
        n = len(texts)
        results = [None] * n
        
        # Preprocess and group by length
        length_groups = defaultdict(list)
        for i, text in enumerate(texts):
            processed = text.lower()[:MAX_LEN]
            length = len(processed)
            if length == 0:
                results[i] = []
            else:
                length_groups[length].append((i, processed))
        
        # Process each length group
        for length, group in length_groups.items():
            for batch_start in range(0, len(group), batch_size):
                batch = group[batch_start:batch_start + batch_size]
                indices = [x[0] for x in batch]
                batch_texts = [x[1] for x in batch]
                batch_len = len(batch_texts)
                
                # Build input using fast conversion
                chars = np.zeros((batch_len, length), dtype=np.int64)
                for i, text in enumerate(batch_texts):
                    chars[i] = _text_to_ids_fast(text)
                
                # Inference
                emissions = self.session.run(None, {'chars': chars})[0]
                
                # CRF decode
                preds = _crf_decode(
                    emissions,
                    self.transitions,
                    self.start_transitions,
                    self.end_transitions
                )
                
                # Decode and map back
                decoded = _decode_predictions_batch(batch_texts, preds)
                for i, idx in enumerate(indices):
                    results[idx] = decoded[i]
        
        return results
    
    def __call__(self, text: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        if isinstance(text, str):
            return self.split(text)
        return self.split_batch(text)


# ============ Global Instance ============
_default_splitter: Optional[Splitter] = None


def _get_splitter() -> Splitter:
    global _default_splitter
    if _default_splitter is None:
        _default_splitter = Splitter()
    return _default_splitter


def split(text: str) -> List[str]:
    """
    Split text into words
    
    Example:
        >>> import dksplit
        >>> dksplit.split("chatgptlogin")
        ['chatgpt', 'login']
    """
    return _get_splitter().split(text)


def split_batch(texts: List[str], batch_size: int = 256) -> List[List[str]]:
    """
    Batch split texts into words
    
    Example:
        >>> import dksplit
        >>> dksplit.split_batch(["openaikey", "microsoftoffice"])
        [['openai', 'key'], ['microsoft', 'office']]
    """
    return _get_splitter().split_batch(texts, batch_size)