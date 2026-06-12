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


def _crf_decode_topk(
    emissions: np.ndarray,
    transitions: np.ndarray,
    start_transitions: np.ndarray,
    end_transitions: np.ndarray,
    k: int
) -> List[np.ndarray]:
    """
    CRF k-best Viterbi decoding for a single sequence.

    Keeps the top-k scoring paths per tag at each step, then returns all
    finite-score paths sorted by total score (best first). The number of
    returned paths is at most k * num_tags.

    Args:
        emissions: (seq_len, num_tags) emission scores
        k: beam width (paths kept per tag)

    Returns:
        List of (seq_len,) int32 tag paths, best first.
    """
    seq_len, num_tags = emissions.shape

    # score[j, r]: score of the r-th best path ending at tag j
    score = np.full((num_tags, k), -np.inf, dtype=np.float32)
    score[:, 0] = start_transitions + emissions[0]
    history = []  # history[t-1][j, r] = flat index (prev_tag * k + prev_rank)

    for t in range(1, seq_len):
        # cand[i*k+r, j] = score[i, r] + transitions[i, j] + emissions[t, j]
        cand = score[:, :, None] + transitions[:, None, :] + emissions[t][None, None, :]
        cand = cand.reshape(num_tags * k, num_tags)

        idx = np.argsort(-cand, axis=0)[:k]  # (k, num_tags)
        history.append(idx.T)                # (num_tags, k)
        score = np.take_along_axis(cand, idx, axis=0).T

    final = score + end_transitions[:, None]
    flat = final.reshape(-1)  # index = tag * k + rank
    order = np.argsort(-flat)

    paths = []
    for fi in order:
        if flat[fi] == -np.inf:
            break
        j, r = divmod(int(fi), k)
        path = np.empty(seq_len, dtype=np.int32)
        path[seq_len - 1] = j
        for t in range(seq_len - 1, 0, -1):
            j, r = divmod(int(history[t - 1][j, r]), k)
            path[t - 1] = j
        paths.append(path)

    return paths


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

    def split_topk(self, text: str, k: int = 3) -> List[List[str]]:
        """
        Split single text, returning the top-k segmentations (best first).

        Each tag path maps to a segmentation; distinct paths can collide on
        the same segmentation, so 2k paths are decoded and deduplicated.
        Short inputs may have fewer than k possible segmentations, in which
        case fewer candidates are returned.

        Example:
            >>> splitter.split_topk("pikahug", k=3)
            [['pikahug'], ['pika', 'hug'], ['pik', 'ahug']]
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        if not text:
            return []

        text = text.lower()[:MAX_LEN]
        char_ids = _text_to_ids_fast(text)

        chars = char_ids.reshape(1, -1)
        emissions = self.session.run(None, {'chars': chars})[0][0]

        # Each segmentation corresponds to exactly 2 tag paths (the first
        # character's tag does not affect word boundaries), so a beam of
        # 2k paths is always enough to yield k unique segmentations.
        paths = _crf_decode_topk(
            emissions,
            self.transitions,
            self.start_transitions,
            self.end_transitions,
            2 * k
        )

        results = []
        seen = set()
        for path in paths:
            words = _decode_predictions_batch([text], [path])[0]
            key = tuple(words)
            if key not in seen:
                seen.add(key)
                results.append(words)
                if len(results) == k:
                    break

        return results

    def split_batch(self, texts: List[str], batch_size: int = 256, *,
                    exact: bool = True) -> List[List[str]]:
        """Batch split (groups by length; no padding).

        exact=True (default): inference runs row by row, so each result is
        guaranteed identical to split() on that text. exact=False: whole-batch
        inference, roughly 2x faster, but results can differ slightly from
        split() depending on batch composition.
        """
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
                if exact:
                    # Row by row keeps results identical to split(): the INT8
                    # model is dynamically quantized, so in a whole-batch run
                    # rows perturb each other's emissions.
                    emissions = np.concatenate(
                        [self.session.run(None, {'chars': chars[i:i + 1]})[0]
                         for i in range(batch_len)],
                        axis=0
                    )
                else:
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


def split_batch(texts: List[str], batch_size: int = 256, *,
                exact: bool = True) -> List[List[str]]:
    """
    Batch split texts into words

    Results are identical to calling split() on each text. Pass exact=False
    for a faster mode whose results can differ slightly from split().

    Example:
        >>> import dksplit
        >>> dksplit.split_batch(["openaikey", "microsoftoffice"])
        [['openai', 'key'], ['microsoft', 'office']]
    """
    return _get_splitter().split_batch(texts, batch_size, exact=exact)


def split_topk(text: str, k: int = 3) -> List[List[str]]:
    """
    Split text into words, returning the top-k segmentations (best first)

    Example:
        >>> import dksplit
        >>> dksplit.split_topk("pikahug", k=3)
        [['pikahug'], ['pika', 'hug'], ['pik', 'ahug']]
    """
    return _get_splitter().split_topk(text, k)


def split3(text: str) -> List[List[str]]:
    """
    Split text into words, returning the top-3 segmentations (best first)

    Example:
        >>> import dksplit
        >>> dksplit.split3("chatgptlogin")
        [['chatgpt', 'login'], ['chatgptlogin'], ['chatgpt', 'log', 'in']]
    """
    return _get_splitter().split_topk(text, 3)


def split5(text: str) -> List[List[str]]:
    """
    Split text into words, returning the top-5 segmentations (best first)

    Example:
        >>> import dksplit
        >>> dksplit.split5("pikahug")
        [['pikahug'], ['pika', 'hug'], ['pik', 'ahug'], ['pikah', 'ug'], ['pi', 'kahug']]
    """
    return _get_splitter().split_topk(text, 5)