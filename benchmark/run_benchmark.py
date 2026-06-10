"""
DKSplit Benchmark - 1,000 real newly registered domains
Reference: Multi-model cross-validation + human audit
Data source: ABTdomain.com daily feed, random sample

CSV columns:
    prefix       - the concatenated input string
    truth        - the canonical (primary) segmentation
    might_right  - an acceptable alternative segmentation (optional)

A prediction counts as:
    Strict EM   - exact match against `truth`
    Lenient EM  - exact match against `truth` OR `might_right`
"""

import csv
import sys
import time
from pathlib import Path


def main():
    import dksplit
    import wordsegment
    import wordninja

    wordsegment.load()

    # Load data (optionally pass a CSV path, e.g. benchmark_5000.csv)
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        csv_path = Path(__file__).parent / "sample_1000.csv"
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prefix = row['prefix'].strip()
            truth = row['truth'].strip().lower()
            might = row.get('might_right', '').strip().lower()
            data.append((prefix, truth, might))

    print(f"Benchmark: {len(data)} real newly registered domains")
    print(f"Reference: Multi-model cross-validation + human audit")
    print(f"Source: ABTdomain.com daily feed, random sample\n")

    models = {
        'DKSplit': lambda t: ' '.join(dksplit.split(t)),
        'WordSegment': lambda t: ' '.join(wordsegment.segment(t)),
        'WordNinja': lambda t: ' '.join(wordninja.split(t)),
    }

    results = {name: {'strict': 0, 'lenient': 0, 'time': 0} for name in models}

    for prefix, truth, might in data:
        for name, fn in models.items():
            start = time.perf_counter()
            result = fn(prefix).lower().strip()
            elapsed = time.perf_counter() - start
            results[name]['time'] += elapsed
            if result == truth:
                results[name]['strict'] += 1
                results[name]['lenient'] += 1
            elif might and result == might:
                results[name]['lenient'] += 1

    print(f"{'Model':<20} {'Strict':>10} {'Lenient':>10} {'Speed':>10}")
    print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for name in models:
        r = results[name]
        strict = r['strict'] / len(data) * 100
        lenient = r['lenient'] / len(data) * 100
        speed = len(data) / r['time']
        print(f"{name:<20} {strict:>9.1f}% {lenient:>9.1f}% {speed:>7.0f}/s")

    # DKSplit top-k recall: acceptable answer (truth or might_right)
    # within the top-k candidates. Requires dksplit >= 0.4.0.
    if hasattr(dksplit, 'split_topk'):
        hits = {1: 0, 3: 0, 5: 0}
        for prefix, truth, might in data:
            accept = {truth}
            if might:
                accept.add(might)
            cands = [' '.join(c) for c in dksplit.split_topk(prefix, 5)]
            for k in hits:
                if any(c in accept for c in cands[:k]):
                    hits[k] += 1
        line = '   '.join(f"top-{k}: {hits[k]/len(data)*100:.1f}%" for k in (1, 3, 5))
        print(f"\nDKSplit top-k recall (acceptable answer within top-k candidates)")
        print(f"  {line}")

    print(f"\nDKSplit v{dksplit.__version__}")
    print(f"Samples with alternative segmentation: {sum(1 for _, _, m in data if m)}")


if __name__ == "__main__":
    main()
