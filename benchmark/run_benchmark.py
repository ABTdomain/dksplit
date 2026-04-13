"""
DKSplit Benchmark - 1,000 real newly registered domains
Reference: Multi-model cross-validation + human audit
Data source: ABTdomain.com daily feed (April 8, 2026), random sample
"""

import csv
import time
from pathlib import Path

def main():
    import dksplit
    import wordsegment
    import wordninja

    wordsegment.load()

    # Load data
    csv_path = Path(__file__).parent / "sample_1000.csv"
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            truth = row['input'].strip(), row['truth'].strip().lower()
            second = row.get('2nd', '').strip().lower()
            data.append((row['input'].strip(), row['truth'].strip().lower(), second))

    print(f"Benchmark: {len(data)} real newly registered domains")
    print(f"Reference: Multi-model cross-validation + human audit")
    print(f"Source: ABTdomain.com daily feed (Apr 8, 2026), random sample\n")

    models = {
        'DKSplit': lambda t: ' '.join(dksplit.split(t)),
        'WordSegment': lambda t: ' '.join(wordsegment.segment(t)),
        'WordNinja': lambda t: ' '.join(wordninja.split(t)),
    }

    results = {name: {'strict': 0, 'lenient': 0, 'time': 0} for name in models}

    for prefix, truth, second in data:
        for name, fn in models.items():
            start = time.perf_counter()
            result = fn(prefix)
            elapsed = time.perf_counter() - start
            results[name]['time'] += elapsed
            if result == truth:
                results[name]['strict'] += 1
                results[name]['lenient'] += 1
            elif second and result == second:
                results[name]['lenient'] += 1

    print(f"{'Model':<20} {'Strict':>10} {'Lenient':>10} {'Speed':>10}")
    print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for name in models:
        r = results[name]
        strict = r['strict'] / len(data) * 100
        lenient = r['lenient'] / len(data) * 100
        speed = len(data) / r['time']
        print(f"{name:<20} {strict:>9.1f}% {lenient:>9.1f}% {speed:>7.0f}/s")

    print(f"\nDKSplit v{dksplit.__version__}")
    print(f"Samples with alternative segmentation: {sum(1 for _,_,s in data if s)}")


if __name__ == "__main__":
    main()
