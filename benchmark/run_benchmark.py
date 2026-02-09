"""
DKSplit Benchmark - 1,000 real newly registered domains
Reference: GPT-5.2 segmentation
Data source: ABTdomain.com daily feed (February 8, 2026), random sample
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
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                data.append((row[0].strip(), row[1].strip()))

    print(f"Benchmark: {len(data)} real newly registered domains")
    print(f"Reference: GPT-5.2 segmentation")
    print(f"Source: ABTdomain.com daily feed (Feb 8, 2026), random sample\n")

    models = {
        'DKSplit': lambda t: ' '.join(dksplit.split(t)),
        'WordSegment': lambda t: ' '.join(wordsegment.segment(t)),
        'WordNinja': lambda t: ' '.join(wordninja.split(t)),
    }

    results = {name: {'correct': 0, 'time': 0} for name in models}

    for prefix, gpt_answer in data:
        for name, fn in models.items():
            start = time.perf_counter()
            result = fn(prefix)
            elapsed = time.perf_counter() - start
            results[name]['time'] += elapsed
            if result == gpt_answer:
                results[name]['correct'] += 1

    print(f"{'Model':<20} {'Accuracy':>10} {'Correct':>10} {'Speed':>10}")
    print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for name in models:
        r = results[name]
        acc = r['correct'] / len(data) * 100
        speed = len(data) / r['time']
        print(f"{name:<20} {acc:>9.1f}% {r['correct']:>7}/{len(data)}  {speed:>7.0f}/s")

    print(f"\nDKSplit v{dksplit.__version__}")


if __name__ == "__main__":
    main()