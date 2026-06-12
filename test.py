"""
DKSplit test script
"""

import random
import time
import dksplit
from dksplit import Splitter

print("=" * 60)
print("DKSplit Test")
print("=" * 60)

# Test 1: Basic usage
print("\n[Test 1] Basic usage")
print("-" * 40)

tests = [
    "schawzerwald",
    "openaikey",
    "expertsexchange",
    "microsoftoffice",
    "iphone15promax",
    "machinelearningengineer",
]

for text in tests:
    result = dksplit.split(text)
    print(f"  {text:<30} -> {result}")

# Test 2: Batch
print("\n[Test 2] Batch")
print("-" * 40)

results = dksplit.split_batch(tests)
for text, result in zip(tests, results):
    print(f"  {text:<30} -> {result}")

# Test 3: Speed
print("\n[Test 3] Speed")
print("-" * 40)

test_data = tests * 1000  # 6000 items
print(f"  Data size: {len(test_data)}")

# Single
start = time.time()
for text in test_data:
    dksplit.split(text)
single_time = time.time() - start
print(f"  Single:    {single_time:.2f}s ({len(test_data)/single_time:.0f}/s)")

# Batch
start = time.time()
dksplit.split_batch(test_data)
batch_time = time.time() - start
print(f"  Batch:     {batch_time:.2f}s ({len(test_data)/batch_time:.0f}/s)")

# Test 4: Edge cases
print("\n[Test 4] Edge cases")
print("-" * 40)

edge_cases = [
    "",
    "a",
    "hello",
    "UPPERCASE",
    "123456",
    "a1b2c3",
]

for text in edge_cases:
    result = dksplit.split(text)
    print(f"  {repr(text):<30} -> {result}")

# Test 5: Top-k candidates
print("\n[Test 5] Top-k candidates")
print("-" * 40)

for text in tests:
    result = dksplit.split3(text)
    print(f"  {text:<30} -> {result}")

print()
ambiguous = ["pikahug", "noranite", "tiantian5"]
for text in ambiguous:
    result = dksplit.split5(text)
    print(f"  {text:<30} -> {result}")

# rank-1 candidate must always equal split()
ok = all(dksplit.split_topk(t, 5)[0] == dksplit.split(t) for t in tests + ambiguous)
print(f"\n  rank-1 == split(): {'OK' if ok else 'MISMATCH'}")

# Top-k edge cases: fewer candidates than k, empty input
for text in ["", "a", "ab"]:
    result = dksplit.split3(text)
    print(f"  {repr(text):<30} -> {result}")

# Test 6: split_batch must return the same results as split(), for any
# batch size and batch composition
print("\n[Test 6] Batch parity (split_batch == split)")
print("-" * 40)

# Pairs that are sensitive to batch composition under whole-batch inference
flip_pairs = [
    ("lollapalooza", "bengalsweets"),
    ("nirbhaya", "pixel3xl"),
    ("engice", "santri"),
    ("karabag", "kuznets"),
    ("gridx", "fuzzy"),
    ("passwd", "outpos"),
    ("w64", "ttx"),
    ("invoker", "mutator"),
    ("timezone", "identity"),
    ("timestamps", "allresults"),
    ("backspace", "workspace"),
    ("curdate", "related"),
]
for a, b in flip_pairs:
    assert dksplit.split_batch([a, b])[0] == dksplit.split(a), \
        f"batch parity broken for {a!r} paired with {b!r}"
print(f"  {len(flip_pairs)} batch-sensitive pairs: OK")

# Property: split_batch must equal per-item split for any batch composition
rng = random.Random(42)
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
corpus = ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 64)))
          for _ in range(300)]
expected = [dksplit.split(t) for t in corpus]
for bs in (1, 2, 17, 256):
    got = dksplit.split_batch(corpus, batch_size=bs)
    n_bad = sum(1 for e, g in zip(expected, got) if e != g)
    assert n_bad == 0, f"{n_bad} parity mismatches at batch_size={bs}"
    print(f"  300 random strings, batch_size={bs:>3}: OK")

# exact=False keeps the fast approximate path available and well-formed
approx = dksplit.split_batch(corpus, exact=False)
assert all("".join(toks) == t for t, toks in zip(corpus, approx))
print("  exact=False fast path: output well-formed")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)