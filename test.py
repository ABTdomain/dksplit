"""
DKSplit test script
"""

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

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)