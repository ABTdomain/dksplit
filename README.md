# DKSplit
[![PyPI](https://img.shields.io/pypi/v/dksplit)](https://pypi.org/project/dksplit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Word segmentation library for concatenated text. Split domain names, brand names, and phrases into words.


## About

DKSplit is developed by [ABTdomain](https://ABTdomain.com), originally built for [DomainKits](https://domainkits.com) - a domain platform.

The model is trained on millions of labeled samples covering domain names, brand names, tech terms, and multi-language phrases. It uses a BiLSTM-CRF architecture (384 embedding, 768 hidden, 3 layers) and is exported to ONNX format with INT8 quantization for fast, lightweight inference.

Originally designed for domain name segmentation, but works well on:
- Brand names: `chatgptlogin` → `chatgpt login`
- Tech terms: `kubernetescluster` → `kubernetes cluster`
- Multi-language phrases: `mercibeaucoup` → `merci beaucoup`

## Install
```bash
pip install dksplit
```

## Usage
```python
import dksplit

# Single
dksplit.split("chatgptlogin")
# ['chatgpt', 'login']

# Batch
dksplit.split_batch(["openaikey", "microsoftoffice"])
# [['openai', 'key'], ['microsoft', 'office']]
```

## Comparison

| Input | DKSplit | WordNinja |
|-------|---------|-----------|
| chatgptlogin | chatgpt login | chat gp t login |
| kubernetescluster | kubernetes cluster | ku berne tes cluster |

## Real World Benchmark

Tested on [Majestic Million](https://majestic.com/reports/majestic-million) domains:

| Input | Output |
|-------|--------|
| amitriptylineinfo | amitriptyline info |
| autoriteprotectiondonnees | autorite protection donnees |
| mountaingoatsoftware | mountain goat software |
| psychologytoday | psychology today |
| affordablecollegesonline | affordable colleges online |
| stephenwolfram | stephen wolfram |
| ralphlauren | ralphlauren |


## Features

- **High-Fidelity Segmentation:** 95%+ accuracy on a diverse range of inputs, from technical identifiers to concatenated common phrases.
- **Robust Brand/Phrase Handling:** Accurately segments new or ambiguous cases, including modern brand names and multi-word phrases (e.g., in English, German, French, etc.).
- INT8 quantized, 9MB model size
- ~800/s single, ~1700/s batch
- **Continuous Improvement:** The model is subject to periodic updates on the GitHub repository to incorporate new vocabulary and address discovered edge cases. 
**Tip:** Pre/post-processing with a custom dictionary can improve accuracy for specialized terms.

## Requirements

- Python >= 3.8
- numpy
- onnxruntime

## Limitations


- **Supported Characters:** Input must be composed of `a-z` and `0-9` only. All characters are automatically converted to lowercase before processing. (Non-alphanumeric characters, spaces, and special symbols are not supported.)
- **Maximum Length:** The model is optimized for short identifiers and phrases, supporting a maximum input length of **64 characters**.
- **Script Support:** Only **Latin script** (including Romanized forms of CJK/Arabic) is supported. Non-Latin scripts (e.g., 汉字, かな, 한글, العربية) will produce unpredictable results.
- **Ambiguity/New Entities:** While highly accurate, the model may occasionally mis-segment very new or highly specialized technical entities (e.g., `cloud` `flare` `status` instead of `cloudflare` `status`).
- **Accuracy Target:** The model is optimized for high speed and low size (9MB). While its accuracy is **high**, it is not designed to match the near-perfect accuracy of slow, high-cost large language models (LLMs).


## Links

- Website: [ABTdomain.com](https://ABTdomain.com)
- Use Case: [domainkits.com](https://domainkits.com)
- Go version: [github.com/ABTdomain/dksplit-go](https://github.com/ABTdomain/dksplit-go)
- Documentation: [dksplit.readthedocs.io](https://dksplit.readthedocs.io)
- PyPI: [pypi.org/project/dksplit](https://pypi.org/project/dksplit)
- Issues: [GitHub Issues](https://github.com/ABTdomain/dksplit/issues)

## License

MIT