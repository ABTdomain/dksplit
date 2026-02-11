# DKSplit

**Version: 0.2.3**

String segmentation using BiLSTM-CRF. Splits concatenated words into meaningful parts.

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

## Features

- **High-Fidelity Segmentation:** 95%+ accuracy on diverse inputs
- **Robust Brand/Phrase Handling:** Modern brand names and multi-language phrases
- INT8 quantized, 9MB model size
- ~800/s single, ~1700/s batch

## Requirements

- Python >= 3.8
- numpy
- onnxruntime

## Limitations

- **Supported Characters:** `a-z` and `0-9` only (auto lowercase)
- **Maximum Length:** 64 characters
- **Script Support:** Latin script only

## Links

- Website: [domainkits.com](https://domainkits.com), [ABTdomain.com](https://ABTdomain.com)
- GitHub: [github.com/ABTdomain/dksplit](https://github.com/ABTdomain/dksplit)
- Hugging Face: [huggingface.co/ABTdomain/dksplit](https://huggingface.co/ABTdomain/dksplit)
- PyPI: [pypi.org/project/dksplit](https://pypi.org/project/dksplit)

## License

[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) · Copyright 2026 ABTdomain

**Attribution:** Use DKsplit by [ABTdomain](https://abtdomain.com)