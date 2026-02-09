# DKSplit

> **v0.2.0** — Retrained model with expanded brand and name coverage. Accuracy improved from 75.4% to 80.5% on real-world domains. API unchanged — just `pip install --upgrade dksplit`.

String segmentation using BiLSTM-CRF. Splits concatenated words into meaningful parts.

DKSplit is a lightweight model trained on millions of labeled samples covering domain names, brand names, tech terms, and multilingual phrases. It uses a BiLSTM-CRF architecture (9.47M parameters) exported to ONNX with INT8 quantization, delivering fast CPU inference in a 9 MB package.

Originally built for domain name analysis at [DomainKits](https://domainkits.com), but works well on any concatenated text — hashtags, URLs, identifiers, compound strings.

## Install
```
pip install dksplit
```

## Usage
```python
import dksplit

dksplit.split("chatgptlogin")
# ['chatgpt', 'login']

dksplit.split("kubernetescluster")
# ['kubernetes', 'cluster']

dksplit.split("mercibeaucoup")
# ['merci', 'beaucoup']

dksplit.split_batch(["openaikey", "microsoftoffice", "bitcoinprice"])
# [['openai', 'key'], ['microsoft', 'office'], ['bitcoin', 'price']]
```

## What's New in v0.2.0

Retrained model with significantly expanded brand and name coverage. The API is unchanged — just upgrade.
```
pip install --upgrade dksplit
```

Accuracy on 1,000 real domains (details in [Benchmark](#benchmark)):

| Version | Accuracy |
|---|---|
| **v0.2.0** | **80.5%** |
| v0.1.0 | 75.4% |

Examples of improvements:

| Input | v0.1.0 | v0.2.0 |
|---|---|---|
| `cloudflarecdn` | cloud flare cdn | **cloudflare cdn** |
| `snowdenyes` | snow den yes | **snowden yes** |
| `robertdeniro` | robert deniro | **robert de niro** |

## Benchmark

### Dataset

1,000 newly registered .com domains randomly sampled from [ABTdomain.com](https://abtdomain.com) daily feed (February 8, 2026). No filtering or cherry-picking — a raw random sample of real-world domain registrations. GPT-5.2 segmentation is used as the reference answer.

The dataset and evaluation script are included in this repository:
```
benchmark/
├── sample_1000.csv        # 1,000 domains with GPT-5.2 reference labels
└── run_benchmark.py       # Evaluation script
```
```
pip install dksplit wordsegment wordninja
python benchmark/run_benchmark.py

```

### Results

| Model | Parameters | Accuracy | Speed | Size | Cost |
|---|---|---|---|---|---|
| GPT-5.2 | Trillion+ | reference | ~2/s | — | $0.19/1K |
| **DKSplit v0.2.0** | **9.47M** | **80.5%** | **508/s** | **9 MB** | **Free** |
| WordSegment | — | 59.1% | 1,008/s | — | Free |
| WordNinja | — | 47.6% | 9,082/s | — | Free |

DKSplit outperforms WordSegment by **21 percentage points** and WordNinja by **33 percentage points**.

Compared to GPT-5.2, DKSplit achieves **80.5% agreement** while running **250x faster** at **zero cost** — making it practical for bulk processing where calling an LLM on every input is not feasible.

> **Note:** The remaining ~20% disagreement with GPT-5.2 largely comes from rare languages, invented words, and genuinely ambiguous cases (e.g., is `christianalucas` → `christiana lucas` or `christian a lucas`?). On standard English inputs, agreement is significantly higher.

### Comparison

| Input | DKSplit v0.2.0 | WordSegment | WordNinja |
|---|---|---|---|
| `chatgptlogin` | **chatgpt login** | chat gpt login | chat gp t login |
| `cloudflarecdn` | **cloudflare cdn** | cloud flare cdn | cloud flare cd n |
| `kubernetescluster` | **kubernetes cluster** | ku bernet es cluster | ku berne tes cluster |
| `instagramlogin` | **instagram login** | insta gram login | insta gram login |
| `ethereumwallet` | **ethereum wallet** | e there um wallet | e there um wallet |
| `spotifyplaylist` | **spotify playlist** | spot if y playlist | spot if y playlist |
| `lululemonoutlet` | **lululemon outlet** | lululemon outlet | lulu lemon outlet |
| `tensorflowlite` | **tensorflow lite** | tensor flow lite | tensor flow lite |
| `mercibeaucoup` | **merci beaucoup** | merci beaucoup | mer ci beau coup |
| `robertdeniro` | **robert de niro** | robert deniro | robert deniro |
| `snowdenyes` | **snowden yes** | snowden yes | snow deny es |
| `youtubedownloader` | **youtube downloader** | youtube downloader | youtube down loader |

## How It Works

DKSplit treats segmentation as a sequence labeling task. Each character receives a label: `1` (start of a new word) or `0` (continuation). A bidirectional LSTM reads the full string in both directions, and a CRF layer enforces valid transitions between labels.

The training data includes:
- LLM-labeled domain name segmentations
- Brand names sourced from Wikidata (companies, software, websites, games, cryptocurrencies)
- Global personal name combinations
- Multilingual phrases (English, French, German, Spanish, and more)
- Tech product names and terminology

At inference, the BiLSTM runs as an INT8-quantized ONNX model and CRF decoding is performed in NumPy — no GPU required.

## Features

- **High accuracy:** 80.5% agreement with GPT-5.2 on real domains, 21pp above WordSegment
- **Fast:** 500+ segmentations/second on CPU
- **Lightweight:** 9 MB model, minimal dependencies (numpy + onnxruntime)
- **Brand-aware:** Recognizes thousands of brands, tech products, and proper nouns
- **Multilingual:** Handles English, French, German, Spanish, and romanized text
- **Offline:** No API keys, no internet required

## Limitations

- **Characters:** Only `a-z` and `0-9`. Input is automatically lowercased.
- **Max length:** 64 characters.
- **Script:** Latin script only. Non-Latin scripts (汉字, かな, 한글, العربية) are not supported.
- **Ambiguity:** Some inputs are genuinely ambiguous. DKSplit optimizes for the most common interpretation.
- **Rare languages:** Accuracy is highest on English and major European languages.

## Requirements

- Python >= 3.8
- numpy
- onnxruntime

## Links

- Website: [domainkits.com](https://domainkits.com), [ABTdomain.com](https://ABTdomain.com)
- GitHub: [github.com/ABTdomain/dksplit](https://github.com/ABTdomain/dksplit)
- PyPI: [pypi.org/project/dksplit](https://pypi.org/project/dksplit)
- Issues: [GitHub Issues](https://github.com/ABTdomain/dksplit/issues)

## License

MIT