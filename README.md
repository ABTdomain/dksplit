# DKSplit

> **v0.2.0** — Retrained model with expanded brand and name coverage. ~7% accuracy improvement on real-world domains. API unchanged — just `pip install --upgrade dksplit`.

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

Accuracy (agreement with GPT-5.2 on 1,000 real domains):

| Model | Accuracy |
|---|---|
| **DKSplit v0.2.0** | **80.5%** |
| WordSegment | 59.1% |
| WordNinja | 47.6% |

DKSplit outperforms WordSegment by **21 percentage points** and WordNinja by **33 percentage points**.

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

DKSplit treats segmentation as a sequence labeling task.

The training data includes:
- LLM-labeled domain name segmentations
- Brand names sourced from Wikidata (companies, software, websites, games, cryptocurrencies)
- Global personal name combinations
- Multilingual phrases (English, French, German, Spanish, and more)
- Tech product names and terminology

At inference, the BiLSTM runs as an INT8-quantized ONNX model and CRF decoding is performed in NumPy — no GPU required.

## Features

- **Brand-aware:** Recognizes thousands of brands, tech products, and proper nouns
- **Multilingual:** Handles English, French, German, Spanish, and romanized text
- **Lightweight:** 9 MB model, minimal dependencies (numpy + onnxruntime)
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