# DKSplit

> **v0.3.1** — Model upgraded to EuroHPC infrastructure (Leonardo Booster, NVIDIA A100). ~3% accuracy improvement over v0.2.x on real-world domains. API unchanged.

String segmentation using BiLSTM-CRF. Splits concatenated words into meaningful parts.

DKSplit is a lightweight model trained on millions of labeled samples covering domain names, brand names, tech terms, and multilingual phrases. It uses a BiLSTM-CRF architecture (9.47M parameters) exported to ONNX with INT8 quantization, delivering fast CPU inference in a 9 MB package.

Originally built for domain name analysis at [DomainKits](https://domainkits.com), but works well on any concatenated text: hashtags, URLs, identifiers, compound strings.

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

## What's New in v0.3.1

Model training upgraded from AWS to EuroHPC Leonardo Booster (NVIDIA A100), with optimized training configuration for better generalization. Improved accuracy on real-world domains, especially for brand names, multilingual inputs, and edge cases. The API is unchanged.
```
pip install --upgrade dksplit
```

Examples of improvements:

| Input | v0.2.x | v0.3.1 |
|---|---|---|
| `cloudflarecdn` | cloud flare cdn | **cloudflare cdn** |
| `databricks` | data bricks | **databricks** |
| `instacart` | insta cart | **instacart** |
| `robinhood` | robin hood | **robinhood** |
| `mailchimp` | mail chimp | **mailchimp** |

## Benchmark

### Dataset

1,000 newly registered .com domains randomly sampled from [ABTdomain.com](https://abtdomain.com) daily feed (April 8, 2026). No filtering or cherry-picking. Ground truth was established through multi-model cross-validation (BiLSTM, Qwen 9B LoRA, Gemma 31B) and human audit.

The dataset and evaluation script are available on [GitHub](https://github.com/ABTdomain/dksplit/tree/main/benchmark).


### Results

Accuracy on 1,000 randomly sampled real-world .com domains, human-audited ground truth:

| Model | Accuracy |
|---|---|
| **DKSplit v0.3.1** | **85.0%** |
| DKSplit v0.2.x | 82.8% |
| WordSegment | 54.0% |
| WordNinja | 46.1% |

DKSplit outperforms WordSegment by **31 percentage points** and WordNinja by **39 percentage points**.

> **Note:** The accuracy above is measured against a single reference segmentation. Domain names are inherently ambiguous. For example, `tiantian5` could be `tiantian 5` (Chinese compound name) or `tian tian 5` (two separate syllables); `noranite` could be `nora nite` or an intact brand; `pikahug` could be `pika hug` or an intact brand name. Our audit found ~5% of test samples have multiple valid segmentations. Accounting for these, effective accuracy is closer to **90%**.

### Comparison

| Input | DKSplit v0.3.1 | WordSegment | WordNinja |
|---|---|---|---|
| `chatgptprompts` | **chatgpt prompts** | chat gpt prompts | chat gp t prompts |
| `tensorflowserving` | **tensorflow serving** | tensor flow serving | tensor flow serving |
| `spotifywrapped` | **spotify wrapped** | spot if y wrapped | spot if y wrapped |
| `ethereumwallet` | **ethereum wallet** | e there um wallet | e there um wallet |
| `cloudflarecdn` | **cloudflare cdn** | cloud flare cdn | cloud flare cd n |
| `kubernetescluster` | **kubernetes cluster** | ku bernet es cluster | ku berne tes cluster |
| `hackathonwinners` | **hackathon winners** | hackathon winners | hack a th on winners |
| `whatsappstatus` | **whatsapp status** | what sapp status | what s app status |
| `drwatsonai` | **dr watson ai** | dr watson a i | dr watson a i |
| `escribirenvozalta` | **escribir en voz alta** | escribir env oz alta | es crib ire nv oz alta |
| `tuvasou` | **tu vas ou** | tuva sou | tuva so u |
| `candidiasenuncamais` | **candidiase nunca mais** | candid iase nunca mais | can didi as e nun cama is |
| `robertdeniro` | **robert de niro** | robert deniro | robert deniro |
| `mercibeaucoup` | **merci beaucoup** | merci beaucoup | mer ci beau coup |

## How It Works

DKSplit treats segmentation as a sequence labeling task.

The training data includes:
- LLM-labeled domain name segmentations
- Brand names
- Personal name combinations
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

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

**Please attribute as:** DKsplit by [ABTdomain](https://abtdomain.com)

## Acknowledgements

<a href="https://eurohpc-ju.europa.eu/"><img src="https://raw.githubusercontent.com/ABTdomain/dksplit/main/docs/images/eurohpc-logo.png" alt="EuroHPC JU" width="80"></a> &nbsp; <a href="https://commission.europa.eu/"><img src="https://raw.githubusercontent.com/ABTdomain/dksplit/main/docs/images/eu-cofunded-logo.png" alt="Co-funded by the EU" width="200"></a>

The v0.3.1 model was trained on the [Leonardo Booster](https://www.hpc.cineca.it/systems/hardware/leonardo/) supercomputer at CINECA, Italy, with computing resources provided by the [EuroHPC Joint Undertaking](https://eurohpc-ju.europa.eu/) through the Playground Access program (project AIFAC_P02_281). We thank EuroHPC JU for enabling SMEs to explore new possibilities with world-class HPC infrastructure.