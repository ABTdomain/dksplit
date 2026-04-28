# DKSplit

> **v0.3.1**: Model upgraded to EuroHPC infrastructure (Leonardo Booster, NVIDIA A100). ~3% accuracy improvement over v0.2.x on real-world domains. API unchanged.

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

1,000 hand-audited domain prefixes drawn from the [Newly Registered Domains Database (NRDS)](https://domainkits.com/download/nrds) (.com feed). No filtering or cherry-picking on segmentation difficulty. Ground truth was established through multi-model cross-validation (BiLSTM, Qwen 9B LoRA, Gemma 31B) and human audit. Each row provides a primary `truth` and an optional `might_right` field for genuinely ambiguous cases (e.g. brand-versus-compound).

The dataset and evaluation script are available on [GitHub](https://github.com/ABTdomain/dksplit/tree/main/benchmark). For the methodology behind this benchmark and the broader model comparison, see the [DKSplit Update blog post](https://abtdomain.com/blog/2026/04/dksplit-update-cleaner-benchmark-first-deberta-run-different-failure-modes/).

### What changed in this benchmark

The numbers below differ from what we previously reported because the benchmark itself changed. Two adjustments matter:

- **Removed samples that did not really test segmentation.** The earlier set carried digit-driven inputs (e.g. `824fisher`, where the digits already provide the boundary) and pure-noise consonant strings (e.g. `hbwhjhzx`, where any output is a guess). We now handle both with deterministic rules in production rather than asking the model to score on them. Removing them stops inflating accuracy with cases the model wasn't actually being tested on.
- **Added a `might_right` column for genuinely ambiguous cases.** Strings like `pikahug` or `noranite` can plausibly be either a brand kept whole or a phrase split apart. Instead of forcing one answer, we accept either (Lenient EM). This makes the lenient score a more honest reflection of how the model performs on real, ambiguous data.

The net effect is a tighter, harder benchmark: the easy-but-uninformative samples are gone, the genuinely ambiguous ones are scored fairly, and what remains focuses on the part of the task that actually matters, which is deciding word boundaries inside concatenated language.

#### Why these changes aren't self-serving

Removing digit-driven and pure-noise samples hurts every model in absolute terms, including DKSplit. Those were "easy" cases everyone got right. The shift in DKSplit's relative position comes from the harder ambiguous cases that remain in the test set, not from selectively dropping cases DKSplit was wrong on. The `might_right` field is reserved for genuinely ambiguous segmentations (brand-versus-compound, pinyin-versus-name), not for any model's output specifically. Where any model's prediction matches `might_right`, it's because that segmentation was already considered acceptable, not because we accepted it after the fact.

The benchmark and the evaluation script are open. You can verify all of the above yourself.

#### How to use the benchmark

You can run the evaluation locally on the same 1,000 samples to reproduce or extend these numbers:

```bash
git clone https://github.com/ABTdomain/dksplit.git
cd dksplit/benchmark
pip install dksplit wordsegment wordninja
python run_benchmark.py
```

This is also useful if you want to:

- **Compare your own segmenter** against DKSplit, WordSegment, and WordNinja on the same set
- **Try a different labeling preference** by re-auditing `truth` and `might_right` for your own use case (SEO recall, strict brand protection, etc.). The benchmark is structured so the same data can be re-labeled without rerunning the evaluation logic.
- **Spot edge cases** by inspecting `sample_1000.csv` directly; pull requests for ambiguous samples we got wrong are welcome

### Results

Accuracy on the 1,000-sample benchmark above:

| Model | Strict EM | Lenient EM |
|---|---|---|
| **DKSplit v0.3.1** | **86.5%** | **91.5%** |
| WordSegment | 65.1% | 69.4% |
| WordNinja | 50.9% | 53.9% |

Strict EM counts only exact matches against `truth`. Lenient EM also accepts the `might_right` alternative when present. DKSplit outperforms WordSegment by 21+ percentage points and WordNinja by 35+ percentage points on both measures.

> **Note:** Domain names are inherently ambiguous. For example, `tiantian5` could be `tiantian 5` (Chinese compound name) or `tian tian 5` (two separate syllables); `noranite` could be `nora nite` or an intact brand; `pikahug` could be `pika hug` or an intact brand name. The Lenient EM column above reflects the cases where multiple segmentations are accepted as correct.

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

At inference, the BiLSTM runs as an INT8-quantized ONNX model and CRF decoding is performed in NumPy. No GPU required.

### Why BiLSTM-CRF

For domain segmentation specifically, character-level BiLSTM-CRF turned out to be a good fit. Subword transformers (e.g. DeBERTa) work at a granularity coarser than this task needs, so a single subword token can span a real word boundary and lose the signal. Dictionary-based segmenters (WordSegment, WordNinja) are great on standard English but break down on newly registered domains, which are mostly brand coinages, multilingual compounds, and intentional misspellings. Large language models can be accurate but are cost-prohibitive at the volume we run (millions of domains per day). BiLSTM-CRF gives us character-level precision with CPU-only inference at around 800 samples per second on a single thread, and a 9 MB deployable artifact.

For more on this trade-off and a head-to-head failure-mode comparison with DeBERTa-V3 and the dictionary baselines, see our [DKSplit Update blog post](https://abtdomain.com/blog/2026/04/dksplit-update-cleaner-benchmark-first-deberta-run-different-failure-modes/).

## Features

- **Brand-aware:** Recognizes thousands of brands, tech products, and proper nouns
- **Multilingual:** Handles English, French, German, Spanish, and romanized text
- **Lightweight:** 9 MB model, minimal dependencies (numpy + onnxruntime)
- **Offline:** No API keys, no internet required

## Used in Production

DKSplit powers the keyword extraction layer behind [DomainKits Keywords Trends](https://domainkits.com/trends/keywords). Every newly registered domain is segmented by DKSplit, then the extracted keywords go through a data-cleaning pass before being aggregated into trend signals. The same cleaned keyword stream feeds several different analyses: hot keywords (currently active), emerging keywords (rising from a low base), and TLD-specific or registrar-specific cuts of the same data.

Combining segmented keywords with the rest of a domain's metadata (TLD, registrar, registration date, WHOIS fields) opens up a range of downstream uses:

- **Brand protection:** spot newly registered domains containing your brand or close lookalikes
- **Keyword tracking:** follow how a topic, technology, or trend shows up in newly registered domains over time
- **Domain investment research:** surface keyword clusters that are gaining traction before they become obvious
- **Market intelligence:** see which categories and verticals are growing in domain registration activity

The accuracy of all of these depends on the segmentation step being correct on novel, never-seen-before domain strings. That is what DKSplit is built for, and why we keep iterating on it.

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