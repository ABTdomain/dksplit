# DKSplit

Fast character-level segmentation for web-style concatenated strings —
domain names, hashtags, usernames, slugs. 9 MB ONNX model, CPU-only.

```bash
pip install dksplit
```

Requires Python >= 3.8. Dependencies: numpy, onnxruntime.

## Usage

```python
import dksplit

# Single best segmentation
dksplit.split("kubernetescluster")
# ['kubernetes', 'cluster']

# Batch (faster for large volumes; results identical to split())
dksplit.split_batch(["openaikey", "microsoftoffice", "bitcoinprice"])
# [['openai', 'key'], ['microsoft', 'office'], ['bitcoin', 'price']]

# Ranked candidates for ambiguous inputs
dksplit.split3("noranite")        # top-3, best first
# [['nora', 'nite'], ['noranite'], ['nor', 'anite']]

dksplit.split5("pikahug")         # top-5
# [['pikahug'], ['pika', 'hug'], ['pik', 'ahug'], ['pikah', 'ug'], ['pi', 'kahug']]

dksplit.split_topk("chatgptlogin", k=3)   # any k
# [['chatgpt', 'login'], ['chatgptlogin'], ['chatgpt', 'log', 'in']]
```

## What can you do with it

Typical uses: spotting brands and lookalikes in newly registered domains
(`yourbrandlogin`, `getyourbrand`), extracting keywords from domains, hashtags,
and URLs, normalizing concatenated identifiers before matching and dedup,
understanding spaceless search queries.

- **`split()`** — one answer per input; pipelines, aggregation, statistics.
- **`split_topk()`** — ranked candidates for recall-sensitive matching or for
  reranking with your own signals (brand lists, frequency data); an acceptable
  segmentation is in the top-3 candidates 98.5% of the time (top-5: 99.3%).

## What's New in v1.0.2

Bugfix: `split_batch()` could differ from `split()` on rare inputs; results
are now guaranteed identical. Pass `exact=False` to keep the old ~2x faster
behavior.

## Benchmark

### Dataset

1,000 hand-audited domain prefixes drawn from the
[Newly Registered Domains Database (NRDS)](https://domainkits.com/download/nrds)
(.com feed). No filtering or cherry-picking on segmentation difficulty. Ground
truth was established through multi-model cross-validation (BiLSTM, Qwen 9B
LoRA, Gemma 31B) and human audit. Each row provides a primary `truth` and an
optional `might_right` field for genuinely ambiguous cases (e.g.
brand-versus-compound).

Both benchmark sets ship in this repo's
[`/benchmark`](https://github.com/ABTdomain/dksplit/tree/main/benchmark)
directory: `sample_1000.csv` and `benchmark_5000.csv`, a larger set built the
same way (also on Hugging Face as
[ABTdomain/dksplit-benchmark](https://huggingface.co/datasets/ABTdomain/dksplit-benchmark)).
To explore domain data yourself, register at
[domainkits.com](https://domainkits.com) — fresh .com NRD downloads are free.

### Results

| Model | Strict EM | Lenient EM |
|---|---|---|
| **DKSplit v1.0.2** | **86.5%** | **91.5%** |
| WordSegment | 65.2% | 69.5% |
| WordNinja | 51.0% | 54.0% |

Strict EM counts only exact matches against `truth`. Lenient EM also accepts
the `might_right` alternative when present.

Top-k coverage (an acceptable segmentation is present within the candidates):

| Benchmark | top-1 | top-3 | top-5 |
|---|---|---|---|
| 1,000 samples | 91.5% | 98.5% | 99.3% |
| 5,000 samples | 90.4% | 97.8% | 99.0% |

### Reproduce it yourself

```bash
git clone https://github.com/ABTdomain/dksplit.git
cd dksplit/benchmark
pip install dksplit wordsegment wordninja
python run_benchmark.py                     # 1,000-sample set
python run_benchmark.py benchmark_5000.csv  # 5,000-sample set
```

Adding your own segmenter to the comparison is a one-line change in
`run_benchmark.py`. Pull requests for ambiguous samples are welcome.

### Comparison

| Input | DKSplit v1.0.2 | WordSegment | WordNinja |
|---|---|---|---|
| `chatgptprompts` | **chatgpt prompts** | chat gpt prompts | chat gp t prompts |
| `spotifywrapped` | **spotify wrapped** | spot if y wrapped | spot if y wrapped |
| `ethereumwallet` | **ethereum wallet** | e there um wallet | e there um wallet |
| `kubernetescluster` | **kubernetes cluster** | ku bernet es cluster | ku berne tes cluster |
| `whatsappstatus` | **whatsapp status** | what sapp status | what s app status |
| `drwatsonai` | **dr watson ai** | dr watson a i | dr watson a i |
| `escribirenvozalta` | **escribir en voz alta** | escribir env oz alta | es crib ire nv oz alta |
| `tuvasou` | **tu vas ou** | tuva sou | tuva so u |
| `candidiasenuncamais` | **candidiase nunca mais** | candid iase nunca mais | can didi as e nun cama is |

## How It Works

DKSplit treats segmentation as a character-level sequence labeling task. The
training data includes LLM-labeled domain segmentations, brand names, personal
name combinations, multilingual phrases (English, French, German, Spanish, and
more), and tech product names. At inference, the BiLSTM runs as an
INT8-quantized ONNX model and CRF decoding is performed in NumPy. No GPU
required; around 800 samples per second on a single CPU thread.

**Why BiLSTM-CRF:** character precision, CPU-only inference, a 9 MB
artifact — built for millions of strings per day. Design rationale and
failure-mode comparisons (dictionary segmenters, DeBERTa-V3, LLMs):
[blog post](https://abtdomain.com/blog/2026/04/dksplit-update-cleaner-benchmark-first-deberta-run-different-failure-modes/).

## Features

- **Brand-aware:** recognizes thousands of brands, tech products, and proper nouns
- **Multilingual:** English, French, German, Spanish, and romanized text
- **Lightweight:** 9 MB model, minimal dependencies (numpy + onnxruntime)
- **Offline:** no API keys, no internet required
- **Top-k candidates:** `split3` / `split5` / `split_topk` return ranked
  alternative segmentations

## Limitations

- **Characters:** `a-z` and `0-9`, auto-lowercased. For best results pass
  letter-only runs: split off digits and separators (`-`, `.`, `_`) with
  simple rules first — those boundaries are a job for rules, not the model.
- **Max length:** 64 characters.
- **Script:** Latin script only. Non-Latin scripts (汉字, かな, 한글, العربية)
  are not supported.
- **Ambiguity:** some inputs are genuinely ambiguous. `split()` optimizes for
  the most common interpretation; use `split_topk()` when you need the
  alternatives.
- **Rare languages:** accuracy is highest on English and major European languages.

## Links

- Website: [domainkits.com](https://domainkits.com), [ABTdomain.com](https://ABTdomain.com)
- PyPI: [pypi.org/project/dksplit](https://pypi.org/project/dksplit)
- Hugging Face (LLM variant): [ABTdomain/dksplit-qwen-lora](https://huggingface.co/ABTdomain/dksplit-qwen-lora)
- Issues: [GitHub Issues](https://github.com/ABTdomain/dksplit/issues)

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Attribution required: credit "DKSplit by [ABTdomain](https://abtdomain.com)"
in your README, documentation, about page, or API response metadata.

## Acknowledgements

<a href="https://eurohpc-ju.europa.eu/"><img src="https://raw.githubusercontent.com/ABTdomain/dksplit/main/docs/images/eurohpc-logo.png" alt="EuroHPC JU" width="80"></a> &nbsp; <a href="https://commission.europa.eu/"><img src="https://raw.githubusercontent.com/ABTdomain/dksplit/main/docs/images/eu-cofunded-logo.png" alt="Co-funded by the EU" width="200"></a>

The model was trained on the
[Leonardo Booster](https://www.hpc.cineca.it/systems/hardware/leonardo/)
supercomputer at CINECA, Italy, with computing resources provided by the
[EuroHPC Joint Undertaking](https://eurohpc-ju.europa.eu/) through the
Playground Access program (EHPC-AIF-2026PG01-281). We thank EuroHPC JU for
enabling SMEs to explore new possibilities with world-class HPC infrastructure.