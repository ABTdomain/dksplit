Requires Python >= 3.8. Dependencies: numpy, onnxruntime.

## Usage

```python
import dksplit

# Single best segmentation
dksplit.split("kubernetescluster")
# ['kubernetes', 'cluster']

# Batch (faster for large volumes)
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

Anywhere you need to turn unspaced strings into words, offline and at volume:

- **Brand protection & lookalike detection.** Segment newly registered domains
  to spot ones containing your brand or close variants (`yourbrandlogin`,
  `getyourbrand`). Use `split_topk` so a brand name split across candidates
  never slips through: matching against all k candidates raises recall.
- **SEO & keyword extraction.** Extract keywords from domain names, URLs, and
  hashtags for trend analysis, content research, or search indexing.
- **Domain investment research.** Decompose large domain lists into keyword
  signals: which words are being registered, in which combinations, on which TLDs.
- **Data cleaning & entity resolution.** Normalize concatenated identifiers,
  usernames, or product codes before matching and deduplication.
- **Search & autocomplete.** Understand queries typed without spaces.

Two ways to consume the output, depending on your task:

- **`split()`** when you need one answer per input: pipelines, aggregation,
  statistics.
- **`split_topk()`** when the input may be ambiguous and you can use a ranked
  shortlist: recall-sensitive matching, human review UIs, or feeding your own
  reranker with domain-specific signals (brand lists, frequency data, metadata).
  An acceptable segmentation is present in the top-3 candidates 98.5% of the
  time, and in the top-5 99.3% of the time (see benchmark below).

## What's New in v1.0.0

First stable release.

- **Stability guarantees:** the model weights are frozen (identical input →
  identical output across v1.x), and the public API (`split`, `split_batch`,
  `split_topk` / `split3` / `split5`) will not break within v1.x (SemVer).
- **Top-k API:** instead of forcing one answer on genuinely ambiguous inputs
  (is `noranite` a brand, or `nora nite`?), `split_topk` returns a small ranked
  set of candidates. Decoded with k-best Viterbi over the same CRF: no model
  change, no new dependencies, only a small speed overhead. Inputs with fewer
  than k possible segmentations return fewer candidates.
- **Why the model is frozen:** at 9 MB and CPU-only it is the right
  accuracy/speed/cost trade-off for high-volume use. Larger models score
  somewhat higher but cost orders of magnitude more compute per string, so we
  froze this model as a stable baseline and put further gains into the
  candidate layer instead of a heavier model.

## Benchmark

> **Scope:** all numbers below are measured on this PyPI package
> (`dksplit==1.0.0`) running standalone, with the dataset and script published
> in [`/benchmark`](https://github.com/ABTdomain/dksplit/tree/main/benchmark).
> You can reproduce them locally.

> **Note:** the benchmark methodology changed in v1.0.0 (samples that didn't
> test segmentation were removed; a `might_right` field was added for genuinely
> ambiguous cases). Numbers are **not comparable** to previously reported
> results. See the [v1.0.0 release notes](https://github.com/ABTdomain/dksplit/releases/tag/v1.0.0)
> and the [methodology blog post](https://abtdomain.com/blog/2026/04/dksplit-update-cleaner-benchmark-first-deberta-run-different-failure-modes/)
> for details.

### Dataset

1,000 hand-audited domain prefixes drawn from the
[Newly Registered Domains Database (NRDS)](https://domainkits.com/download/nrds)
(.com feed). No filtering or cherry-picking on segmentation difficulty. Ground
truth was established through multi-model cross-validation (BiLSTM, Qwen 9B
LoRA, Gemma 31B) and human audit. Each row provides a primary `truth` and an
optional `might_right` field for genuinely ambiguous cases (e.g.
brand-versus-compound).

This benchmark is multi-audited, but it is only a reference point. No fixed
test set can cover every brand coinage, multilingual compound, or naming
convention in real registrations, and we make no claim of 100% coverage. The
honest way to judge DKSplit is on your own data: download a fresh batch of
newly registered domains from
[domainkits.com/download/nrds](https://domainkits.com/download/nrds) (free,
domain-name-only files) and run them through it.

### Results

| Model | Strict EM | Lenient EM |
|---|---|---|
| **DKSplit v1.0.0** | **86.5%** | **91.5%** |
| WordSegment | 65.2% | 69.5% |
| WordNinja | 51.0% | 54.0% |

Strict EM counts only exact matches against `truth`. Lenient EM also accepts
the `might_right` alternative when present.

Top-k coverage (an acceptable segmentation is present within the candidates):

| Benchmark | top-1 | top-3 | top-5 |
|---|---|---|---|
| 1,000 samples | 91.5% | 98.5% | 99.3% |
| 5,000 samples | 90.4% | 97.8% | 99.0% |

The 5,000-sample set is a larger benchmark built the same way as the
1,000-sample set: same NRDS .com source, same multi-model cross-validation and
human audit, same `truth` / `might_right` format. It ships in the same
[`/benchmark`](https://github.com/ABTdomain/dksplit/tree/main/benchmark)
directory as `benchmark_5000.csv` and is also published on Hugging Face as
[ABTdomain/dksplit-benchmark](https://huggingface.co/datasets/ABTdomain/dksplit-benchmark).

> Domain names are inherently ambiguous: `tiantian5` could be `tiantian 5`
> (compound name) or `tian tian 5`; `noranite` could be `nora nite` or an
> intact brand. The Lenient EM column and the `might_right` field exist to
> score such cases fairly.

### Reproduce it yourself

```bash
git clone https://github.com/ABTdomain/dksplit.git
cd dksplit/benchmark
pip install dksplit wordsegment wordninja
python run_benchmark.py                     # 1,000-sample set
python run_benchmark.py benchmark_5000.csv  # 5,000-sample set
```

Also useful if you want to:

- **Compare your own segmenter** against DKSplit, WordSegment, and WordNinja
  on the same set
- **Re-label for your use case** (SEO recall, strict brand protection, etc.):
  the benchmark is structured so the same data can be re-audited without
  changing the evaluation logic
- **Spot edge cases** by inspecting `sample_1000.csv` directly; pull requests
  for ambiguous samples we got wrong are welcome

### Comparison

| Input | DKSplit v1.0.0 | WordSegment | WordNinja |
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

DKSplit treats segmentation as a character-level sequence labeling task. The
training data includes LLM-labeled domain segmentations, brand names, personal
name combinations, multilingual phrases (English, French, German, Spanish, and
more), and tech product names. At inference, the BiLSTM runs as an
INT8-quantized ONNX model and CRF decoding is performed in NumPy. No GPU
required; around 800 samples per second on a single CPU thread.

**Why BiLSTM-CRF:** dictionary-based segmenters (WordSegment, WordNinja) work
on standard English but break down on newly registered domains: mostly brand
coinages, multilingual compounds, and intentional misspellings. Subword
transformers (e.g. DeBERTa) operate at a granularity coarser than this task
needs, so a single subword token can span a real word boundary and lose the
signal. Large language models can be accurate but cost orders of magnitude more
per string, which doesn't work at millions of inputs per day. Character-level
BiLSTM-CRF hits the practical optimum: character precision, CPU-only inference,
a 9 MB artifact. For a head-to-head failure-mode comparison with DeBERTa-V3 and
the dictionary baselines, see the
[DKSplit Update blog post](https://abtdomain.com/blog/2026/04/dksplit-update-cleaner-benchmark-first-deberta-run-different-failure-modes/).

**LLM variant:** we have also fine-tuned a Qwen LoRA on the same task and
published the checkpoint at
[ABTdomain/dksplit-qwen-lora](https://huggingface.co/ABTdomain/dksplit-qwen-lora).
It is useful for research, evaluation, and offline batch jobs where you want a
generative model's behavior on edge cases.

## Features

- **Brand-aware:** recognizes thousands of brands, tech products, and proper nouns
- **Multilingual:** English, French, German, Spanish, and romanized text
- **Lightweight:** 9 MB model, minimal dependencies (numpy + onnxruntime)
- **Offline:** no API keys, no internet required
- **Top-k candidates:** `split3` / `split5` / `split_topk` return ranked
  alternative segmentations

## Limitations

- **Characters:** only `a-z` and `0-9`. Input is automatically lowercased.
  Any other character (hyphen, dot, underscore) is mapped to an unknown token
  and kept in the output rather than treated as a separator:
  `split("your-brandlogin")` returns `['your-', 'brand', 'login']`. For real
  domains and URLs, split on `-`, `.`, and `/` first and feed each
  letters-and-digits run separately. In general, boundaries between letters
  and non-letters should be split with conventional rules first, not handed
  to the model.
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
Playground Access program (project AIFAC_P02_281). We thank EuroHPC JU for
enabling SMEs to explore new possibilities with world-class HPC infrastructure.