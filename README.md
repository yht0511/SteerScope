# SteerScope

Official code for the anonymous ICLR 2027 submission **“Does Steering Break Your Model? A Multi-Dimensional Evaluation Suite for LLM Steering Methods.”**

SteerScope is an evaluation suite for language-model steering. It measures target efficacy, behavioral side effects, generalization, and dependence on training data under a shared experimental protocol. This repository contains the method implementations, evaluators, formal experiment configurations, scheduler, and analysis notebook used by the paper.

> This repository is anonymized for double-blind review. Author and paper links will be added after the review period.

## Setup

The experiments require Linux, NVIDIA GPUs, Git, and access to the gated Gemma 2 weights on Hugging Face. Jailbreak Safety evaluation also requires access to `meta-llama/Llama-3.1-8B-Instruct`. SteerScope uses Python 3.12; the pinned EasySteer runtime is installed in a separate Python 3.10 environment.

```bash
git clone --recurse-submodules <anonymous-repository-url> SteerScope
cd SteerScope
./scripts/setup.sh
```

The setup script:

1. creates `.venv` from the committed `uv.lock`;
2. initializes the pinned EasySteer submodule;
3. creates `.venv-easysteer` and installs EasySteer;
4. downloads the evaluator datasets and released concept files.

EasySteer dependencies are pinned in `easysteer-requirements.txt` for Python 3.10, PyTorch 2.10.0 and CUDA 12.8. Use a compatible NVIDIA driver. The setup script checks that EasySteer and vLLM import successfully before downloading datasets.

Copy the environment template and set the required credentials:

```bash
cp .env.example .env
set -a
source .env
set +a
```

| Variable | Purpose |
|---|---|
| `STEERSCOPE_GENERATION_API_KEY` | DeepSeek-V3.2 prompt and data generation |
| `STEERSCOPE_GENERATION_BASE_URL` | OpenAI-compatible generation endpoint |
| `STEERSCOPE_JUDGE_API_KEY` | GPT-4o-mini online LM-judge evaluation |
| `STEERSCOPE_JUDGE_BASE_URL` | OpenAI-compatible judge endpoint |
| `HF_TOKEN` | Access to gated model weights |
| `HF_HOME` | Optional Hugging Face cache directory |
| `STEERSCOPE_CACHE_DIR` | Optional SteerScope cache directory |

Downloaded datasets and model weights remain subject to their upstream licenses and terms. See [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

## Reproducing the experiments

Run any of the four released experiment profiles:

```bash
./scripts/run_2b_l10.sh
./scripts/run_2b_l20.sh
./scripts/run_9b_l20.sh
./scripts/run_9b_l31.sh
```

Each command runs the complete resumable workflow: data generation, method training, factor-sweep evaluation, online LM judging, generalization, sample efficiency, and sample sensitivity. Completed LM requests and evaluator results are cached, so rerunning the same command resumes the experiment.

Results are written to:

```text
outputs/paper/<model>/<layer>/
```

The released scheduler profiles assume an eight-GPU machine. The 9B profile reserves up to 70 GiB per worker plus an 8 GiB safety margin; its defaults target 80 GB GPUs. SFT uses two GPUs per training job. GPU assignments, memory reservations, EasySteer parallelism, and API concurrency can be adjusted in:

```text
steerscope/sweep/paper/scheduler_configs/
```

Inspect a run without launching work:

```bash
./scripts/run_2b_l20.sh --dry-run --no-wandb
./scripts/run_9b_l20.sh --dry-run --no-wandb
```

Scheduler options may follow the script name. For example:

```bash
./scripts/run_2b_l20.sh --methods DiffMean PCA LAT
```

The sweep generates its training data through the configured generation API. Re-querying a hosted generation model or judge does not guarantee identical paper numbers, even at temperature zero.

Dataset downloaders pin source revisions; the four Neuronpedia files are checked against recorded hashes. `source_revisions.json` records release download versions, not historical paper data versions. Retain generated data and evaluator outputs with each experiment.

Generation and judging use separate API credentials. Cache keys distinguish model, endpoint, temperature, and prompt. Older API-cache entries that lack these identities are not reused. Transient request failures are retried. Judge responses that still cannot be parsed after the configured retries receive **0**, with fallback counts in evaluator metadata and details in `judge_parse_failures.jsonl`.

## Experiment configurations

The repository contains four complete profiles:

| Backbone | Layers |
|---|---|
| `google/gemma-2-2b-it` | 10, 20 |
| `google/gemma-2-9b-it` | 20, 31 |

Method, generalization, and data-dependence YAML files are generated from:

```text
steerscope/sweep/paper/generate_configs.py
```

Regenerate and validate them with:

```bash
.venv/bin/python steerscope/sweep/paper/generate_configs.py
.venv/bin/python steerscope/sweep/paper/validate_configs.py
```

Do not edit generated experiment YAML files by hand. Resource settings belong in the scheduler profiles.

## Analysis

After the 2B/l20 and 9B/l20 experiments finish, run the analysis notebook from top to bottom:

```bash
.venv/bin/jupyter lab steerscope/sweep/paper/scheduler_metrics_analysis.ipynb
```

The first cell defines the result roots and figure destination. Tables and publication figures are written under:

```text
steerscope/sweep/paper/paper_figures/
```

The analysis selects one global factor for each method and uses that fixed factor in all downstream best-factor comparisons. It does not select a separate factor for each concept.

Sample efficiency uses the same 10 evaluation prompts per concept at every budget, including the full-data reference. Sample sensitivity uses 20 prompts with greedy decoding and all five subset seeds (42–46). Sensitivity is the mean of the per-concept seed SDs; incomplete seed panels are rejected. Scheduler summaries and the notebook share this definition. Generalization tables distinguish signed improvement (`steered − baseline`) from Instruction/Fluency degradation (`baseline − steered`).

Run the CPU tests and configuration checks with:

```bash
.venv/bin/python -m pytest steerscope/tests steerscope/sweep/paper/analysis/test_metrics_tables.py
.venv/bin/python steerscope/sweep/paper/validate_configs.py
```

Optional artifact tests read `outputs/paper` or `STEERSCOPE_RELEASE_ROOT`. Passing CPU tests or a scheduler dry-run does not test GPU memory requirements, model access, or hosted API availability.

## Repository layout

```text
SteerScope/
├── scripts/
│   ├── setup.sh
│   ├── run_2b_l10.sh
│   ├── run_2b_l20.sh
│   ├── run_9b_l20.sh
│   └── run_9b_l31.sh
├── steerscope/
│   ├── models/             # steering methods
│   ├── evaluators/         # evaluation metrics
│   ├── evaluation/         # evaluator graph and result store
│   ├── inference/          # inference backends
│   ├── data/               # dataset downloaders
│   ├── studies/            # data-dependence utilities
│   └── sweep/paper/        # formal configs, scheduler, and analysis
├── reference/EasySteer/    # pinned submodule
├── pyproject.toml
└── uv.lock
```

See [`steerscope/README.md`](steerscope/README.md) for instructions on adding methods and evaluators.

## License

SteerScope is released under the Apache License 2.0. See [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).
