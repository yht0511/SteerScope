# SteerScope

Official code for the anonymous ICLR 2027 submission **“Does Steering Break Your Model? A Multi-Dimensional Evaluation Suite for LLM Steering Methods.”**

SteerScope is an evaluation suite for language-model steering. It measures target efficacy, behavioral side effects, generalization, and dependence on training data under a shared experimental protocol. This repository contains the method implementations, evaluators, formal experiment configurations, scheduler, and analysis notebook used by the paper.

> This repository is anonymized for double-blind review. Author and paper links will be added after the review period.

## Setup

The experiments require Linux, NVIDIA GPUs, Git, and access to the gated Gemma 2 weights on Hugging Face. Jailbreak Safety evaluation also requires access to `meta-llama/Llama-3.1-8B-Instruct`. SteerScope uses Python 3.12; the pinned EasySteer runtime is installed in a separate Python 3.10 environment.

```bash
curl -fL \
  https://anonymous.4open.science/api/repo/SteerScope-321B/zip \
  -o SteerScope.zip
unzip SteerScope.zip -d SteerScope
rm SteerScope.zip
cd SteerScope
chmod +x ./scripts/*
bash scripts/setup.sh
```

The setup script:

1. creates `.venv` from the committed `uv.lock` and downloads the NLTK `punkt` and `punkt_tab` resources required by IFEval;
2. fetches the pinned EasySteer version and its vLLM submodule (also supported when installing from ZIP);
3. creates `.venv-easysteer` and installs EasySteer;
4. downloads the evaluator datasets and released concept files.

EasySteer dependencies are pinned in `easysteer-requirements.txt` for Python 3.10, PyTorch 2.10.0 and CUDA 12.8. Use a compatible NVIDIA driver. The setup script checks that EasySteer and vLLM import successfully before downloading datasets.

Copy the environment template and set the required credentials:

```bash
cp .env.example .env
nano .env
set -a
source .env
set +a
```

| Variable | Purpose |
|---|---|
| `STEERSCOPE_GENERATION_API_KEY` | DeepSeek-V3.2 prompt and data generation |
| `STEERSCOPE_GENERATION_BASE_URL` | OpenAI-compatible generation endpoint |
| `STEERSCOPE_GENERATION_MODEL` | Optional API model override for data and steering-prompt generation |
| `STEERSCOPE_JUDGE_API_KEY` | GPT-4o-mini online LM-judge evaluation |
| `STEERSCOPE_JUDGE_BASE_URL` | OpenAI-compatible judge endpoint |
| `STEERSCOPE_JUDGE_MODEL` | Optional API model override for LM judging |
| `HF_TOKEN` | Access to gated model weights |
| `HF_HOME` | Optional Hugging Face cache directory |
| `STEERSCOPE_CACHE_DIR` | Optional SteerScope cache directory |

Nonempty model environment variables override the YAML model names; leaving them
empty preserves the paper defaults. Set the exact model IDs accepted by your API
provider, then reload `.env` before launching. 

Downloaded datasets and model weights remain subject to their upstream licenses and terms. See [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

## Reproducing the experiments

Run any of the four released experiment profiles:

```bash
./scripts/run_2b_l10.sh
./scripts/run_2b_l20.sh
./scripts/run_9b_l20.sh
./scripts/run_9b_l31.sh
```

We ran each of the four experiment profiles on eight NVIDIA A100 GPUs (80 GB each).
Results are saved to `outputs/paper/<model>/<layer>/`.


For a smaller 10-concept run with Gemma-2-2B-it at layer 20:

```bash
./scripts/run_2b_l20_10concepts.sh
```

This runs the same complete workflow for 21 methods, excluding HyperSteer, FLAS,
and SFT. Its configurations are in `steerscope/sweep/paper/2b/l20_10concepts/`, with a
separate scheduler config at `steerscope/sweep/paper/scheduler_configs/2b_l20_10concepts.yaml`.
Results go to `outputs/paper/2b/l20_10concepts/`. This profile uses two GPUs.

GPU assignments and resource settings can be adjusted in `steerscope/sweep/paper/scheduler_configs/`

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

The first cell defines the result roots and figure destination.Tables are displayed in the notebook, and figures are saved under:

```text
steerscope/sweep/paper/paper_figures/
```

The analysis selects one global factor for each method and uses that fixed factor in all downstream best-factor comparisons.


## Repository layout

```text
SteerScope/
├── scripts/
│   ├── setup.sh
│   ├── run_2b_l10.sh
│   ├── run_2b_l20.sh
│   ├── run_2b_l20_10concepts.sh
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
