# Third-party software and data

SteerScope is released under the Apache License 2.0. This document records
material third-party sources used by the repository and by its reproduction
workflow. It is an attribution summary, not a replacement for the original
license texts or usage terms.

## Code included in or adapted by this repository

| Component | Use in SteerScope | Upstream | License |
|---|---|---|---|
| AxBench | Foundation from which the benchmark was derived | [stanfordnlp/axbench](https://github.com/stanfordnlp/axbench) | Apache-2.0 |
| Google Instruction Following Evaluation | Vendored instruction definitions used by the IFEval evaluator | [google-research/google-research](https://github.com/google-research/google-research/tree/95e3a1da2d27cb9c8289f6fd3076cfed608c3c94/instruction_following_eval) at `95e3a1da2d27cb9c8289f6fd3076cfed608c3c94` | Apache-2.0; preserved in `steerscope/evaluators/ifeval_official/LICENSE` |
| MATH evaluation utilities | Vendored answer normalization and extraction logic | [hendrycks/math](https://github.com/hendrycks/math) at `985bdc1696e88e8643f081a0ff4719da39f2ae2a` | MIT; preserved in `steerscope/evaluators/math_official/LICENSE` |
| EasySteer | Optional high-throughput inference backend, included as a pinned Git submodule | [ZJU-REAL/EasySteer](https://github.com/ZJU-REAL/EasySteer) at `fff1f61837e34af0f70351d46756187a69530f7e` | Apache-2.0 |
| FLAS | The `FLAS` adapter and `flas_core` implementation follow the released method implementation | [flas-ai/FLAS](https://github.com/flas-ai/FLAS) at `720ef8a67697d9b94130b374b5b3a1522a782566` | Apache-2.0 |
| ODESteer | The `ODESteer` and `StepODESteer` numerical implementation follows the authors' public implementation | [ZhaoHongjue/odesteer](https://github.com/ZhaoHongjue/odesteer) at `8a3c481d6493ecb3325eea5ef9c448cccfced7eb` | See the upstream repository for current terms |

Python packages installed from `pyproject.toml` and `uv.lock` remain subject
to their respective licenses. The lockfile records the resolved package names,
versions, and sources.

## Assets downloaded during setup

The repository does not redistribute the following model weights or benchmark
datasets. `scripts/setup.sh` downloads them from their original hosts. Each
asset remains governed by its upstream license, data card, access conditions,
and terms of use.

| Asset | Upstream source |
|---|---|
| Gemma 2 model weights | [Google Gemma on Hugging Face](https://huggingface.co/google) |
| GemmaScope explanations | [Neuronpedia exports](https://www.neuronpedia.org/) |
| AlpacaEval instructions | [tatsu-lab/alpaca_eval](https://huggingface.co/datasets/tatsu-lab/alpaca_eval) |
| Feature descriptions | [yoavgur/Feature-Descriptions](https://github.com/yoavgur/Feature-Descriptions) |
| MMLU | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) |
| SuperGLUE | [aps/super_glue](https://huggingface.co/datasets/aps/super_glue) |
| MATH | [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math) |
| IFEval | [Google Research instruction_following_eval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) |
| TruthfulQA | [sylinrl/TruthfulQA](https://github.com/sylinrl/TruthfulQA) |
| BBQ | [nyu-mll/BBQ](https://github.com/nyu-mll/BBQ) |
| JailbreakBench behaviors | [JailbreakBench/JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) |
| X-AlpacaEval | [zhihz0535/X-AlpacaEval](https://huggingface.co/datasets/zhihz0535/X-AlpacaEval) |

Before redistributing any downloaded asset or generated derivative, consult
the exact revision recorded by its generated manifest and the corresponding
upstream terms.
