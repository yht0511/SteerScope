# Extending SteerScope

SteerScope separates method training, inference, evaluator graphs, and paper orchestration.

## Add a steering method

1. Implement the method under `steerscope/models/`.
2. Export its class from `steerscope/__init__.py`.
3. Add its training recipe and factor grid to `steerscope/sweep/paper/generate_configs.py`.
4. Regenerate and validate the four profiles.
5. Add focused unit tests for training artifacts and inference routing.

## Add an evaluator

Implement an evaluator under `steerscope/evaluators/` and declare its dataset, inference settings, dependencies, and report settings in the generated YAML. Evaluators own data construction, inference mode, scoring, cache context, and output schemas; the evaluation engine executes their dependency graph.

## Regenerate paper configurations

```bash
.venv/bin/python steerscope/sweep/paper/generate_configs.py
.venv/bin/python steerscope/sweep/paper/validate_configs.py
```

Do not edit generated method, generalization, or study YAMLs by hand. Scheduler resource settings live in `steerscope/sweep/paper/scheduler_configs/`.
