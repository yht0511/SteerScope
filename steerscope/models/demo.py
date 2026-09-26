"""Human-readable model used to inspect the evaluation pipeline."""

import sys

from .model import Model


class DemoModel(Model):
    """Return deterministic or manually entered text without loading an LLM."""

    requires_training_args = False
    load_trained_weights = False
    uses_intervention_positions = False
    requires_mean_activations = False
    lightweight_runtime = True

    def __str__(self):
        return "DemoModel"

    def make_model(self, **kwargs):
        self._print("model setup", "no weights are needed")

    def load(self, dump_dir=None, **kwargs):
        self._print("load", "no artifact is needed")

    def save(self, dump_dir, **kwargs):
        self._print("save", "nothing to save")

    def train(self, examples, **kwargs):
        self._print("train", f"received {len(examples)} examples; nothing to train")

    def to(self, device):
        self.device = device
        self._print("device", str(device))
        return self

    def predict_steer(self, examples, **kwargs):
        interactive = bool(kwargs.get("demo_interactive", False))
        default = str(kwargs.get("demo_output", "Demo response"))
        if interactive and not sys.stdin.isatty():
            raise RuntimeError(
                "DemoModel interactive mode needs a terminal. Run evaluate.py "
                "directly, or set inference.demo_interactive: false."
            )

        outputs = []
        total = len(examples)
        self._print(
            "inference",
            f"{total} prompt(s), mode={'manual' if interactive else 'automatic'}",
        )
        for number, (_, row) in enumerate(examples.iterrows(), start=1):
            prompt = str(row.get("input", ""))
            print(f"\n[DemoModel] prompt {number}/{total}")
            print("-" * 72)
            print(prompt)
            print("-" * 72)
            if interactive:
                output = input("[DemoModel] model output> ")
            else:
                output = default.format(
                    prompt=prompt,
                    concept=row.get("input_concept", ""),
                    factor=row.get("factor", ""),
                    input_id=row.get("input_id", ""),
                )
                print(f"[DemoModel] model output: {output}")
            outputs.append(output)
        self._print("inference complete", f"returned {len(outputs)} output(s)")
        return {"steered_generation": outputs}

    @staticmethod
    def _print(step, detail):
        print(f"[DemoModel] {step}: {detail}")
