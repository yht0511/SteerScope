import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml

from steerscope.scripts.args.training_args import TrainingArgs


class TestTrainingArgsModelScope(unittest.TestCase):
    def parse(self, config, *extra_args, ignore_unknown=False):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            argv = ["train.py", "--config", str(config_path), *extra_args]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                return TrainingArgs(section="train", ignore_unknown=ignore_unknown)

    def test_rank_exists_only_on_explicitly_configured_model(self):
        args = self.parse({
            "train": {
                "batch_size": 8,
                "models": {
                    "DiffMean": {},
                    "LoRA": {"low_rank_dimension": 4},
                },
            }
        })

        self.assertFalse(hasattr(args, "low_rank_dimension"))
        self.assertFalse(hasattr(args.models["DiffMean"], "low_rank_dimension"))
        self.assertEqual(args.models["LoRA"].low_rank_dimension, 4)
        self.assertEqual(args.models["DiffMean"].batch_size, 8)
        self.assertEqual(args.models["LoRA"].batch_size, 8)

    def test_rank_can_be_set_with_model_param(self):
        args = self.parse(
            {"train": {"models": {"LoRA": {}}}},
            "--model_param",
            "LoRA.low_rank_dimension=6",
        )

        self.assertEqual(args.models["LoRA"].low_rank_dimension, 6)

    def test_rank_is_rejected_at_train_section_level(self):
        with self.assertRaisesRegex(ValueError, "Model-only parameter"):
            self.parse({
                "train": {
                    "low_rank_dimension": 3,
                    "models": {"LoRA": {}},
                }
            })

    def test_rank_is_rejected_as_global_cli_option(self):
        with self.assertRaisesRegex(ValueError, "Model-only option"):
            self.parse(
                {"train": {"models": {"LoRA": {}}}},
                "--low_rank_dimension",
                "3",
                ignore_unknown=True,
            )


if __name__ == "__main__":
    unittest.main()
