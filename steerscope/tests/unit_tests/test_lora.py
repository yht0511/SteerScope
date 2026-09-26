from unittest.mock import MagicMock

import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel

from steerscope.models.lora import LoRA


def test_lora_loads_while_distributed_is_initialized(tmp_path):
    base_model = GPT2LMHeadModel(GPT2Config(
        n_layer=1,
        n_head=1,
        n_embd=8,
        n_positions=16,
        vocab_size=32,
    ))
    adapter_model = get_peft_model(base_model, LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=["c_attn"],
        use_rslora=True,
        task_type="CAUSAL_LM",
    ))
    adapter_dir = tmp_path / "lora" / "0"
    adapter_model.save_pretrained(adapter_dir)
    base_model = adapter_model.unload()

    dist.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'rendezvous'}",
        rank=0,
        world_size=1,
    )
    try:
        model = LoRA(
            base_model,
            tokenizer=MagicMock(),
            layer=0,
            training_args=MagicMock(),
            device="cpu",
        )
        model.load(tmp_path, concept_id=0)
    finally:
        dist.destroy_process_group()

    assert model.ax_model.peft_config["default"].r == 2
