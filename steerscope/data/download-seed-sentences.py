from pathlib import Path

from datasets import load_dataset, DatasetDict, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent

# text data
wikisum_ds = load_dataset("zhengxuanzenwu/wikitext-2-split-128", revision="817f68fefc4d740360dded88d91f53089f21c10d")
text_train = []
for example in wikisum_ds["train"]:
    text_train += [example["text"]]

# math data
gsm_ds = load_dataset("openai/gsm8k", "main", revision="740312add88f781978c0658806c59bc2815b9866")
math_train = []
for example in gsm_ds["train"]:
    math_train += [example["answer"]]

# code data
code_ds = load_dataset("christopher/rosetta-code", revision="11d8b38cbd90b0ae5bde304ae2d4ba5ace10af05")
code_all = []
for example in code_ds["train"]:
    if len(example["code"]) > 500:
        code_all += [example["code"][:500]]
code_train = code_all

data = {
    "text_train": text_train[:1000],
    "math_train": math_train[:1000],
    "code_train": code_train[:1000],
}
dataset = DatasetDict({
    "text_train": Dataset.from_dict({"input": data["text_train"]}),
    "math_train": Dataset.from_dict({"input": data["math_train"]}),
    "code_train": Dataset.from_dict({"input": data["code_train"]}),
})

dataset.save_to_disk(SCRIPT_DIR / "seed_sentences")


# text instructions
dolly_ds = load_dataset("databricks/databricks-dolly-15k", revision="bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a")
text_train = []
for example in dolly_ds["train"]:
    if example["category"] == "open_qa" and example["context"] == "":
        if len(example["instruction"]) < 500:
            text_train += [example["instruction"]]

# math instructions
gsm_ds = load_dataset("openai/gsm8k", "main", revision="740312add88f781978c0658806c59bc2815b9866")
math_train = []
for example in gsm_ds["train"]:
    if len(example["question"]) < 500:
        math_train += [example["question"]]

# code instructions
alpaca_ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", revision="7cae181e29701a8663a07a3ea43c8e105b663ba1")
code_train = []
for example in alpaca_ds["train"]:
    if example["input"] == "":
        if len(example["instruction"]) < 500:
            code_train += [example["instruction"]]

data = {
    "text_train": text_train[:1000],
    "math_train": math_train[:1000],
    "code_train": code_train[:1000],
}
dataset = DatasetDict({
    "text_train": Dataset.from_dict({"input": data["text_train"]}),
    "math_train": Dataset.from_dict({"input": data["math_train"]}),
    "code_train": Dataset.from_dict({"input": data["code_train"]}),
})

dataset.save_to_disk(SCRIPT_DIR / "seed_instructions")
