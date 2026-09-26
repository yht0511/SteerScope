from .model import Model
from pyvene import (
    IntervenableConfig,
    IntervenableModel
)

import fcntl
import math
import os, requests, torch
import json
import tempfile
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from .interventions import (
    JumpReLUSAECollectIntervention,
    AdditionIntervention,
    SubspaceIntervention,
    DictionaryAdditionIntervention, # please try this one
    DictionaryMinClampingIntervention,
    SigmoidMaskAdditionIntervention,
)
from ..utils.model_utils import (
    remove_gradient_parallel_to_decoder_directions,
    gather_residual_activations,
    get_lr,
    calculate_l1_losses
)
from huggingface_hub import hf_hub_download
from transformers import get_scheduler

from torch.utils.data import DataLoader
from .probe import DataCollator, make_data_module

from sklearn.metrics import roc_auc_score

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)

# using pyreft out-of-the-box
import pyreft


def _neuronpedia_max_activation(feature_ref, master_data_dir="steerscope/data"):
    """Resolve and cache the historical Neuronpedia feature scale."""
    parts = str(feature_ref).rstrip("/").split("/")
    if len(parts) < 3:
        raise ValueError(
            f"Cannot parse Neuronpedia feature reference: {feature_ref!r}."
        )
    model_name, sae_name = parts[-3], parts[-2]
    try:
        feature_id = int(parts[-1])
    except ValueError as error:
        raise ValueError(
            f"Cannot parse Neuronpedia feature ID from {feature_ref!r}."
        ) from error

    cache_dir = Path(master_data_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{model_name}_{sae_name}_max_activations.json"
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        cached = {}
        if cache_path.exists():
            try:
                with cache_path.open(encoding="utf-8") as cache_file:
                    cached = json.load(cache_file)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid Neuronpedia max-activation cache: {cache_path}."
                ) from error
        if str(feature_id) in cached:
            cached_value = float(cached[str(feature_id)])
            return (
                cached_value
                if math.isfinite(cached_value) and cached_value > 0
                else 50.0
            )

        sae_path = str(feature_ref).split("https://www.neuronpedia.org/")[-1]
        url = f"https://www.neuronpedia.org/api/feature/{sae_path}"
        api_key = os.environ.get("NP_API_KEY")
        headers = {"X-Api-Key": api_key} if api_key else {}
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        try:
            max_activation = float(response.json()["activations"][0]["maxValue"])
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"Neuronpedia returned no max activation for {feature_ref}."
            ) from error
        if not math.isfinite(max_activation) or max_activation <= 0:
            max_activation = 50.0
        cached[str(feature_id)] = max_activation

        with tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            dir=cache_dir,
            suffix=".json.tmp",
            encoding="utf-8",
        ) as temporary:
            json.dump(cached, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        try:
            os.chmod(temporary_path, 0o644)
            os.replace(temporary_path, cache_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return max_activation


def load_metadata_flatten(metadata_path):
    """
    Load flatten metadata from a JSON lines file.
    """
    metadata = []
    concept_id = 0
    with open(metadata_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            concept, ref = data["concept"], data["ref"]
            concept_genres_map = data["concept_genres_map"][concept]
            ref = data["ref"]
            flatten_data = {
                "concept": concept,
                "ref": ref,
                "concept_genres_map": {concept: concept_genres_map},
                "concept_id": concept_id
            }
            metadata += [flatten_data]  # Return the metadata as is
            concept_id += 1
    return metadata


def save_pruned_sae(
    metadata_path, dump_dir, modified_refs=None, savefile="GemmaScopeSAE.pt",
    sae_params=None, concept_ids=None
):
    # Save SAE weights and biases for inference
    logger.warning("Saving SAE weights and biases for inference")
    flatten_metadata = load_metadata_flatten(metadata_path)
    if concept_ids is not None:
        flatten_metadata = [flatten_metadata[int(i)] for i in concept_ids]
    # logger.warning(flatten_metadata)

    # mess with SAE refs if needed (for DBM stuff)
    if modified_refs is not None:
        for i, ref in enumerate(modified_refs):
            flatten_metadata[i]["ref"] = "/".join(flatten_metadata[i]["ref"].split("/")[:-1]) + "/" + str(ref)

    # Save pruned SAE weights and biases
    # https://www.neuronpedia.org/llama3.1-8b/20-llamascope-res-131k/65591
    # https://www.neuronpedia.org/api/feature/llama3.1-8b/20-llamascope-res-131k/65591
    sae_path = flatten_metadata[0]["ref"].split("https://www.neuronpedia.org/")[-1]
    sae_url = f"https://www.neuronpedia.org/api/feature/{sae_path}"
    headers = {"X-Api-Key": os.environ.get("NP_API_KEY")}
    while True:
        try:
            response = requests.get(sae_url, headers=headers).json()
            break
        except Exception as e:
            logger.warning(f"Error fetching SAE from Neuronpedia: {e}. Retrying...")
            continue
    hf_repo = response["source"]["hfRepoId"]
    hf_folder = response["source"]["hfFolderId"]
    hf_filename = f"{hf_folder}/params.npz"
    if hf_repo is None:
        raise ValueError(f"Neuronpedia did not provide an SAE repository for {sae_url}.")
    if sae_params is None:
        logger.warning(f"Loading SAE params from {hf_repo}/{hf_filename}")
        path_to_params = hf_hub_download(
            repo_id=hf_repo,
            filename=hf_filename,
            force_download=False,
        )
        sae_params = np.load(path_to_params)
        logger.warning(f"Loaded SAE params from {path_to_params}")
        logger.warning(f"SAE params: {list(sae_params.keys())}")
    sae_pt_params = {k: torch.from_numpy(v) for k, v in sae_params.items()}
    pruned_sae_pt_params = {
        "b_dec": sae_pt_params["b_dec"],
        "W_dec": [],
        "W_enc": [],
        "b_enc": [],
        "threshold": []
    }
    for concept_id, m in enumerate(flatten_metadata):
        sae_id = int(m["ref"].split("/")[-1])
        pruned_sae_pt_params["W_dec"].append(sae_pt_params["W_dec"][[sae_id], :])
        pruned_sae_pt_params["W_enc"].append(sae_pt_params["W_enc"][:, [sae_id]])
        pruned_sae_pt_params["b_enc"].append(sae_pt_params["b_enc"][[sae_id]])
        pruned_sae_pt_params["threshold"].append(sae_pt_params["threshold"][[sae_id]])
    for k, v in pruned_sae_pt_params.items():
        if k == "b_dec":
            continue
        if k == "W_enc":
            pruned_sae_pt_params[k] = torch.cat(v, dim=1)
        else:
            pruned_sae_pt_params[k] = torch.cat(v, dim=0)
    if savefile is not None:
        torch.save(pruned_sae_pt_params, dump_dir / savefile) # sae only has one file
    return sae_params


class GemmaScopeSAE(Model):
    requires_training_args = False
    requires_calibration_scale = True

    @classmethod
    def training_fingerprint_context(cls):
        return {"calibration": "neuronpedia_selected_feature_max_v1"}

    @classmethod
    def prepare_calibration_artifact(
        cls,
        metadata_path,
        dump_dir,
        concept_ids,
        master_data_dir="steerscope/data",
        model_name="GemmaScopeSAE",
    ):
        """Build the concept-indexed SAE scale artifact during training."""
        concept_ids = sorted({int(concept_id) for concept_id in concept_ids})
        if not concept_ids:
            raise ValueError("SAE calibration requires at least one concept ID.")

        metadata = load_metadata_flatten(metadata_path)
        if concept_ids[-1] >= len(metadata):
            raise IndexError(
                f"SAE concept ID {concept_ids[-1]} is not present in "
                f"{metadata_path}."
            )

        dump_dir = Path(dump_dir)
        scale_path = dump_dir / f"{model_name}_scale.pt"
        # Inference indexes scales with the original global concept ID.
        scales = torch.full(
            (concept_ids[-1] + 1,), torch.nan, dtype=torch.float32
        )
        if scale_path.exists():
            existing = torch.load(
                scale_path, map_location="cpu", weights_only=True
            ).reshape(-1)
            # A valid globally indexed artifact covers the maximum requested
            # ID. Do not mistake an older compact sampled artifact for one.
            if existing.numel() > concept_ids[-1]:
                scales.copy_(existing[: scales.numel()])

        for concept_id in concept_ids:
            if torch.isfinite(scales[concept_id]) and scales[concept_id] > 0:
                continue
            scales[concept_id] = _neuronpedia_max_activation(
                metadata[concept_id]["ref"], master_data_dir
            )

        # Shared loaders validate all rows. Unselected rows are unreachable
        # because evaluation checks the manifest's concept-ID set first.
        scales[~torch.isfinite(scales)] = 1.0

        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=dump_dir,
            suffix=".pt.tmp",
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            torch.save(scales, temporary_path)
            os.chmod(temporary_path, 0o644)
            os.replace(temporary_path, scale_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        logger.warning(
            "Saved %s training-time SAE max activations to %s.",
            len(scales),
            scale_path,
        )
        return scale_path

    def __str__(self):
        return 'GemmaScopeSAE'

    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "latent")
        intervention_type = kwargs.get("intervention_type", "addition")
        if mode == "steering":
            if intervention_type == "addition":
                ax = AdditionIntervention(
                    embed_dim=self.model.config.hidden_size,
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            elif intervention_type == "clamping":
                ax = DictionaryAdditionIntervention(
                    embed_dim=self.model.config.hidden_size,
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            elif intervention_type == "min_clamping":
                ax = DictionaryMinClampingIntervention(
                    embed_dim=self.model.config.hidden_size,
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            else:
                raise ValueError(f"Invalid intervention type for steering: {intervention_type}")
        else:
            ax = JumpReLUSAECollectIntervention(
                embed_dim=self.model.config.hidden_size,
                low_rank_dimension=kwargs.get("low_rank_dimension", 1),
            )
        ax = ax.train()
        ax_config = IntervenableConfig(representations=[{
            "layer": l,
            "component": f"model.layers[{l}].output",
            "low_rank_dimension": kwargs.get("low_rank_dimension", 1),
            "intervention": ax} for l in [self.layer]])
        ax_model = IntervenableModel(ax_config, self.model)
        ax_model.set_device(self.device)
        self.ax = ax
        self.ax_model = ax_model

    def load(self, dump_dir=None, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        logger.warning(f"Loading SAE from {dump_dir}/{model_name}.pt")
        params = torch.load(
            f"{dump_dir}/{model_name}.pt", map_location="cpu", weights_only=True
        )
        pt_params = {k: v.to(self.device) for k, v in params.items()}
        kwargs["low_rank_dimension"] = params['W_enc'].shape[1]
        self.make_model(**kwargs)
        if isinstance(self.ax, SubspaceIntervention) or isinstance(self.ax, AdditionIntervention):
            self.ax.proj.weight.data = pt_params['W_dec']
        else:
            try:
                self.ax.load_state_dict(pt_params, strict=True)
            except Exception as e:
                # let it passing
                logger.warning(f"Error loading state dict: {e}")
                self.ax.load_state_dict(pt_params, strict=False)

        scale_file = Path(dump_dir) / f"{model_name}_scale.pt"
        if not scale_file.exists():
            raise FileNotFoundError(
                f"{self.__class__.__name__} requires training-time SAE max "
                f"activations, but {scale_file} does not exist. Retrain the "
                "method to create a complete checkpoint."
            )
        scales = torch.load(
            scale_file, map_location="cpu", weights_only=True
        ).reshape(-1)
        if not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError(
                f"SAE calibration scale {scale_file} must contain only finite, "
                "positive values."
            )
        if scales.numel() > params["W_dec"].shape[0]:
            raise ValueError(
                f"SAE calibration scale {scale_file} has {scales.numel()} entries, "
                f"but {model_name}.pt has only {params['W_dec'].shape[0]} features."
            )
        self.calibration_scales = scales

    def _selected_feature_id(self):
        if hasattr(self, "top_feature"):
            return int(self.top_feature)
        return None

    def calibrate(self, examples, **kwargs):
        metadata_path = kwargs.get("metadata_path")
        concept_id = kwargs.get("concept_id")
        if metadata_path is None or concept_id is None:
            raise ValueError(
                f"{self.__class__.__name__} calibration requires metadata_path "
                "and concept_id."
            )
        metadata = load_metadata_flatten(metadata_path)
        if int(concept_id) >= len(metadata):
            raise IndexError(
                f"SAE concept ID {concept_id} is not present in {metadata_path}."
            )
        feature_ref = metadata[int(concept_id)]["ref"]
        selected_feature_id = self._selected_feature_id()
        if selected_feature_id is not None:
            feature_ref = (
                "/".join(feature_ref.rstrip("/").split("/")[:-1])
                + f"/{selected_feature_id}"
            )
        max_activation = _neuronpedia_max_activation(
            feature_ref,
            kwargs.get("master_data_dir", "steerscope/data"),
        )
        self.calibration_scale = torch.tensor(
            [max_activation], dtype=torch.float32
        )
        logger.warning(
            "%s feature %s Neuronpedia max activation: %.4f",
            self.__class__.__name__,
            feature_ref.rsplit("/", 1)[-1],
            max_activation,
        )
        return self.calibration_scale

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        if kwargs.get("disable_neuronpedia_max_act", False):
            logger.warning(f"Using evaluation-dataset max activations instead of Neuronpedia.")
            # Use max/mean activation from the evaluation dataset.
            return super().pre_compute_mean_activations(dump_dir, **kwargs)

        if hasattr(self, "calibration_scales"):
            metadata = kwargs.get("metadata")
            if metadata is None:
                raise ValueError(
                    f"Metadata is required to load {self.__class__.__name__} "
                    "training-time max activations."
                )
            max_activations = {}
            for position, feature in enumerate(metadata):
                concept_id = int(feature.get("concept_id", position))
                if concept_id >= self.calibration_scales.numel():
                    continue
                sae_id = int(str(feature["ref"]).rstrip("/").split("/")[-1])
                max_activations[sae_id] = float(
                    self.calibration_scales[concept_id]
                )
            self.max_activations = max_activations
            return max_activations

        # The evaluator-owned pipeline already carries the same metadata that
        # the original latent parquet files contained.
        metadata = kwargs.get("metadata")
        if metadata is not None:
            sae_links = [feature["ref"] for feature in metadata]
        else:
            # Loop over all praqut files in dump_dir.
            sae_links = []
            for file in os.listdir(dump_dir):
                if file.endswith(".parquet") and file.startswith("latent_data"):
                    df = pd.read_parquet(os.path.join(dump_dir, file))
                    # sort by concept_id from small to large and enumerate through all concept_ids.
                    for sae_link in sorted(df["sae_link"].unique()):
                        sae_links += [sae_link]

        model_name, sae_name = sae_links[0].split("/")[-3], sae_links[0].split("/")[-2]
        max_activations = {} # sae_id to max_activation

        # Load existing max activations file and skip if exists.
        max_activations_file = os.path.join(
            kwargs.get("master_data_dir", "steerscope/data"),
            f"{model_name}_{sae_name}_max_activations.json")
        if os.path.exists(max_activations_file):
            with open(max_activations_file, "r") as f:
                max_activations = json.load(f)
            max_activations = {int(k): v for k, v in max_activations.items()}

        has_new = False
        for sae_link in tqdm(sae_links):
            sae_path = sae_link.split("https://www.neuronpedia.org/")[-1]
            sae_id = int(sae_link.split("/")[-1])
            if sae_id in max_activations:
                continue
            url = f"https://www.neuronpedia.org/api/feature/{sae_path}"
            headers = {"X-Api-Key": os.environ["NP_API_KEY"]}
            response = requests.get(url, headers=headers)
            max_activation = response.json()["activations"][0]["maxValue"]
            max_activations[sae_id] = max_activation if max_activation > 0 else 50
            has_new = True

        if has_new:
            with open(max_activations_file, "w") as f:
                json.dump(max_activations, f)

        self.max_activations = max_activations
        return max_activations


class GemmaScopeSAEMaxDiff(GemmaScopeSAE):
    """
    pick the SAE feature with the largest difference in activation between the two classes
    """
    requires_training_args = True

    def __str__(self):
        return 'GemmaScopeSAEMaxDiff'

    def make_model(self, **kwargs):
        if kwargs.get("mode", "latent") == "train":
            # load the entire SAE
            self.sae_params = kwargs.get("sae_params", None)
            metadata_path = kwargs.get("metadata_path", None)
            self.sae_width = self.sae_params['W_dec'].shape[0]
            self.metadata_path = metadata_path
            kwargs["low_rank_dimension"] = self.sae_width
            sae_pt_params = {k: torch.from_numpy(v) for k, v in self.sae_params.items()}
            super().make_model(**kwargs)
            self.ax.load_state_dict(sae_pt_params, strict=False)
            self.ax.eval()
            self.ax.to(self.device)
        else:
            super().make_model(**kwargs)

    def make_dataloader(self, examples, **kwargs):
        data_module = make_data_module(self.tokenizer, self.model, examples)
        train_dataloader = DataLoader(
            data_module["train_dataset"], shuffle=True, batch_size=self.training_args.batch_size,
            collate_fn=data_module["data_collator"])
        return train_dataloader

    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples)
        torch.cuda.empty_cache()
        prefix_length = kwargs.get("prefix_length", 1)

        sum_positive_acts = torch.zeros(self.sae_width).to(self.device)
        sum_negative_acts = torch.zeros(self.sae_width).to(self.device)
        positive_count = 0
        negative_count = 0

        for epoch in range(self.training_args.n_epochs):
            for step, batch in enumerate(train_dataloader):
                # prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                inputs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                }

                # get SAE latents
                act_in = gather_residual_activations(
                    self.model, self.layer, inputs)
                ax_acts_batch = self.ax(act_in[:, prefix_length:])  # no bos token
                seq_lens = inputs["attention_mask"].sum(dim=1) - prefix_length # no bos token

                # add avg latents for each sequence
                for i in range(len(batch)):
                    acts = ax_acts_batch[i, :seq_lens[i]]
                    label = batch["labels"][i]
                    avg_acts = acts.mean(dim=0)
                    if label == 1:
                        sum_positive_acts += avg_acts
                        positive_count += 1
                    else:
                        sum_negative_acts += avg_acts
                        negative_count += 1

        # get latent activations
        positive_acts = sum_positive_acts / positive_count
        negative_acts = sum_negative_acts / negative_count
        max_diff = (positive_acts - negative_acts).argmax().item()
        self.top_feature = max_diff

        # print top 10 features
        top = (positive_acts - negative_acts).topk(10)
        for i in range(10):
            logger.warning(f"Feature {top.indices[i].item()}: {top.values[i].item()}")

    def save(self, dump_dir, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        logger.warning(f"saving {model_name}")
        top_feature = self.top_feature
        top_features = []
        if os.path.exists(dump_dir / f"{model_name}_top_features.json"):
            with open(dump_dir / f"{model_name}_top_features.json", "r") as f:
                top_features = json.load(f)
        top_features += [top_feature]
        with open(dump_dir / f"{model_name}_top_features.json", "w") as f:
            json.dump(top_features, f)

        # save the pruned SAE
        save_pruned_sae(
            self.metadata_path, dump_dir, modified_refs=top_features,
            savefile=f"{model_name}.pt", sae_params=self.sae_params,
            concept_ids=(
                [kwargs["concept_id"]]
                if kwargs.get("concept_id") is not None
                else None
            ),
        )
        self._save_calibration_scale(dump_dir, model_name)

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        if (
            hasattr(self, "calibration_scales")
            and not kwargs.get("disable_neuronpedia_max_act", False)
        ):
            return super().pre_compute_mean_activations(dump_dir, **kwargs)
        # setup
        model_name = kwargs.get("model_name", self.__str__())
        metadata = kwargs.get("metadata", None)
        if metadata is None:
            assert False, f"Metadata is required for {model_name}"

        # get original sae features (keys of our dict)
        sae_links = []
        for feature in metadata:
            sae_links += [feature["ref"]]

        # load the selected top features (values of our dict)
        file = os.path.join(dump_dir, f"../train/{model_name}_top_features.json")
        with open(file, "r") as f:
            top_features = json.load(f)
        top_features = [int(f) for f in top_features]

        model_name, sae_name = sae_links[0].split("/")[-3], sae_links[0].split("/")[-2]
        max_activations = {} # sae_id to max_activation

        # load existing max activations file and skip if exists.
        max_activations_file = os.path.join(
            kwargs.get("master_data_dir", "steerscope/data"),
            f"{model_name}_{sae_name}_max_activations.json")
        if os.path.exists(max_activations_file):
            with open(max_activations_file, "r") as f:
                max_activations = json.load(f)
            max_activations = {int(k): v for k, v in max_activations.items()}

        has_new = False
        shuffled_max_activations = {}
        for i, sae_id in enumerate(tqdm(top_features)):
            orig_sae_path = sae_links[i].split("https://www.neuronpedia.org/")[-1]
            orig_sae_id = int(orig_sae_path.split("/")[-1])
            sae_id = int(sae_id)
            sae_path = "/".join(orig_sae_path.split("/")[:-1]) + "/" + str(sae_id)
            if sae_id not in max_activations:
                url = f"https://www.neuronpedia.org/api/feature/{sae_path}"
                headers = {"X-Api-Key": os.environ["NP_API_KEY"]}
                response = requests.get(url, headers=headers)
                max_activation = response.json()["activations"][0]["maxValue"]
                max_activations[sae_id] = max_activation if max_activation > 0 else 50
            shuffled_max_activations[orig_sae_id] = max_activations[sae_id]
            has_new = True

        if has_new:
            with open(max_activations_file, "w") as f:
                json.dump(max_activations, f)

        # logger.warning(f"Max activations: {shuffled_max_activations}")
        self.max_activations = shuffled_max_activations
        return max_activations


class GemmaScopeSAEMaxAUC(GemmaScopeSAEMaxDiff):
    """
    pick the SAE feature with the best ROC AUC score
    """
    def __str__(self):
        return 'GemmaScopeSAEMaxAUC'

    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples)
        torch.cuda.empty_cache()
        prefix_length = kwargs.get("prefix_length", 1)

        all_positive_acts = []
        all_negative_acts = []

        for epoch in range(self.training_args.n_epochs):
            for step, batch in enumerate(train_dataloader):
                # prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                inputs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                }

                # get SAE latents
                act_in = gather_residual_activations(
                    self.model, self.layer, inputs)
                ax_acts_batch = self.ax(act_in[:, prefix_length:])  # no bos token; shape = (batch_size, seq_len, sae_width)
                seq_lens = inputs["attention_mask"].sum(dim=1) - prefix_length # no bos token

                # add max latents for each sequence
                for i in range(seq_lens.shape[0]):
                    acts = ax_acts_batch[i, :seq_lens[i]] # shape = (seq_len, sae_width)
                    label = batch["labels"][i]
                    max_acts = torch.max(acts, dim=0).values # shape = (sae_width, )
                    if label == 1:
                        all_positive_acts += [max_acts]
                    else:
                        all_negative_acts += [max_acts]

        # get latent activations
        positive_acts = torch.stack(all_positive_acts) # shape = (num_positive_examples, sae_width)
        negative_acts = torch.stack(all_negative_acts) # shape = (num_negative_examples, sae_width)
        true_labels = torch.ones(len(positive_acts))
        false_labels = torch.zeros(len(negative_acts))
        all_labels = torch.cat([true_labels, false_labels], dim=0).detach().cpu().to(torch.float32) # shape = (num_examples, )
        all_acts = torch.cat([positive_acts, negative_acts], dim=0).detach().cpu().to(torch.float32) # shape = (num_examples, sae_width)

        # compute AUC for each feature
        auc_scores = []
        for i in range(self.sae_width):
            auc_scores += [roc_auc_score(all_labels, all_acts[:, i])]

        # print top 10 features by AUC
        top = torch.tensor(auc_scores).topk(10)
        for i in range(10):
            logger.warning(f"Feature {top.indices[i].item()}: {top.values[i].item()}")
        self.top_feature = top.indices[0].item() # only pick the first feature


class GemmaScopeSAEBinaryMask(GemmaScopeSAE):
    """Learn a binary mask over SAE features and retain the selected decoder direction for inference."""
    requires_training_args = True

    def __str__(self):
        return 'GemmaScopeSAEBinaryMask'

    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "latent")
        if mode == "train":
            self.sae_params = kwargs.get("sae_params", None)
            metadata_path = kwargs.get("metadata_path", None)
            if self.sae_params is not None:
                logger.warning(f"Setting up SAE for binary mask with shape {self.sae_params['W_dec'].shape}")
                ax = SigmoidMaskAdditionIntervention(
                    embed_dim=self.model.config.hidden_size,
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                    sae_width=self.sae_params['W_dec'].shape[0],
                )
                ax.proj.weight.data = torch.from_numpy(self.sae_params['W_dec']).t()
                ax = ax.train()
                ax_config = IntervenableConfig(representations=[{
                    "layer": l,
                    "component": f"model.layers[{l}].output",
                    "low_rank_dimension": kwargs.get("low_rank_dimension", 1),
                    "intervention": ax} for l in [self.layer]])
                ax_model = IntervenableModel(ax_config, self.model)
                ax_model.set_device(self.device)
                self.ax = ax
                self.ax_model = ax_model
                self.metadata_path = metadata_path
        else:
            # defer to parent class
            super().make_model(**kwargs)

    def save(self, dump_dir, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        top_feature = self.ax.get_latent_weights().argmax().item()
        # log the top 10 features
        logger.warning(f"Top 10 features: {self.ax.get_latent_weights().topk(10).indices.tolist()}")
        top_features = []
        if os.path.exists(dump_dir / f"{model_name}_top_features.json"):
            with open(dump_dir / f"{model_name}_top_features.json", "r") as f:
                top_features = json.load(f)
        top_features += [top_feature]
        with open(dump_dir / f"{model_name}_top_features.json", "w") as f:
            json.dump(top_features, f)

        # save the pruned SAE

        save_pruned_sae(
            self.metadata_path, dump_dir, modified_refs=top_features,
            savefile=f"{model_name}.pt", sae_params=self.sae_params,
        )
        self._save_calibration_scale(dump_dir, model_name)

    def _selected_feature_id(self):
        return int(self.ax.get_latent_weights().argmax().item())

    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples, **kwargs)
        torch.cuda.empty_cache()

        # Optimizer and lr
        optimizer = torch.optim.AdamW(
            self.ax_model.parameters(),
            lr=self.training_args.lr, weight_decay=self.training_args.weight_decay)
        num_training_steps = self.training_args.n_epochs * max(1, len(train_dataloader) // self.training_args.gradient_accumulation_steps)
        temperature_start = float(self.training_args.temperature_start)
        temperature_end = float(self.training_args.temperature_end)
        temperature_schedule = (
            torch.linspace(
                temperature_start, temperature_end, num_training_steps
            ).to(torch.bfloat16).to("cuda")
        )
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer,
            num_warmup_steps=0, num_training_steps=num_training_steps)
        norm_loss_fn = torch.nn.MSELoss()
        # Main training loop.
        rank = self.process_rank
        progress_bar, curr_step = tqdm(range(num_training_steps), position=rank, leave=True), 0

        for epoch in range(self.training_args.n_epochs):
            for step, batch in enumerate(train_dataloader):
                # prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                unit_locations={"sources->base": (
                    None,
                    inputs["intervention_locations"].permute(1, 0, 2).tolist()
                )}

                # forward
                _, cf_outputs = self.ax_model(
                    base={
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"]
                    }, unit_locations=unit_locations, labels=inputs["labels"],
                    use_cache=False, subspaces=[{"idx": 0}]*self.num_of_layers)

                # loss
                loss = cf_outputs.loss.mean()
                loss /= self.training_args.gradient_accumulation_steps
                # grads
                loss.backward()

                # Perform optimization step every gradient_accumulation_steps
                if (step + 1) % self.training_args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    temp = temperature_schedule[curr_step] if len(temperature_schedule) > curr_step else temperature_end
                    self.ax_model.set_temperature(temp)
                    torch.nn.utils.clip_grad_norm_(self.ax_model.parameters(), 1.0)
                    curr_step += 1
                    curr_lr = get_lr(optimizer)
                    # optim
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        lr=f"{curr_lr:.6f}", loss=f"{loss:.6f}", temp=f"{temp:.6f}")
        progress_bar.close()
