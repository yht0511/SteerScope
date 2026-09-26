from .model import Model
import torch, einops, random, math
from tqdm.auto import tqdm
import os
import pandas as pd
import torch.nn.functional as F
from typing import Tuple
from pyvene import (
    IntervenableConfig,
    IntervenableModel
)
from .interventions import (
    TopKReLUIntervention,
    TopKReLUSubspaceIntervention,
    AdditionIntervention,
    SubspaceIntervention,
    ThresholdingIntervention,
    SteeringVectorIntervention,
    PreferenceVectorIntervention,
    AdditionSuppressionIntervention,
)
from ..utils.constants import EXAMPLE_TAG
from torch.utils.data import DataLoader
from ..utils.model_utils import (
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
    gather_residual_activations,
    get_lr,
    calculate_l1_losses
)
import numpy as np
from transformers import get_scheduler
from transformers import set_seed
from ..inference.utils import prepare_df
from ..utils.data_utils import make_preference_data_module
from ..utils.training import (
    GRADIENT_ACCUMULATION_SEMANTICS,
    is_optimizer_step,
    normalize_loss_for_accumulation,
    optimizer_steps_per_epoch,
)


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute masked sequence log probabilities, adapted from eric-mitchell/direct-preference-optimization."""
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # Replace masked labels with a valid gather index.
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    gemma: float,
                    simpo_scaler: float,
                    winning_lens: torch.LongTensor,
                    losing_lens: torch.LongTensor,
                    label_smoothing: float = 0.0,
                    loss_type: str = "dpo",
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute DPO-family losses and detached chosen/rejected rewards."""

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    ref_logratios_reverse = reference_rejected_logps - reference_chosen_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if loss_type == "ipo":
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    elif loss_type == "dpo":
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
    elif loss_type == "simpo":
        losses = -F.logsigmoid((beta / winning_lens) * policy_chosen_logps - (beta / losing_lens) * policy_rejected_logps - gemma)
    elif loss_type == "scaled_simpo":
        scaled_policy_chosen_logps = (
            torch.max(ref_logratios_reverse * simpo_scaler, torch.ones_like(ref_logratios_reverse)) / winning_lens) * policy_chosen_logps
        scaled_policy_rejected_logps = (1.0 / losing_lens) * policy_rejected_logps
        losses = -F.logsigmoid(scaled_policy_chosen_logps - scaled_policy_rejected_logps)
        """
        negative steering:

        input: steering prefix + original instruction
        winning output: original response
        losing output: steered response

        scaler = p_ref(losing output) - p_ref(winning output)
        losses = -F.logsigmoid(
            (torch.max(scaler, 1) / winning_lens) * policy_chosen_logps - (1.0 / losing_lens) * policy_rejected_logps)
        """
    elif loss_type == "apo_zero":
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        losses = -F.logsigmoid(beta * chosen_logratios) + F.logsigmoid(beta * rejected_logratios)
    else:
        raise ValueError(f"Loss type {loss_type} not supported")

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def masked_kl_distillation_loss(student_logits, teacher_logits, labels):
    """Compute per-example next-token KL loss over positions whose labels are not masked."""
    # Ensure the shapes align
    assert student_logits.shape[:-1] == labels.shape, "student_logits and labels shape mismatch"
    assert teacher_logits.shape[:-1] == labels.shape, "teacher_logits and labels shape mismatch"

    # Align each logit with its next-token label.
    labels = labels[:, 1:].clone()
    student_logits = student_logits[:, :-1, :]
    teacher_logits = teacher_logits[:, :-1, :]

    # Create a mask for valid tokens (labels != -100).
    loss_mask = (labels != -100).float()  # shape: (batch_size, seq_len-1)

    # Convert teacher logits to probabilities and student logits to log-probabilities.
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    # Compute elementwise KL divergence for each token (over the class dimension).
    kl_elementwise = F.kl_div(student_log_probs, teacher_probs, reduction='none')
    token_kl = kl_elementwise.sum(dim=-1)  # shape: (batch_size, seq_len-1)

    # Compute the loss per sample: sum over tokens and divide by the number of valid tokens.
    sample_loss = (token_kl * loss_mask).sum(dim=1) / loss_mask.sum(dim=1)
    return sample_loss


class PreferenceModel(Model):
    # the base class for all preference models
    preference_pairs = ["orig_add"] # "orig_add", "orig_sub", "steered_add", "steered_sub"
    def __str__(self):
        return 'PreferenceModel'

    @classmethod
    def training_fingerprint_context(cls):
        return {
            **super().training_fingerprint_context(),
            "preference_training_version": 3,
            "partial_batch_semantics": "actual_collated_batch_size",
            "inner_minibatch_semantics": "example_weighted_mean_v1",
            "gradient_accumulation_semantics": (
                GRADIENT_ACCUMULATION_SEMANTICS
            ),
        }

    def _actual_preference_batch_size(self, batch):
        """Return the collated batch size, including a final partial batch."""
        if not self.preference_pairs:
            raise ValueError("Preference training requires at least one pair type.")
        sizes = {
            int(batch[f"{pair}_{side}_input_ids"].shape[0])
            for pair in self.preference_pairs
            for side in ("winning", "losing")
        }
        if len(sizes) != 1:
            raise ValueError(
                f"Preference batch fields have inconsistent sizes: {sorted(sizes)}."
            )
        return sizes.pop()

    @staticmethod
    def _normalize_preference_minibatch_loss(
        steer_losses,
        *,
        expanded_batch_size,
        outer_step,
        num_outer_microbatches,
        accumulation_steps,
    ):
        """Weight an inner minibatch and its outer accumulation window."""

        minibatch_loss = steer_losses.sum() / expanded_batch_size
        return normalize_loss_for_accumulation(
            minibatch_loss,
            outer_step,
            num_outer_microbatches,
            accumulation_steps,
        )

    @staticmethod
    def _pair_preference_examples(examples):
        """Convert the benchmark's paired rows into RePS preference triples."""
        preference_columns = {"winning_output", "losing_output"}
        present_columns = preference_columns.intersection(examples.columns)
        if present_columns:
            if present_columns != preference_columns:
                raise ValueError(
                    "RePS data must contain both winning_output and "
                    "losing_output."
                )
            if examples[list(preference_columns)].isna().any().any():
                raise ValueError("RePS data contains a missing preference output.")
            return examples.reset_index(drop=True)

        required_columns = {"pair_id", "input", "output", "category"}
        missing_columns = required_columns - set(examples.columns)
        if missing_columns:
            raise ValueError(
                "RePS requires paired positive and neutral rows with columns: "
                f"{sorted(missing_columns)}."
            )

        positive = examples[examples["category"] == "positive"].copy()
        neutral = examples[examples["category"] == "negative"].copy()
        if positive.empty or neutral.empty:
            raise ValueError("RePS requires both positive and neutral examples.")
        if positive["pair_id"].duplicated().any() or neutral["pair_id"].duplicated().any():
            raise ValueError("RePS requires exactly one row per category and pair_id.")

        positive_ids = set(positive["pair_id"].tolist())
        neutral_ids = set(neutral["pair_id"].tolist())
        if positive_ids != neutral_ids:
            raise ValueError(
                "RePS positive and neutral rows must have identical pair_id sets."
            )

        neutral_by_pair = neutral.set_index("pair_id")
        aligned_neutral = neutral_by_pair.loc[positive["pair_id"].tolist()]
        if positive["input"].tolist() != aligned_neutral["input"].tolist():
            raise ValueError(
                "RePS positive and neutral rows must share the same input for "
                "each pair_id."
            )
        if positive["output"].isna().any() or aligned_neutral["output"].isna().any():
            raise ValueError("RePS paired rows contain a missing output.")

        positive["winning_output"] = positive["output"]
        positive["losing_output"] = aligned_neutral["output"].tolist()
        return positive.reset_index(drop=True)

    def make_preference_dataloader(self, examples, **kwargs):
        examples = self._pair_preference_examples(examples)
        data_module = make_preference_data_module(self.tokenizer, examples, **kwargs)
        g = torch.Generator()
        g.manual_seed(self.seed)
        train_dataloader = DataLoader(
            data_module["train_dataset"], shuffle=True,
            batch_size=self.training_args.batch_size,
            collate_fn=data_module["data_collator"],
            generator=g)
        return train_dataloader

    def _compute_metrics(self, chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps,
                         chosen_rewards, rejected_rewards, losses):
        """Helper method to compute metrics for a minibatch"""
        metrics = {}

        # Compute reward accuracies
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        metrics[f'rewards_train/steer_chosen_rewards'] = chosen_rewards.mean().cpu().float().numpy().tolist()
        metrics[f'rewards_train/steer_rejected_rewards'] = rejected_rewards.mean().cpu().float().numpy().tolist()
        metrics[f'rewards_train/steer_margins'] = (chosen_rewards - rejected_rewards).mean().cpu().float().numpy().tolist()
        metrics[f'rewards_train/pos_steer_reward_accuracies'] = np.array(reward_accuracies.cpu().numpy().tolist()[:len(reward_accuracies)//2]).mean()
        metrics[f'rewards_train/neg_steer_reward_accuracies'] = np.array(reward_accuracies.cpu().numpy().tolist()[len(reward_accuracies)//2:]).mean()
        metrics[f'rewards_train/steer_accuracies'] = reward_accuracies.mean().cpu().numpy().tolist()
        metrics[f'logps_train/steer_chosen'] = chosen_logps.detach().mean().cpu().float().numpy().tolist()
        metrics[f'logps_train/steer_rejected'] = rejected_logps.detach().mean().cpu().float().numpy().tolist()
        metrics[f'loss/steer'] = losses.mean().detach().cpu().float().numpy().tolist()

        return metrics

    def train(self, examples, **kwargs):
        if self.use_wandb:
            import wandb
            logging_metadata = kwargs["logging_metadata"]
            run_name = f"{logging_metadata['model_name']}_{logging_metadata['layer']}_{logging_metadata['concept_id']}"
            wandb_proj = kwargs.get("wandb_project", None)
            wandb_name = kwargs.get("wandb_name", None)
            run = wandb.init(
                project=f"{wandb_proj}",
                entity=wandb_name,
                name=run_name,
                dir="wandb",
            )

        train_dataloader = self.make_preference_dataloader(
            examples, **kwargs)
        torch.cuda.empty_cache()

        # Optimizer and lr
        optimizer = torch.optim.AdamW(
            self.ax_model.parameters(),
            lr=self.training_args.lr, weight_decay=self.training_args.weight_decay)
        print(optimizer.param_groups)
        accumulation_steps = int(
            self.training_args.gradient_accumulation_steps or 1
        )
        num_microbatches = len(train_dataloader)
        num_training_steps = self.training_args.n_epochs * (
            optimizer_steps_per_epoch(num_microbatches, accumulation_steps)
        )
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer,
            num_warmup_steps=0, num_training_steps=num_training_steps)
        # Main training loop.
        rank = self.process_rank
        progress_bar, curr_step, logging_step = tqdm(range(num_training_steps), position=rank, leave=True), 0, 0

        for epoch in range(self.training_args.n_epochs):
            for step, batch in enumerate(train_dataloader):
                actual_batch_size = self._actual_preference_batch_size(batch)
                expanded_batch_size = actual_batch_size * len(self.preference_pairs)
                minibatch_size = self.training_args.batch_size
                num_minibatches = (expanded_batch_size + minibatch_size - 1) // minibatch_size

                winning_inputs = {
                    "input_ids": [],
                    "attention_mask": [],
                    "labels": [],
                    "intervention_locations": [],
                    "steering_factors": [],
                }
                losing_inputs = {
                    "input_ids": [],
                    "attention_mask": [],
                    "labels": [],
                    "intervention_locations": [],
                    "steering_factors": [],
                }
                for i in range(actual_batch_size):
                    for pair in self.preference_pairs:
                        winning_inputs["input_ids"].append(batch[f"{pair}_winning_input_ids"][i])
                        winning_inputs["attention_mask"].append(batch[f"{pair}_winning_attention_mask"][i])
                        winning_inputs["labels"].append(batch[f"{pair}_winning_labels"][i])
                        winning_inputs["intervention_locations"].append(batch[f"{pair}_winning_intervention_locations"][i])
                        losing_inputs["input_ids"].append(batch[f"{pair}_losing_input_ids"][i])
                        losing_inputs["attention_mask"].append(batch[f"{pair}_losing_attention_mask"][i])
                        losing_inputs["labels"].append(batch[f"{pair}_losing_labels"][i])
                        losing_inputs["intervention_locations"].append(batch[f"{pair}_losing_intervention_locations"][i])
                        if "_add" in pair:
                            winning_inputs["steering_factors"].append(torch.tensor(random.choice(self.training_args.steering_factors)))
                            losing_inputs["steering_factors"].append(torch.tensor(random.choice(self.training_args.steering_factors)))
                        else:
                            if self.training_args.substraction_type == "null_it_out":
                                winning_inputs["steering_factors"].append(torch.tensor(0.0))
                                losing_inputs["steering_factors"].append(torch.tensor(0.0))
                            else:
                                winning_inputs["steering_factors"].append(torch.tensor(-1.0 * random.choice(self.training_args.steering_factors)))
                                losing_inputs["steering_factors"].append(torch.tensor(-1.0 * random.choice(self.training_args.steering_factors)))

                batch_metrics = {}
                loss_sum = 0

                for mb in range(num_minibatches):
                    start_idx = mb * minibatch_size
                    end_idx = min((mb + 1) * minibatch_size, expanded_batch_size)

                    if start_idx >= expanded_batch_size:
                        break

                    minibatch_inputs = {
                        k: torch.stack(winning_inputs[k][start_idx:end_idx]+losing_inputs[k][start_idx:end_idx], dim=0).to(self.device)
                        for k, _ in winning_inputs.items()
                    }

                    if isinstance(self.ax, list):
                        unit_locations = {"sources->base": (
                            None,
                            # repeat along first dimension
                            minibatch_inputs["intervention_locations"].permute(1, 0, 2).tolist() * len(self.ax)
                        )}
                    else:
                        unit_locations = {"sources->base": (
                            None,
                            minibatch_inputs["intervention_locations"].permute(1, 0, 2).tolist()
                        )}

                    subspaces = [{
                        "k": self.training_args.topk,
                        "steering_factor": minibatch_inputs["steering_factors"],
                    }]
                    subspace_repeat = 1 if not isinstance(self.ax, list) else len(self.ax)
                    subspaces = subspaces * subspace_repeat

                    ref_outputs, policy_outputs_orig = self.ax_model(
                        base={
                            "input_ids": minibatch_inputs["input_ids"],
                            "attention_mask": minibatch_inputs["attention_mask"]
                        }, unit_locations=unit_locations,
                        output_original_output=True,
                        subspaces=subspaces, use_cache=False)



                    policy_outputs_orig_logps = _get_batch_logps(
                        policy_outputs_orig.logits, minibatch_inputs["labels"],
                        average_log_prob=False)
                    ref_logps = _get_batch_logps(
                        ref_outputs.logits, minibatch_inputs["labels"],
                        average_log_prob=False)

                    minibatch_size_actual = minibatch_inputs["input_ids"].shape[0]
                    steer_chosen_logps = policy_outputs_orig_logps[:minibatch_size_actual//2]
                    steer_rejected_logps = policy_outputs_orig_logps[minibatch_size_actual//2:]
                    steer_ref_chosen_logps = ref_logps[:minibatch_size_actual//2]
                    steer_ref_rejected_logps = ref_logps[minibatch_size_actual//2:]

                    winning_lens = minibatch_inputs["attention_mask"][:minibatch_size_actual//2].sum(dim=-1)
                    losing_lens = minibatch_inputs["attention_mask"][minibatch_size_actual//2:].sum(dim=-1)
                    pos_loss_kwargs = {
                        'beta': self.training_args.beta,
                        'gemma': self.training_args.gemma,
                        'simpo_scaler': self.training_args.simpo_scaler,
                        'reference_free': self.training_args.reference_free,
                        'label_smoothing': self.training_args.label_smoothing,
                        'loss_type': self.training_args.loss_type,
                        'winning_lens': winning_lens,
                        'losing_lens': losing_lens}
                    steer_losses, steer_chosen_rewards, steer_rejected_rewards = preference_loss(
                        steer_chosen_logps, steer_rejected_logps,
                        steer_ref_chosen_logps, steer_ref_rejected_logps,
                        **pos_loss_kwargs)

                    steer_loss = steer_losses.mean()
                    # Weight partial inner and outer windows by their actual sizes.
                    minibatch_loss = self._normalize_preference_minibatch_loss(
                        steer_losses,
                        expanded_batch_size=expanded_batch_size,
                        outer_step=step,
                        num_outer_microbatches=num_microbatches,
                        accumulation_steps=accumulation_steps,
                    )

                    # Backward pass for this minibatch
                    minibatch_loss.backward()

                    # Track total loss for logging
                    loss_sum += steer_loss.detach() * (end_idx - start_idx)

                    # Accumulate metrics for this minibatch
                    minibatch_metrics = self._compute_metrics(
                        steer_chosen_logps, steer_rejected_logps,
                        steer_ref_chosen_logps, steer_ref_rejected_logps,
                        steer_chosen_rewards, steer_rejected_rewards,
                        steer_losses
                    )

                    # Accumulate metrics
                    for k, v in minibatch_metrics.items():
                        if k not in batch_metrics:
                            batch_metrics[k] = [v * (end_idx - start_idx)]
                        else:
                            batch_metrics[k].append(v * (end_idx - start_idx))

                # Calculate the average loss and metrics across all minibatches
                metrics = {}
                for k, v in batch_metrics.items():
                    metrics[k] = sum(v) / expanded_batch_size

                loss = loss_sum / expanded_batch_size
                metrics[f'loss/train'] = loss.cpu().float().numpy().tolist()
                metrics[f'loss/steer'] = loss.cpu().float().numpy().tolist()

                # Perform optimization step every gradient_accumulation_steps
                if is_optimizer_step(
                    step,
                    num_microbatches,
                    accumulation_steps,
                ):
                    torch.nn.utils.clip_grad_norm_(self.ax_model.parameters(), 1.0)
                    curr_lr = get_lr(optimizer)
                    # optim
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    progress_bar.set_description(
                        "lr %.6f || loss %.6f || steer acc %.6f" % (
                            curr_lr, loss, metrics.get('rewards_train/steer_accuracies', 0.0)))
                    if self.use_wandb:
                        wandb.log(metrics, step=curr_step)
                    curr_step += 1



        progress_bar.close()
        if self.use_wandb:
            run.finish()

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        self.max_activations = {}
        return self.max_activations
