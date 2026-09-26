import torch, random
from torch import nn
from pyvene import (
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
    CollectIntervention,
    InterventionOutput,
    SigmoidMaskIntervention,
)


class LowRankRotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, m, init_orth=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(n, m), requires_grad=True)
        if init_orth:
            torch.nn.init.orthogonal_(self.weight)

    def forward(self, x):
        return torch.matmul(x.to(self.weight.dtype), self.weight)


class TopKReLUSubspaceIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    Phi(h) = (h - h@v) + Mean(TopK(ReLU(h@v)))*v
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"])
        with torch.no_grad():
            self.proj.bias.fill_(0)

    def forward(
        self, base, source=None, subspaces=None
    ):
        v = []
        if "subspaces" in subspaces:
            for subspace in subspaces["subspaces"]:
                v += [self.proj.weight[subspace]]
        else:
            for i in range(base.shape[0]):
                v += [self.proj.weight[0]]
        v = torch.stack(v, dim=0).unsqueeze(dim=-1) # bs, h, 1
        
        # get latent
        latent = torch.relu(torch.bmm(base, v)).squeeze(dim=-1) # bs, s, 1
        topk_acts, topk_indices = latent.topk(k=subspaces["k"], dim=-1, sorted=False)
        non_topk_latent = latent.clone()
        non_topk_latent.scatter_(-1, topk_indices, 0)

        # get orthogonal component
        proj_vec = torch.bmm(latent.unsqueeze(dim=-1), v.permute(0, 2, 1)) # bs, s, 1 * bs, 1, h = bs, s, h
        base_orthogonal = base - proj_vec

        # get steering magnitude using mean of topk activations of prompt latent
        max_mean_latent = topk_acts.mean(dim=-1, keepdim=True) # bs, 1
        # steering vector
        steering_vec = max_mean_latent.unsqueeze(dim=-1) * v.permute(0, 2, 1) # bs, 1, h

        # addition intervention
        output = base_orthogonal + steering_vec

        return InterventionOutput(
            output=output.to(base.dtype),
            latent=[latent, non_topk_latent]
        )


class HyperAdditionIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        # Pretrained weights are loaded after zero initialization.
        super().__init__(**kwargs, keep_last_dim=True)
        self.low_rank_dimension = kwargs["low_rank_dimension"]
        self.v : torch.Tensor = None
    
    def _update_v(self, new_vect: torch.Tensor):
        self.v = new_vect
        
    def _reset_v(self):
        self.v = None

    def forward(self, base, source=None, subspaces=None):
        # use subspaces["idx"] to select the correct weight vector
        steering_vec = subspaces["mag"].unsqueeze(dim=-1) * self.v
        output = base + steering_vec.unsqueeze(dim=1)
        return output


class HyperTopKReLUIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    Phi(h) = h + Mean(TopK(ReLU(h@v)))*v
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.low_rank_dimension = kwargs["low_rank_dimension"]
        self.v : torch.Tensor = None
    
    def _update_v(self, new_vect: torch.Tensor):
        self.v = new_vect
        
    def _reset_v(self):
        self.v = None

    def forward(
        self, base, source=None, subspaces=None
    ):
        assert self.v is not None, "v is not set. Please set v before calling forward."
        assert self.v.shape == (base.shape[0], self.embed_dim), "v shape mismatch."
        v = self.v.unsqueeze(dim=-1) # bs, h, 1
        # steering vector
        steering_vec = v.permute(0, 2, 1) # bs, 1, h
        # addition intervention
        output = base + steering_vec

        return InterventionOutput(
            output=output.to(base.dtype)
        )
        
        
class SimpleAdditionIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    Phi(h) = h + Mean(TopK(ReLU(h@v)))*v
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.low_rank_dimension = kwargs["low_rank_dimension"]
        self.v : torch.Tensor = None
    
    def _update_v(self, new_vect: torch.Tensor):
        self.v = new_vect
        
    def _reset_v(self):
        self.v = None

    def forward(
        self, base, source=None, subspaces=None
    ):
        assert self.v is not None, "v is not set. Please set v before calling forward."
        assert self.v.shape == (base.shape[0], self.embed_dim), "v shape mismatch."
        steering_vec = self.v.unsqueeze(dim=1) # bs, 1, h
        if subspaces:
            if "mag" in subspaces.keys():
                # If a scaler magnitude is provided, scale the steering vector
                steering_vec = subspaces["mag"].unsqueeze(-1).unsqueeze(-1) * steering_vec  
        
        # addition intervention
        output = base + steering_vec

        return InterventionOutput(
            output=output.to(base.dtype)
        )
        
class TopKReLUIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    Phi(h) = h + Mean(TopK(ReLU(h@v)))*v
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"])
        with torch.no_grad():
            self.proj.weight.fill_(0.01)
            self.proj.bias.fill_(0)

    def forward(
        self, base, source=None, subspaces=None
    ):
        v = []
        if "subspaces" in subspaces:
            for subspace in subspaces["subspaces"]:
                v += [self.proj.weight[subspace]]
        else:
            for i in range(base.shape[0]):
                v += [self.proj.weight[0]]
        v = torch.stack(v, dim=0).unsqueeze(dim=-1) # bs, h, 1
        
        # get latent
        latent = torch.relu(torch.bmm(base, v)).squeeze(dim=-1) # bs, s, 1
        topk_acts, topk_indices = latent.topk(k=subspaces["k"], dim=-1, sorted=False)
        non_topk_latent = latent.clone()
        non_topk_latent.scatter_(-1, topk_indices, 0)

        # get steering magnitude using mean of topk activations of prompt latent
        max_mean_latent = topk_acts.mean(dim=-1, keepdim=True) # bs, 1
        # steering vector
        steering_vec = max_mean_latent.unsqueeze(dim=-1) * v.permute(0, 2, 1) # bs, 1, h

        # addition intervention
        output = base + steering_vec

        return InterventionOutput(
            output=output.to(base.dtype),
            latent=[latent, non_topk_latent]
        )


class ConceptReFTIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """Concept-level rank-one ReFT intervention from arXiv:2404.03592."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.W_proj = nn.Parameter(torch.zeros(
            kwargs["n_concepts"], self.embed_dim, kwargs["low_rank_dimension"]))
        self.W_source = nn.Parameter(torch.zeros(
            kwargs["n_concepts"], self.embed_dim, kwargs["low_rank_dimension"]))
        self.b_source = nn.Parameter(torch.zeros(
            kwargs["n_concepts"], kwargs["low_rank_dimension"]))

    def encode(
        self, base, source=None, subspaces=None
    ):
        """High-dimensional concept space."""
        proj_weight = self.W_proj[subspaces["input_subspaces"]] # batch_size, embed_dim, low_rank_dimension
        rotated_base = torch.bmm(base, proj_weight) # [batch_size, seq_len, embed_dim] X [batch_size, embed_dim, low_rank_dimension]

        return rotated_base # batch_size, seq_len, low_rank_dimension

    def forward(
        self, base, source=None, subspaces=None
    ):
        proj_weight = self.W_proj[subspaces["idx"]] # batch_size, embed_dim, low_rank_dimension
        source_weight = self.W_source[subspaces["idx"]] # batch_size, embed_dim, low_rank_dimension
        source_bias = self.b_source[subspaces["idx"]].unsqueeze(dim=1) # batch_size, 1, low_rank_dimension

        rotated_base = torch.bmm(base.float(), proj_weight) # batch_size, seq_len, low_rank_dimension
        output = base + torch.bmm(
            ((torch.bmm(base, source_weight) + source_bias) - rotated_base), # batch_size, seq_len, low_rank_dimension
            proj_weight.transpose(-1, -2)
        )
        return output.to(base.dtype)
    

class AdditionIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        # Note that we initialise these to zeros because we're loading in pre-trained weights.
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj = torch.nn.Linear(
                self.embed_dim, kwargs["low_rank_dimension"], bias=True)

    def forward(self, base, source=None, subspaces=None):
        # use subspaces["idx"] to select the correct weight vector
        steering_vec = subspaces["max_act"].unsqueeze(dim=-1) * \
            subspaces["mag"].unsqueeze(dim=-1) * self.proj.weight[subspaces["idx"]]
        output = base + steering_vec.unsqueeze(dim=1)
        return output
    

class AdditionSuppressionIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        # Note that we initialise these to zeros because we're loading in pre-trained weights.
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj = torch.nn.Linear(
                self.embed_dim, kwargs["low_rank_dimension"], bias=True)
        self.bias = nn.Parameter(torch.zeros(kwargs["low_rank_dimension"]))

    def forward(self, base, source=None, subspaces=None):
        # use subspaces["idx"] to select the correct weight vector
        steering_vec = self.proj.weight[subspaces["idx"]]

        neg_mask = subspaces["mag"] <= 0  # Suppression or null-out training.
        pos_mask = subspaces["mag"] >= 0.0  # Positive steering.

        neg_steering_factor = torch.einsum("b, b->b", (subspaces["mag"] - self.proj.bias[subspaces["idx"]]), neg_mask) # bs, s
        pos_steering_factor = torch.einsum("b, b->b", (subspaces["mag"] + self.proj.bias[subspaces["idx"]]), pos_mask) # bs, 1
        combined_steering_factor = neg_steering_factor + pos_steering_factor # bs, s
        steering_vec = torch.einsum("bs, b->bs", steering_vec, combined_steering_factor) # bs, s, d
        output = base + steering_vec.unsqueeze(dim=1)
        return output


class AdditionSuppressionLongContextIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        # Note that we initialise these to zeros because we're loading in pre-trained weights.
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj = torch.nn.Linear(
                self.embed_dim, kwargs["low_rank_dimension"], bias=True)
        self.bias = nn.Parameter(torch.zeros(kwargs["low_rank_dimension"]))

    def forward(self, base, source=None, subspaces=None):
        # use subspaces["idx"] to select the correct weight vector
        steering_vec = self.proj.weight[subspaces["idx"]]
        neg_mask = subspaces["mag"] <= 0 # bs, 1, 1, this is only for null it out training #this is when applying suppression
        pos_mask = subspaces["mag"] > 0.0  # Positive steering.

        latent = torch.ones_like((torch.bmm(base, steering_vec.unsqueeze(-1))+ self.proj.bias[subspaces["idx"]].unsqueeze(-1).unsqueeze(-1)) > 0) # bs, s, 1
        if latent.shape[1] > 100:
            mask = torch.zeros_like(latent)  # Create a mask of zeros with same shape as latent
            mask[:, -100:, :] = 1  # Set first 100 positions to 1
            latent = latent * mask
        
        neg_mask = torch.einsum("bsq, b->bsq", latent, neg_mask)
        neg_steering_factor = torch.einsum("bsq, b->bs", neg_mask, (subspaces["mag"] - self.proj.bias[subspaces["idx"]])) # bs, s
        pos_steering_factor = torch.einsum("b, b->b", (subspaces["mag"] + self.proj.bias[subspaces["idx"]]), pos_mask).unsqueeze(-1) # bs, 1
        combined_steering_factor = neg_steering_factor + pos_steering_factor # bs, s
        steering_vec = torch.einsum("bh, bs->bsh", steering_vec, combined_steering_factor) # bs, s, d
        output = base + steering_vec

        return output
    

class SamplingAdditionIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj = torch.nn.Linear(
                self.embed_dim, kwargs["low_rank_dimension"], bias=True)

    def forward(self, base, source=None, subspaces=None):
        # Normalize base to unit vectors along the embedding dimension
        base_norm = torch.norm(base, p=2, dim=-1, keepdim=True)
        base_normalized = base / (base_norm + 1e-8)  # Add epsilon to prevent division by zero
        steering_direction = self.proj.weight[subspaces["idx"]]  # [batch, d]

        alpha = torch.tensor(2.0, device=base.device)
        beta = torch.tensor(1.0, device=base.device)
        interpolation = torch.distributions.Beta(alpha, beta).sample(
            (base.shape[0],)).to(base.device)  # [batch]
        interpolation = interpolation.unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]
        steering_direction = steering_direction.unsqueeze(1)  # [batch, 1, d]
        combined_direction = (1 - interpolation) * base_normalized + \
                           interpolation * steering_direction
        
        combined_direction = combined_direction / \
            (torch.norm(combined_direction, p=2, dim=-1, keepdim=True) + 1e-8)
        
        steering_magnitude = subspaces["max_act"].unsqueeze(-1) * \
            subspaces["mag"].unsqueeze(-1)  # [batch, 1]
        output = base + (steering_magnitude.unsqueeze(1) * combined_direction)
        
        return output


class ThresholdingIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj = torch.nn.Linear(
                self.embed_dim, kwargs["low_rank_dimension"], bias=True)

    def forward(self, base, source=None, subspaces=None):
        steering_direction = self.proj.weight[subspaces["idx"]]  # [batch, embed_dim]
        projection = torch.bmm(
            base,  # [batch, seq_len, embed_dim]
            steering_direction.unsqueeze(-1)  # [batch, embed_dim, 1]
        ).squeeze(-1)  # [batch, seq_len]
        desired_magnitude = subspaces["max_act"] * subspaces["mag"]  # [batch]
        mask = (projection < desired_magnitude.unsqueeze(-1)).float()  # [batch, seq_len]
        steering_vec = desired_magnitude.unsqueeze(-1) * steering_direction  # [batch, embed_dim]
        masked_steering = mask.unsqueeze(-1) * steering_vec.unsqueeze(1)  # [batch, seq_len, embed_dim]
        output = base + masked_steering
        
        return output


class SigmoidMaskAdditionIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        # The low-rank dimension indexes concept-specific SAE masks.
        self.proj = torch.nn.Linear(
            kwargs["sae_width"], self.embed_dim, bias=True)
        self.mask = torch.nn.Parameter(
            torch.zeros(kwargs["low_rank_dimension"], kwargs["sae_width"]), requires_grad=True)
        self.source = torch.nn.Parameter(
            0.001 *torch.ones(kwargs["low_rank_dimension"], kwargs["sae_width"]), requires_grad=True)
        self.temperature = torch.nn.Parameter(torch.tensor(0.01), requires_grad=False)
    
    def get_temperature(self) -> torch.Tensor:
        return self.temperature

    def set_temperature(self, temp: torch.Tensor) -> None:
        self.temperature.data = temp
    
    def get_latent_weights(self) -> torch.Tensor:
        mask_sigmoid = torch.sigmoid(self.mask / torch.tensor(self.temperature))
        masked_source = (torch.relu(self.source) * mask_sigmoid).unsqueeze(0)
        return masked_source
    
    def forward(self, base, source=None, subspaces=None) -> torch.Tensor:
        # use subspaces["idx"] to select the correct weight vector
        masked_source = self.get_latent_weights()
        steering_vec = self.proj(masked_source)
        output = base + steering_vec.unsqueeze(dim=1)
        return output


class SubspaceIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=True)
    
    def forward(self, base, source=None, subspaces=None):
        prefix_length = subspaces["prefix_length"]
        if base.shape[1] > 1:
            cached_base_prefix = base[:,:prefix_length].clone()
        v = self.proj.weight[subspaces["idx"]].unsqueeze(dim=-1) # bs, h, 1

        # get orthogonal component
        latent = torch.relu(torch.bmm(base, v)) # bs, s, 1
        proj_vec = torch.bmm(latent, v.permute(0, 2, 1)) # bs, s, 1 * bs, 1, h = bs, s, h
        base_orthogonal = base - proj_vec

        steering_scale = subspaces["max_act"].unsqueeze(-1).unsqueeze(-1) * \
            subspaces["mag"].unsqueeze(-1).unsqueeze(-1)
        steering_vec = steering_scale * v.permute(0, 2, 1) # bs, 1, h
        
        # Replace the projection component with the steering vector
        output = base_orthogonal + steering_vec 
        if base.shape[1] > 1:
            output[:,:prefix_length] = cached_base_prefix
        return output


class DictionaryAdditionIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """SAE feature replacement with reconstruction-error preservation."""
    def __init__(self, **kwargs):
        # Pretrained SAE weights are loaded after zero initialization.
        super().__init__(**kwargs, keep_last_dim=True)
        self.W_enc = nn.Parameter(torch.zeros(self.embed_dim, kwargs["low_rank_dimension"]))
        self.W_dec = nn.Parameter(torch.zeros(kwargs["low_rank_dimension"], self.embed_dim))
        self.threshold = nn.Parameter(torch.zeros(kwargs["low_rank_dimension"]))
        self.b_enc = nn.Parameter(torch.zeros(kwargs["low_rank_dimension"]))
        self.b_dec = nn.Parameter(torch.zeros(self.embed_dim))
    
    def encode(self, input_acts):
        pre_acts = torch.matmul(input_acts, self.W_enc) + self.b_enc  # Shape: [batch_size, seq_len, low_rank_dimension]
        mask = (pre_acts > self.threshold)  # Shape: [batch_size, seq_len, low_rank_dimension]
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        reconstructed = torch.matmul(acts, self.W_dec) + self.b_dec  # Shape: [batch_size, seq_len, embed_dim]
        return reconstructed

    def forward(self, base, source=None, subspaces=None):
        """
        base: Residual stream activity x, shape [batch_size, seq_len, embed_dim]
        subspaces: Dictionary containing 'idx' and 'mag'
        """
        acts = self.encode(base)
        SAE_x = self.decode(acts)
        error_x = base - SAE_x
        
        acts_modified = acts.clone()
        feature_acts = subspaces['mag'] * subspaces["max_act"]
        acts_modified[:, :, subspaces['idx']] = feature_acts.to(base.dtype)

        modified_SAE_x = self.decode(acts_modified)
        x_new = modified_SAE_x + error_x 

        return x_new


class DictionaryMinClampingIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """SAE feature clamping with reconstruction-error preservation."""
    def __init__(self, **kwargs):
        # Note that we initialize these to zeros because we're loading in pre-trained weights.
        # If you want to train your own SAEs then we recommend using appropriate initialization.
        super().__init__(**kwargs, keep_last_dim=True)
        self.W_enc = nn.Parameter(torch.zeros(self.embed_dim, kwargs["low_rank_dimension"]))
        self.W_dec = nn.Parameter(torch.zeros(kwargs["low_rank_dimension"], self.embed_dim))
        self.threshold = nn.Parameter(torch.zeros(kwargs["low_rank_dimension"]))
        self.b_enc = nn.Parameter(torch.zeros(kwargs["low_rank_dimension"]))
        self.b_dec = nn.Parameter(torch.zeros(self.embed_dim))
    
    def encode(self, input_acts):
        pre_acts = torch.matmul(input_acts, self.W_enc) + self.b_enc  # Shape: [batch_size, seq_len, low_rank_dimension]
        mask = (pre_acts > self.threshold)  # Shape: [batch_size, seq_len, low_rank_dimension]
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        reconstructed = torch.matmul(acts, self.W_dec) + self.b_dec  # Shape: [batch_size, seq_len, embed_dim]
        return reconstructed

    def forward(self, base, source=None, subspaces=None):
        """
        base: Residual stream activity x, shape [batch_size, seq_len, embed_dim]
        subspaces: Dictionary containing 'idx' and 'mag'
        """
        acts = self.encode(base)
        SAE_x = self.decode(acts)
        error_x = base - SAE_x
        
        acts_modified = acts.clone()
        proposed_feature_acts = subspaces['mag'] * subspaces["max_act"]

        # minimum is current value if it is positive
        acts_modified[:, :, subspaces['idx']] = torch.max(
            acts[:, :, subspaces['idx']], proposed_feature_acts).to(base.dtype)

        modified_SAE_x = self.decode(acts_modified)
        x_new = modified_SAE_x + error_x 

        return x_new


class JumpReLUSAECollectIntervention(
    CollectIntervention
):
    """To collect SAE latent activations"""
    def __init__(self, **kwargs):
        # Note that we initialise these to zeros because we're loading in pre-trained weights.
        super().__init__(**kwargs, keep_last_dim=True)
        self.W_enc = nn.Parameter(torch.zeros(self.embed_dim, kwargs["low_rank_dimension"]))
        self.W_dec = nn.Parameter(torch.zeros(kwargs["low_rank_dimension"], self.embed_dim))
        self.threshold = nn.Parameter(torch.zeros(kwargs["low_rank_dimension"]))
        self.b_enc = nn.Parameter(torch.zeros(kwargs["low_rank_dimension"]))
        self.b_dec = nn.Parameter(torch.zeros(self.embed_dim))
    
    def forward(self, base, source=None, subspaces=None):
        pre_acts = base @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts
    

class ProbeIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"])

    def forward(
        self, base, source=None, subspaces=None
    ):
        v = []
        if "subspaces" in subspaces:
            for subspace in subspaces["subspaces"]:
                v += [self.proj.weight[subspace]]
        else:
            for i in range(base.shape[0]):
                v += [self.proj.weight[0]]
        v = torch.stack(v, dim=0).unsqueeze(dim=-1) # bs, h, 1
        
        # get latent
        latent = torch.bmm(base, v).squeeze(dim=-1) # bs, s

        return InterventionOutput(
            output=base,
            latent=[latent]
        )
    

class SparseProbeIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"])
        with torch.no_grad():
            self.proj.weight.fill_(0.01)
            self.proj.bias.fill_(0)

    def forward(
        self, base, source=None, subspaces=None
    ):
        v = []
        if "subspaces" in subspaces:
            for subspace in subspaces["subspaces"]:
                v += [self.proj.weight[subspace]]
        else:
            for i in range(base.shape[0]):
                v += [self.proj.weight[0]]
        v = torch.stack(v, dim=0).unsqueeze(dim=-1) # bs, h, 1
        
        # get latent
        latent = torch.relu(torch.bmm(base, v)).squeeze(dim=-1) # bs, s, 1
        topk_acts, topk_indices = latent.topk(k=subspaces["k"], dim=-1, sorted=False)
        non_topk_latent = latent.clone()
        non_topk_latent.scatter_(-1, topk_indices, 0)

        # get steering magnitude using mean of topk activations of prompt latent
        max_mean_latent = topk_acts.mean(dim=-1, keepdim=False) # bs

        return InterventionOutput(
            output=base,
            latent=[max_mean_latent, non_topk_latent, latent]
        )
    

class SteeringVectorIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    Phi(h) = h + v
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"])
        with torch.no_grad():
            self.proj.weight.fill_(0.01)
            self.proj.bias.fill_(0)

    def forward(
        self, base, source=None, subspaces=None
    ):
        v = []
        if "subspaces" in subspaces:
            for subspace in subspaces["subspaces"]:
                v += [self.proj.weight[subspace]]
        else:
            for i in range(base.shape[0]):
                v += [self.proj.weight[0]]
        v = torch.stack(v, dim=0).unsqueeze(dim=-1) # bs, h, 1
        latent = torch.relu(torch.bmm(base, v)).squeeze(dim=-1) # bs, s, 1
        steering_vec = v.permute(0, 2, 1) # bs, 1, h

        # addition intervention
        output = base + steering_vec

        return InterventionOutput(
            output=output.to(base.dtype),
            latent=[latent]
        )



class ConceptVectorIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    Phi(h) = h + v
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"])

    def forward(
        self, base, source=None, subspaces=None
    ):
        v = []
        if "subspaces" in subspaces:
            for subspace in subspaces["subspaces"]:
                v += [self.proj.weight[subspace]]
        else:
            for i in range(base.shape[0]):
                v += [self.proj.weight[0]]
        v = torch.stack(v, dim=0).unsqueeze(dim=-1) # bs, h, 1
        latent = torch.relu(torch.bmm(base, v)).squeeze(dim=-1) # bs, s, 1
        steering_vec = v.permute(0, 2, 1) # bs, 1, h

        # addition intervention
        if "steering_factor" in subspaces:
            steering_factor = subspaces["steering_factor"].unsqueeze(dim=-1).unsqueeze(dim=-1) # bs, 1, 1
            output = base + steering_factor * steering_vec
        else:
            output = base + steering_vec

        return InterventionOutput(
            output=output.to(base.dtype),
            latent=[latent]
        )
    

class PreferenceVectorIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"])
        dropout = kwargs.get("dropout", 0.0)
        self.dropout = torch.nn.Dropout(dropout)
        self.intervention_positions_dropout = kwargs.get("intervention_positions_dropout", 0.0)
        with torch.no_grad():
            self.proj.bias.fill_(0)

    def forward(
        self, base, source=None, subspaces=None
    ):
        v = []
        if "subspaces" in subspaces:
            for subspace in subspaces["subspaces"]:
                v += [self.proj.weight[subspace]]
        else:
            for i in range(base.shape[0]):
                v += [self.proj.weight[0]]
        v = torch.stack(v, dim=0).unsqueeze(dim=-1) # bs, h, 1
        v_norm = torch.norm(v, dim=1, keepdim=True) # bs, 1, 1
        latent = torch.relu((torch.bmm(base, v) + self.proj.bias).squeeze(dim=-1)) # bs, s, 1
        steering_vec = v.permute(0, 2, 1) # bs, 1, h
        steering_vec = self.dropout(steering_vec)
        
        if "steering_factor" in subspaces:
            steering_factor = subspaces["steering_factor"].unsqueeze(dim=-1).unsqueeze(dim=-1) # bs, 1, 1
            zero_mask = steering_factor == 0.0  # Null-out training.
            nonzero_mask = steering_factor != 0.0 # bs, 1, 1
            # h - (h@v)/||v||^2 * v, steering coefficient is (h@v)/||v||^2
            null_it_out_steering_factor = -(latent.unsqueeze(dim=-1) / v_norm**2)*zero_mask # bs, s, 1 * bs, 1, 1 = bs, s, 1
            combined_steering_factor = null_it_out_steering_factor + (steering_factor + self.proj.bias*nonzero_mask) # bs, s, 1
            dropout_mask = torch.rand_like(combined_steering_factor.float()) > self.intervention_positions_dropout
            combined_steering_factor *= dropout_mask
            steering_vec = steering_vec * combined_steering_factor # bs, s, d
        output = base + steering_vec

        return InterventionOutput(
            output=output.to(base.dtype),
            latent=[latent]
        )


class LoraIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    LoRA(h') = h + BAx
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.r = kwargs["low_rank_dimension"]
        self.lora_alpha = kwargs["alpha"] if "alpha" in kwargs else kwargs["low_rank_dimension"]
        if "dropout" in kwargs and kwargs["dropout"] > 0.0:
            self.lora_dropout = nn.Dropout(p=kwargs["dropout"])
        else:
            self.lora_dropout = lambda x: x

        # Actual trainable parameters
        self.lora_A = nn.Parameter(torch.zeros(kwargs["input_dim"], kwargs["low_rank_dimension"]))
        self.lora_B = nn.Parameter(torch.zeros(kwargs["low_rank_dimension"], self.embed_dim))

        # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.lora_A = nn.Parameter(self.lora_A.to(torch.bfloat16))
        self.lora_B = nn.Parameter(self.lora_B.to(torch.bfloat16))

    def forward(
        self, base, source=None, subspaces=None, **kwargs
    ):
        original_input = kwargs["args"]
        return base + self.lora_dropout(original_input) @ self.lora_A @ self.lora_B
    

class PreferenceLoraIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.r = kwargs["low_rank_dimension"]
        self.lora_alpha = kwargs["alpha"] if "alpha" in kwargs else kwargs["low_rank_dimension"]
        self.dropout = kwargs.get("dropout", 0.0)
        self.intervention_positions_dropout = kwargs.get("intervention_positions_dropout", 0.0)
        if self.dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=self.dropout)
        else:
            self.lora_dropout = lambda x: x

        # Actual trainable parameters
        self.lora_A = nn.Parameter(torch.zeros(kwargs["input_dim"], kwargs["low_rank_dimension"]))
        self.lora_B = nn.Parameter(torch.zeros(kwargs["low_rank_dimension"], self.embed_dim))

        # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.lora_A = nn.Parameter(self.lora_A.to(torch.bfloat16))
        self.lora_B = nn.Parameter(self.lora_B.to(torch.bfloat16))

    def forward(
        self, base, source=None, subspaces=None, **kwargs
    ):
        original_input = kwargs["args"]
        
        # Pre-compute B×A for applying steering factor
        lora_weights = self.lora_A @ self.lora_B
        
        if "steering_factor" in subspaces:
            # Apply steering factor directly to the weights
            steering_factor = subspaces["steering_factor"].unsqueeze(dim=-1).unsqueeze(dim=-1).to(base.dtype)  # bs, 1, 1            
            # Apply steering factor directly to the weights
            lora_weights = steering_factor * lora_weights

        lora_output = self.lora_dropout(original_input @ lora_weights)
        output = base + lora_output.to(base.dtype)
        
        return InterventionOutput(
            output=output,
            latent=[None]
        )
    

class LoreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.learned_source(base) - rotated_base), self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))


class PreferenceLoreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """LoReFT intervention whose factor scales the learned source transformation."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs.get("dropout", 0.0))
        self.intervention_positions_dropout = kwargs.get("intervention_positions_dropout", 0.0)
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        rotated_base = self.rotate_layer(base)
        source_output = self.learned_source(base)
        
        # Calculate a "latent" value - norm of the activation difference
        latent = torch.norm(source_output - rotated_base, dim=-1)  # bs, s
        
        diff = (source_output - rotated_base)
        if "steering_factor" in subspaces:
            steering_factor = subspaces["steering_factor"].unsqueeze(dim=-1).unsqueeze(dim=-1)  # bs, 1, 1
            diff = steering_factor * diff

        # Calculate the final output
        output = base + torch.matmul(diff, self.rotate_layer.weight.T)
        
        return InterventionOutput(
            output=self.dropout(output.to(base.dtype)),
            latent=[latent]
        )


class NodireftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    NodiReFT(h) = h + W2^T(W1h + b)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj_layer = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=False).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        output = base + torch.matmul(
            self.learned_source(base), self.proj_layer.weight
        )
        return self.dropout(output.to(base.dtype))


class PreferenceNodireftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    NodiReFT(h) = h + W2^T(W1h + b)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj_layer = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=False).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        
    def forward(
        self, base, source=None, subspaces=None
    ):

        diff = torch.matmul(
            self.learned_source(base), self.proj_layer.weight
        )
        if "steering_factor" in subspaces:
            steering_factor = subspaces["steering_factor"].unsqueeze(dim=-1).unsqueeze(dim=-1)  # bs, 1, 1
            diff = steering_factor * diff

        # Calculate the final output
        output = base + diff
        
        return InterventionOutput(
            output=self.dropout(output.to(base.dtype)),
            latent=[diff]
        )

