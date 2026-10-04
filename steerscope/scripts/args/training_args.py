from dataclasses import dataclass
import argparse
import sys
import yaml
from steerscope.utils.api_clients import apply_api_model_overrides
from typing import Optional, List


class ModelContainer:
    def __init__(self):
        self._models = {}
    
    def add_model(self, name, params):
        self._models[name] = params
        if name.isidentifier():
            setattr(self, name, params)
        else:
            print(f"Warning: Model name '{name}' is not a valid Python identifier. Use dictionary access.")
    
    def __getitem__(self, key):
        return self._models[key]
    
    def __iter__(self):
        return iter(self._models.items())
    
    def keys(self):
        return self._models.keys()
    
    def values(self):
        return self._models.values()
    
    def items(self):
        return self._models.items()


@dataclass
class ModelParams:
    batch_size: Optional[int] = None
    n_epochs: Optional[int] = None
    topk: Optional[int] = None
    lr: Optional[float] = None
    dropout: Optional[float] = 0.0
    coeff_l1_loss_null: Optional[float] = None
    coeff_latent_l1_loss: Optional[float] = None
    coeff_l1_loss: Optional[float] = None
    coeff_l2_loss: Optional[float] = None
    coeff_norm_loss: Optional[float] = None
    dataset_category: Optional[str] = "continuation"
    intervention_positions: Optional[str] = "all_prompt"
    intervention_positions_dropout: Optional[float] = 0.0
    intervention_layers: Optional[List[int]] = None
    reft_layers: Optional[List[int]] = None
    reft_positions: Optional[str] = "l1"
    reft_type: Optional[str] = "Loreft"
    exclude_bos: Optional[bool] = True
    binarize_dataset: Optional[bool] = False
    train_on_negative: Optional[bool] = False
    intervention_type: Optional[str] = "addition" # clamping   
    gradient_accumulation_steps: Optional[int] = 1
    lora_layers: Optional[List[int]] = None
    lora_components: Optional[List[str]] = None
    lora_alpha: Optional[int] = None
    weight_decay: Optional[float] = 0.0
    temperature_start: Optional[float] = 1e-2
    temperature_end: Optional[float] = 1e-7
    use_synergy: Optional[bool] = False
    use_dpo_loss: Optional[bool] = False
    loss_type: Optional[str] = "scaled_simpo"
    beta: Optional[float] = 1.0
    gemma: Optional[float] = 0.0
    reference_free: Optional[bool] = True
    label_smoothing: Optional[float] = 0.0
    steering_factors: Optional[List[float]] = None
    negative_only: Optional[bool] = False
    simpo_scaler: Optional[float] = 1.0
    use_positional_embedding: Optional[bool] = False
    preference_pairs: Optional[List[str]] = None
    steering_prompt_type: Optional[str] = "prepend"
    substraction_type: Optional[str] = "null_it_out" # normal or null_it_out
    output_length: Optional[int] = 768
    hypernet_name_or_path: Optional[str] = None
    hypernet_initialize_from_pretrained: Optional[bool] = True
    num_hidden_layers: Optional[int] = None
    lm_model: Optional[str] = None
    prompt_temperature: Optional[float] = 0.0
    spherical_kappa: Optional[float] = 20.0
    spherical_beta: Optional[float] = 0.1
    hidra_projected_dim: Optional[int] = 8192
    hidra_negative_slope: Optional[float] = 0.7
    hidra_projection_seed: Optional[int] = 42
    hidra_normalize_direction: Optional[bool] = False
    austeer_topk: Optional[int] = 10
    ode_degree: Optional[int] = 2
    ode_n_components: Optional[int] = 8000
    ode_gamma: Optional[float] = 0.1
    ode_coef0: Optional[float] = 1.0
    ode_linear_classifier: Optional[str] = "lr"
    ode_sketch_seed: Optional[int] = 42
    ode_solver: Optional[str] = "euler"
    ode_steps: Optional[int] = 10
    flas_num_blocks: Optional[int] = 1
    flas_n_steps: Optional[int] = 3
    flas_t_min: Optional[float] = 0.5
    flas_t_max: Optional[float] = 2.0
    flas_div_weight: Optional[float] = 0.1
    flas_total_steps: Optional[int] = 80000
    flas_warmup_steps: Optional[int] = 2000
    flas_max_length: Optional[int] = 256
    flas_concept_max_length: Optional[int] = 64
    flas_n_val_samples: Optional[int] = 100
    flas_val_n_concepts: Optional[int] = 0
    flas_val_every: Optional[int] = 500
    flas_val_batches: Optional[int] = 100
    flas_patience: Optional[int] = 30
    flas_num_workers: Optional[int] = 4
class TrainingArgs:
    def __init__(
        self,
        description: str = "Training Script",
        config_file: str = None,
        section: str = "train",  # Specify section to load
        custom_args: Optional[List[dict]] = None,
        override_config: bool = True,
        ignore_unknown: bool = False
    ):
        parser = argparse.ArgumentParser(description=description)

        # Add config file argument
        parser.add_argument(
            '--config',
            type=str,
            default=config_file,
            help='Path to the YAML configuration file.'
        )

        # Add argument for model-specific parameters
        parser.add_argument(
            '--model_param', '-mp',
            action='append',
            default=[],
            help='Specify model-specific parameters in format "model_name.param=value"'
        )

        # Define global and hierarchical parameters
        global_params = [
            'concept_path', 'model_name', 'layer', 'component',
            'data_dir', 'dump_dir', 'run_name', 'seed', 'subset_seed', 'use_bf16', 'overwrite_data_dir', 'max_concepts', 'num_concepts', 'concept_ids',
            'overwrite_metadata_dir', 'overwrite_inference_data_dir', 'max_num_of_examples', 'use_dpo_loss',
            'use_wandb', 'wandb_project', 'wandb_name', 'output_length'
        ]
        hierarchical_params = [
            'batch_size', 'n_epochs', 'topk',
            'lr', 'coeff_l1_loss_null', 'coeff_l1_loss', 'coeff_l2_loss', 'coeff_norm_loss', 
            'dataset_category', 'intervention_positions', 'intervention_layers',
            'exclude_bos', 'binarize_dataset', 'intervention_type', 'gradient_accumulation_steps',
            'coeff_latent_l1_loss', 'reft_layers', 'reft_positions', 'reft_type', 'lora_layers',
            'lora_components', 'lora_alpha', 'weight_decay', 'temperature_start', 'temperature_end',
            'train_on_negative', 'use_synergy', 'bow_penalty', 'bow_C', 'loss_type', 'beta', 'gemma', 
            'reference_free', 'label_smoothing', 'steering_factors', 'negative_only', 'simpo_scaler', 
            'intervention_positions_dropout', 'dropout', 'preference_pairs', 'steering_prompt_type',
            'hypernet_name_or_path', 'hypernet_initialize_from_pretrained',
            'num_hidden_layers', 'lm_model', 'prompt_temperature',
            'spherical_kappa', 'spherical_beta', 'hidra_projected_dim',
            'hidra_negative_slope', 'hidra_projection_seed',
            'hidra_normalize_direction', 'austeer_topk',
            'ode_degree', 'ode_n_components', 'ode_gamma', 'ode_coef0',
            'ode_linear_classifier', 'ode_sketch_seed', 'ode_solver',
            'ode_steps', 'flas_num_blocks', 'flas_n_steps', 'flas_t_min',
            'flas_t_max', 'flas_div_weight', 'flas_total_steps',
            'flas_warmup_steps', 'flas_max_length',
            'flas_concept_max_length', 'flas_n_val_samples',
            'flas_val_n_concepts', 'flas_val_every', 'flas_val_batches',
            'flas_patience', 'flas_num_workers'
        ]
        model_only_params = ['low_rank_dimension']
        all_params = global_params + hierarchical_params

        forbidden_cli_options = {f"--{param}" for param in model_only_params}
        provided_forbidden_options = sorted({
            argument.split("=", 1)[0]
            for argument in sys.argv[1:]
            if argument.split("=", 1)[0] in forbidden_cli_options
        })
        if provided_forbidden_options:
            names = ", ".join(provided_forbidden_options)
            raise ValueError(
                f"Model-only option(s) cannot be used globally: {names}. "
                "Use --model_param Model.parameter=value."
            )

        # Add arguments for all parameters
        for param in all_params:
            parser.add_argument(f'--{param}', type=self._infer_type(param), help=f'Specify {param}.')

        # Add any custom arguments provided
        if custom_args:
            for arg in custom_args:
                parser.add_argument(*arg['args'], **arg['kwargs'])

        # Use parse_known_args if ignore_unknown is True
        if ignore_unknown:
            args, unknown = parser.parse_known_args()
            if unknown:
                print(f"TrainingArgs: ignoring unknown arguments: {unknown}")
        else:
            args = parser.parse_args()

        # Load the YAML configuration file
        config_file_path = args.config
        if not config_file_path:
            raise ValueError("A config file must be provided.")
        with open(config_file_path, 'r') as file:
            config = apply_api_model_overrides(yaml.safe_load(file) or {})

        # Select the specified section
        section_data = config.get(section, {})
        if not section_data and not ignore_unknown:
            raise ValueError(f"Section '{section}' not found in the YAML configuration.")
        section_data = section_data or {}
        forbidden_global_model_params = sorted(
            set(model_only_params).intersection(section_data)
        )
        if forbidden_global_model_params:
            names = ", ".join(forbidden_global_model_params)
            raise ValueError(
                f"Model-only parameter(s) cannot be set at '{section}' level: "
                f"{names}. Configure them under '{section}.models.<Model>' or "
                "use --model_param Model.parameter=value."
            )

        # Initialize global parameters
        for param in global_params:
            arg_value = getattr(args, param, None)
            config_value = section_data.get(param, None)
            setattr(self, param, arg_value if arg_value is not None else config_value)

        # Initialize hierarchical parameters with global defaults
        for param in hierarchical_params:
            arg_value = getattr(args, param, None)
            config_value = section_data.get(param, None)
            setattr(self, param, arg_value if arg_value is not None else config_value)

        # Initialize models list
        self.models_list = []
        self.model_params = {}
        if 'models' in section_data:
            if isinstance(section_data['models'], dict):
                self.models_list = list(section_data['models'].keys())
                self.model_params = section_data['models']
            elif isinstance(section_data['models'], list):
                self.models_list = section_data['models']
            else:
                raise ValueError("Invalid format for 'models' in config")
        else:
            self.models_list = []

        # Create models container
        self.models = ModelContainer()

        # Initialize per-model parameters
        for model_name in self.models_list:
            params = ModelParams()
            # Set hierarchical parameters to global defaults
            for param in hierarchical_params:
                setattr(params, param, getattr(self, param, None))
            # Override with per-model parameters if available
            if model_name in self.model_params:
                model_config = self.model_params[model_name]
                for param_name, param_value in model_config.items():
                    if param_name in hierarchical_params or param_name in model_only_params:
                        setattr(params, param_name, param_value)
            # Add the model to the container
            self.models.add_model(model_name, params)

        # Process model-specific parameters from command line
        for param_str in args.model_param:
            if '.' not in param_str or '=' not in param_str:
                print(f"Warning: Invalid model parameter format: {param_str}. Expected 'model_name.param=value'")
                continue
                
            key, value = param_str.split('=', 1)
            model_name, param_name = key.split('.', 1)
            
            if model_name not in self.models.keys():
                print(f"Warning: Model {model_name} not found in config")
                continue
                
            if param_name not in hierarchical_params and param_name not in model_only_params:
                print(f"Warning: Parameter {param_name} is not a valid model parameter")
                continue
                
            # Convert value to appropriate type
            param_type = self._infer_type(param_name)
            try:
                if param_type == list:
                    # Handle list parameters
                    value = [item.strip() for item in value.strip('[]').split(',')]
                    # Convert numeric values in the list if needed
                    for i, item in enumerate(value):
                        if item.isdigit():
                            value[i] = int(item)
                        elif self._is_float(item):
                            value[i] = float(item)
                else:
                    value = param_type(value)
                
                # Set the parameter
                setattr(self.models[model_name], param_name, value)
                print(f"Set {model_name}.{param_name} = {value}")
            except ValueError as e:
                print(f"Warning: Could not convert {value} to {param_type.__name__} for {model_name}.{param_name}: {e}")

        # Additional attributes
        self.config_file = config_file_path

        # Print the final configuration
        print("Final Configuration:")
        print("Global Parameters:")
        for key in global_params + hierarchical_params:
            print(f"{key}: {getattr(self, key)}")
        print("\nPer-Model Parameters:")
        for model_name, params in self.models:
            print(f"{model_name}:")
            for field_name in ModelParams.__dataclass_fields__:
                print(f"  {field_name}: {getattr(params, field_name)}")
            for param_name in model_only_params:
                if hasattr(params, param_name):
                    print(f"  {param_name}: {getattr(params, param_name)}")

    @staticmethod
    def _infer_type(param_name: str):
        bool_params = ['use_bf16', 'exclude_bos', 'binarize_dataset', 'train_on_negative', 
                       'use_synergy', 'use_dpo_loss', 'use_wandb', 'reference_free', 'negative_only',
                       'hidra_normalize_direction']
        int_params = ['layer', 'batch_size', 'n_epochs', 'topk', 'seed', 'subset_seed', 'low_rank_dimension',
                      'gradient_accumulation_steps', 'lora_alpha', 'max_concepts', 'num_concepts', 'max_num_of_examples', 'output_length',
                      'hidra_projected_dim', 'hidra_projection_seed',
                      'austeer_topk', 'ode_degree', 'ode_n_components',
                      'ode_sketch_seed', 'ode_steps', 'flas_num_blocks',
                      'flas_n_steps', 'flas_total_steps', 'flas_warmup_steps',
                      'flas_max_length', 'flas_concept_max_length',
                      'flas_n_val_samples', 'flas_val_n_concepts',
                      'flas_val_every', 'flas_val_batches', 'flas_patience',
                      'flas_num_workers']
        float_params = [
            'lr', 'coeff_l1_loss_null', 'coeff_l1_loss', 'coeff_l2_loss', 'coeff_norm_loss', 
            'coeff_latent_l1_loss', 'weight_decay', 'temperature_start', 'temperature_end', 
            'bow_C', 'beta', 'gemma', 'label_smoothing', 'simpo_scaler', 'dropout', 'intervention_positions_dropout',
            'prompt_temperature', 'spherical_kappa', 'spherical_beta',
            'hidra_negative_slope', 'ode_gamma', 'ode_coef0', 'flas_t_min',
            'flas_t_max', 'flas_div_weight']
        str_params = [
            'concept_path', 'model_name', 'component', 
            'data_dir', 'dump_dir', 'run_name', 'dataset_category', 'intervention_positions',
            'intervention_type', 'reft_positions', 'reft_type', 'overwrite_data_dir',
            'overwrite_metadata_dir', 'overwrite_inference_data_dir', 'bow_penalty', 'loss_type',
            'lm_model',
            'wandb_project', 'wandb_name', 'steering_prompt_type',
            'ode_linear_classifier', 'ode_solver']
        list_params = ['concept_ids', 'intervention_layers', 'reft_layers', 'lora_layers', 'lora_components', 'steering_factors', 'preference_pairs']

        if param_name in int_params:
            return int
        elif param_name in float_params:
            return float
        elif param_name in str_params:
            return str
        elif param_name in bool_params:
            return bool 
        elif param_name in list_params:
            return list
        else:
            return str

    @staticmethod
    def _is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False
