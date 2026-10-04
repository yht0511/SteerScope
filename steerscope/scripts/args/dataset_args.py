from dataclasses import dataclass, field
import argparse
import yaml
from steerscope.utils.api_clients import apply_api_model_overrides
from typing import Optional, List, Type

@dataclass
class DatasetArgs:
    models: List[str] = field(default_factory=list)
    steering_datasets: List[str] = field(default_factory=list)
    output_length: Optional[int] = 32
    inference_batch_size: Optional[int] = 8
    # generation related params
    api_concurrency: Optional[int] = 64
    temperature: Optional[float] = 1.0
    data_dir: Optional[str] = None
    train_dir: Optional[str] = None
    dump_dir: Optional[str] = None
    concept_path: Optional[str] = None
    num_of_examples: Optional[int] = None
    rotation_freq: Optional[int] = 1_000
    seed: Optional[int] = None
    max_concepts: Optional[int] = None
    model_name: Optional[str] = None
    steering_model_name: Optional[str] = None
    n_steering_factors: Optional[int] = None
    steering_factors: Optional[List[str]] = None
    multishot_factors_parquet: Optional[str] = None
    steering_layers: Optional[List[int]] = None
    steering_layer: Optional[int] = None
    master_data_dir: Optional[str] = None
    steering_batch_size: Optional[int] = None
    steering_output_length: Optional[int] = None
    steering_num_of_examples: Optional[int] = None
    steering_intervention_type: Optional[str] = None
    lm_model: Optional[str] = None
    run_name: Optional[str] = None
    use_bf16: Optional[bool] = False
    dataset_category: Optional[str] = "instruction"
    lm_use_cache: Optional[bool] = True
    disable_neuronpedia_max_act: Optional[bool] = False
    overwrite_data_dir: Optional[str] = None
    overwrite_metadata_dir: Optional[str] = None
    overwrite_inference_data_dir: Optional[str] = None
    intervene_on_prompt: Optional[bool] = True
    model_params: dict = field(default_factory=dict)  # Add this field to store model_param key-value pairs
    use_wandb: Optional[bool] = False
    steering_prompt_type: Optional[str] = "prepend"
    disable_local_model: Optional[bool] = True
    keep_legacy_output_format: Optional[bool] = False
    steer_data_type: Optional[str] = "concept"  # New parameter to indicate rule-based or concept-based approach
    n_shot: Optional[List[int]] = None  # Number of examples to use for few-shot learning
    defense: Optional[List[str]] = None
    multishot_factors_parquet: Optional[str] = None
    suppress_eval_dir: Optional[str] = None

    def __init__(
        self,
        description: str = "Dataset Creation",
        config_file: str = None,
        section: str = "train",  # Default to 'train' section
        custom_args: Optional[List[dict]] = None,
        override_config: bool = True,
        ignore_unknown: bool = False
    ):
        parser = argparse.ArgumentParser(description=description)
        
        # Command-line argument for YAML configuration file
        parser.add_argument(
            '--config',
            type=str,
            default=config_file,
            help='Path to the YAML configuration file.'
        )

        # Add support for model_param arguments
        parser.add_argument(
            '--model_param',
            action='append',
            help='Model parameters in the format "key=value". Can be used multiple times.'
        )
        
        # Add alias for steering_layer
        parser.add_argument(
            '--layer',
            type=int,
            dest='steering_layer',
            help='Alias for steering_layer parameter.'
        )

        fields = self.__dataclass_fields__
        for field_name, field_def in fields.items():
            if field_name in ['config_file', 'model_params']:
                continue
                
            # Determine field type and appropriate action based on type
            field_type = field_def.type
            if hasattr(field_type, '__origin__') and field_type.__origin__ is List:
                parser.add_argument(
                    f'--{field_name}',
                    type=str, 
                    nargs='+',
                    help=f'Specify {field_name} as a list of values.'
                )
            else:
                arg_type = self._get_argparse_type(field_type)
                parser.add_argument(
                    f'--{field_name}',
                    type=arg_type,
                    help=f'Specify {field_name}.',
                )

        if custom_args:
            for arg in custom_args:
                parser.add_argument(*arg['args'], **arg['kwargs'])
                if '--suppress_eval_dir' in arg['args']:
                    self.suppress_eval_dir = arg['kwargs'].get('default', None)

        # Use parse_known_args to handle unknown arguments based on ignore_unknown flag
        args, unknown = parser.parse_known_args()
        if unknown and not ignore_unknown:
            parser.error(f"Unrecognized arguments: {unknown}")
        elif unknown:
            print(f"DatasetArgs: ignoring unknown arguments: {unknown}")

        # Set default values from dataclass fields
        for field_name, field_def in fields.items():
            default_value = field_def.default
            if default_value == field_def.default_factory:
                default_value = field_def.default_factory()
            setattr(self, field_name, default_value)

        # Load the YAML configuration file if provided
        config_file_path = args.config
        if config_file_path:
            with open(config_file_path, 'r') as file:
                config = apply_api_model_overrides(yaml.safe_load(file) or {})

            # Select the specified section
            section_data = config.get(section, {})
            if not section_data:
                print(f"Warning: Section '{section}' not found in the YAML configuration.")
            else:
                # Initialize attributes from the selected section
                for field_name in fields:
                    if field_name == 'config_file':
                        continue
                    if field_name in section_data:
                        setattr(self, field_name, section_data[field_name])

        # Overwrite with command-line arguments if provided
        if override_config:
            for field_name in vars(args):
                if field_name in ['config']:
                    continue
                arg_value = getattr(args, field_name)
                if arg_value is not None:
                    setattr(self, field_name, arg_value)

        self.config_file = config_file_path

        # Parse model_param arguments
        self.model_params = {}
        if hasattr(args, 'model_param') and args.model_param:
            for param in args.model_param:
                if '=' in param:
                    key, value = param.split('=', 1)
                    self.model_params[key] = value
                else:
                    print(f"Warning: Invalid model_param format: {param}. Expected format: key=value")

        print("Final Configuration:")
        for key in fields:
            print(f"{key}: {getattr(self, key)}")

    @staticmethod
    def _get_argparse_type(field_type: Type) -> Type:
        if hasattr(field_type, '__origin__') and field_type.__origin__ is Optional:
            field_type = field_type.__args__[0]
        if field_type == int:
            return int
        elif field_type == float:
            return float
        elif field_type == bool:
            return lambda x: (str(x).lower() in ['true', '1', 'yes'])
        else:
            return str
