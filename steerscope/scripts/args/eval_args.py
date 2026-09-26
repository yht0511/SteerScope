from dataclasses import MISSING, dataclass, field
import argparse
import yaml
from typing import Any, Optional, List, Type

@dataclass
class EvalArgs:
    models: List[str] = field(default_factory=lambda: [])
    targets: Any = None
    artifact_dirs_by_model: dict[str, str] = field(default_factory=dict)
    evaluators: dict[str, Any] = field(default_factory=dict)
    steering_layers: Optional[List[int]] = None
    steering_layer: Optional[int] = None
    report_to: List[str] = field(default_factory=lambda: [])
    rotation_freq: Optional[int] = 1_000
    data_dir: Optional[str] = None
    dump_dir: Optional[str] = None
    root_dump_dir: Optional[str] = None
    lm_model: Optional[str] = None
    model_name: Optional[str] = None
    steering_model_name: Optional[str] = None
    steering_batch_size: Optional[int] = 8
    steering_output_length: Optional[int] = 128
    steering_intervention_type: Optional[str] = None
    temperature: Optional[float] = 1.0
    seed: Optional[int] = 42
    use_bf16: Optional[bool] = False
    output_length: Optional[int] = None
    winrate_baseline: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    run_name: Optional[str] = None
    winrate_split_ratio: Optional[float] = 0.5
    master_data_dir: Optional[str] = "steerscope/data"
    prompt_steering_data_dir: Optional[str] = None
    overwrite_evaluate_dump_dir: Optional[str] = None
    overwrite_cache: Optional[bool] = False
    disable_neuronpedia_max_act: Optional[bool] = False
    intervene_on_prompt: Optional[bool] = True
    runtime_backend: str = "legacy"
    easysteer_url: str = "http://127.0.0.1:8017"
    easysteer_timeout: float = 120.0
    steer_data_type: Optional[str] = "concept"
    defense: Optional[List[str]] = field(default_factory=lambda: [])
    n_shot: Optional[List[int]] = None
    multishot_factors_parquet: Optional[str] = None
    suppress_eval_dir: Optional[str] = None
    evaluation_run_id: Optional[str] = None
    shared_request_cache_dir: Optional[str] = None
    progress_root: Optional[str] = None
    progress_source_root: Optional[str] = None
    generate_reports: bool = True
    report_formats: List[str] = field(default_factory=lambda: ["png", "pdf"])
    report_dpi: int = 300
    report_only: bool = False

    def __init__(
        self,
        description: str = "Evaluation Script",
        config_file: str = None,
        section: str = "train",  # Specify section to load
        custom_args: Optional[List[dict]] = None,
        override_config: bool = True,
        ignore_unknown: bool = False
    ):
        """
        Initializes EvalArgs by parsing command-line arguments and loading configurations from a YAML file.
        """
        parser = argparse.ArgumentParser(description=description)

        # Add config file argument
        parser.add_argument(
            '--config',
            type=str,
            default=config_file,
            help='Path to the YAML configuration file.'
        )

        # Add arguments corresponding to the dataclass fields
        fields = self.__dataclass_fields__
        for field_name, field_def in fields.items():
            if field_name == 'config_file':
                continue

            if field_name == 'report_only':
                parser.add_argument(
                    '--report-only', '--report_only',
                    dest='report_only',
                    action='store_true',
                    default=None,
                    help='Rebuild evaluator reports from saved results without inference.',
                )
                continue

            # Handle list-type fields specially for command line input
            if hasattr(field_def.type, '__origin__') and field_def.type.__origin__ is list:
                parser.add_argument(
                    f'--{field_name}',
                    nargs='+',  # This allows multiple values
                    help=f'Specify {field_name} (can provide multiple values).',
                )
            else:
                arg_type = self._get_argparse_type(field_def.type)
                parser.add_argument(
                    f'--{field_name}',
                    type=arg_type,
                    help=f'Specify {field_name}.',
                )

        # Add any custom arguments provided
        if custom_args:
            for arg in custom_args:
                parser.add_argument(*arg['args'], **arg['kwargs'])

        # Use parse_known_args instead of parse_args if ignore_unknown is True
        if ignore_unknown:
            args, unknown = parser.parse_known_args()
            if unknown:
                print(f"EvalArgs: ignoring unknown arguments: {unknown}")
        else:
            args = parser.parse_args()

        # Load the YAML configuration file
        config_file_path = args.config
        if not config_file_path:
            raise ValueError("A config file must be provided.")
        with open(config_file_path, 'r') as file:
            config = yaml.safe_load(file)

        # Select the specified section
        section_data = config.get(section, {})
        if not section_data:
            raise ValueError(f"Section '{section}' not found in the YAML configuration.")

        # Initialize attributes from the selected section
        for field_name in fields:
            if field_name == 'config_file':
                continue
            field_def = fields[field_name]
            if field_name in section_data:
                value = section_data[field_name]
            elif field_def.default is not MISSING:
                value = field_def.default
            elif field_def.default_factory is not MISSING:
                value = field_def.default_factory()
            else:
                value = None
            setattr(self, field_name, value)

        # Overwrite with command-line arguments if provided
        if override_config:
            for field_name in vars(args):
                if field_name in ['config']:
                    continue
                arg_value = getattr(args, field_name)
                if arg_value is not None:
                    setattr(self, field_name, arg_value)

        # Additional attributes
        self.config_file = config_file_path

        # Print the final configuration
        print("Final Configuration:")
        for key in fields:
            print(f"{key}: {getattr(self, key)}")

    @staticmethod
    def _get_argparse_type(field_type: Type) -> Type:
        """
        Helper method to get the argparse type from the dataclass field type.
        """
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
