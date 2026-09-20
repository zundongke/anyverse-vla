# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ast
import builtins
import datetime as dt
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from lerobot import envs
from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig, EvalConfig, PeftConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim import OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.hub import HubMixin

TRAIN_CONFIG_NAME = "train_config.json"


def _parse_optional_str_list(value: Any, field_name: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError) as e:
                raise ValueError(
                    f"Invalid `{field_name}` list format. Example: --dataset.{field_name}='[\"a\",\"b\"]'"
                ) from e
            if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
                raise ValueError(f"`dataset.{field_name}` must be a list of strings.")
            return [item.strip() for item in parsed if item.strip()]
        return [part.strip() for part in stripped.split(",") if part.strip()]
    if isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise ValueError(f"`dataset.{field_name}` must be a list of strings.")
        return [item.strip() for item in value if item.strip()]
    raise ValueError(f"`dataset.{field_name}` must be a string, list of strings, or null.")


@dataclass
class TrainPipelineConfig(HubMixin):
    dataset: DatasetConfig
    env: envs.EnvConfig | None = None
    policy: PreTrainedConfig | None = None
    # Set `dir` to where you would like to save all of the run outputs. If you run another training session
    # with the same value for `dir` its contents will be overwritten unless you set `resume` to true.
    output_dir: Path | None = None
    job_name: str | None = None
    # Set `resume` to true to resume a previous run. In order for this to work, you will need to make sure
    # `dir` is the directory of an existing run with at least one checkpoint in it.
    # Note that when resuming a run, the default behavior is to use the configuration from the checkpoint,
    # regardless of what's provided with the training command at the time of resumption.
    resume: bool = False
    # `seed` is used for training (eg: model initialization, dataset shuffling)
    # AND for the evaluation environments.
    seed: int | None = 1000
    # Number of workers for the dataloader.
    num_workers: int = 4
    batch_size: int = 8
    steps: int = 100_000
    eval_freq: int = 20_000
    log_freq: int = 200
    tolerance_s: float = 1e-4
    step_profile: bool = False
    step_profile_start: int = 10
    step_profile_steps: int = 20
    step_profile_cuda_sync: bool = True
    save_checkpoint: bool = True
    # Checkpoint is saved every `save_freq` training iterations and after the last training step.
    save_freq: int = 20_000
    use_policy_training_preset: bool = True
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    peft: PeftConfig | None = None

    # RA-BC (Reward-Aligned Behavior Cloning) parameters
    use_rabc: bool = False  # Enable reward-weighted training
    rabc_progress_path: str | None = None  # Path to precomputed SARM progress parquet file
    rabc_kappa: float = 0.01  # Hard threshold for high-quality samples
    rabc_epsilon: float = 1e-6  # Small constant for numerical stability
    rabc_head_mode: str | None = "sparse"  # For dual-head models: "sparse" or "dense"
    use_language_mismatch_regularization: bool = False
    language_mismatch_ratio: float = 0.0
    language_mismatch_mode: str = "action_rank"
    language_mismatch_margin: float = 0.05
    language_mismatch_weight: float = 0.3

    # Rename map for the observation to override the image and state keys
    rename_map: dict[str, str] = field(default_factory=dict)
    checkpoint_path: Path | None = field(init=False, default=None)

    def validate(self) -> None:
        if isinstance(self.dataset.repo_id, str):
            parsed_repo_ids = self.dataset.repo_id.strip()
            if parsed_repo_ids.startswith("[") and parsed_repo_ids.endswith("]"):
                try:
                    parsed_repo_ids = ast.literal_eval(parsed_repo_ids)
                except (SyntaxError, ValueError) as e:
                    raise ValueError(
                        "Invalid `dataset.repo_id` list format. Example: "
                        "--dataset.repo_id='[\"repo_a\",\"repo_b\"]'"
                    ) from e
                if not isinstance(parsed_repo_ids, list) or not all(
                    isinstance(repo_id, str) for repo_id in parsed_repo_ids
                ):
                    raise ValueError(
                        "`dataset.repo_id` list format is invalid. It must be a list of strings."
                    )
                self.dataset.repo_id = [repo_id.strip() for repo_id in parsed_repo_ids if repo_id.strip()]
        if isinstance(self.dataset.repo_id_to_root, str):
            try:
                repo_id_to_root = ast.literal_eval(self.dataset.repo_id_to_root)
            except (SyntaxError, ValueError) as e:
                raise ValueError(
                    "Invalid `dataset.repo_id_to_root` format. Example: "
                    "--dataset.repo_id_to_root='{\"repo_a\":\"/path/a\",\"repo_b\":\"/path/b\"}'"
                ) from e
            if not isinstance(repo_id_to_root, dict) or not all(
                isinstance(repo_id, str) and isinstance(root, str)
                for repo_id, root in repo_id_to_root.items()
            ):
                raise ValueError("`dataset.repo_id_to_root` must be a dictionary[str, str].")
            self.dataset.repo_id_to_root = {
                repo_id.strip(): root.strip()
                for repo_id, root in repo_id_to_root.items()
                if repo_id.strip() and root.strip()
            }
        self.dataset.ego_repo_ids = _parse_optional_str_list(self.dataset.ego_repo_ids, "ego_repo_ids")
        self.dataset.ego_source_camera_keys = _parse_optional_str_list(
            self.dataset.ego_source_camera_keys, "ego_source_camera_keys"
        )
        self.dataset.ego_black_camera_keys = _parse_optional_str_list(
            self.dataset.ego_black_camera_keys, "ego_black_camera_keys"
        )
        if isinstance(self.dataset.ego_target_camera_key, str):
            self.dataset.ego_target_camera_key = self.dataset.ego_target_camera_key.strip() or None
        if isinstance(self.dataset.episodes, str):
            try:
                self.dataset.episodes = ast.literal_eval(self.dataset.episodes)
            except (SyntaxError, ValueError) as e:
                raise ValueError(
                    "Invalid `dataset.episodes` format. Examples: "
                    "--dataset.episodes='[0,1]' or "
                    "--dataset.episodes='{\"repo_a\":[0],\"repo_b\":[0]}'"
                ) from e
        if self.dataset.episodes is not None:
            if isinstance(self.dataset.episodes, list):
                if not all(isinstance(ep, int) and ep >= 0 for ep in self.dataset.episodes):
                    raise ValueError("`dataset.episodes` list must contain non-negative integers.")
            elif isinstance(self.dataset.episodes, dict):
                if not all(
                    isinstance(repo_id, str)
                    and isinstance(episodes, list)
                    and all(isinstance(ep, int) and ep >= 0 for ep in episodes)
                    for repo_id, episodes in self.dataset.episodes.items()
                ):
                    raise ValueError(
                        "`dataset.episodes` dict must be dictionary[str, list[non-negative int]]."
                    )
            else:
                raise ValueError(
                    "`dataset.episodes` must be list[int], dict[str, list[int]], or None."
                )
        if isinstance(self.dataset.camera_keys, str):
            camera_keys_spec = self.dataset.camera_keys.strip()
            if camera_keys_spec.startswith("[") and camera_keys_spec.endswith("]"):
                try:
                    parsed_camera_keys = ast.literal_eval(camera_keys_spec)
                except (SyntaxError, ValueError) as e:
                    raise ValueError(
                        "Invalid `dataset.camera_keys` list format. Example: "
                        "--dataset.camera_keys='[\"cam_high\"]'"
                    ) from e
                if not isinstance(parsed_camera_keys, list) or not all(
                    isinstance(camera_key, str) for camera_key in parsed_camera_keys
                ):
                    raise ValueError("`dataset.camera_keys` must be a list of strings.")
                self.dataset.camera_keys = [
                    camera_key.strip() for camera_key in parsed_camera_keys if camera_key.strip()
                ]
            else:
                self.dataset.camera_keys = [
                    camera_key.strip()
                    for camera_key in camera_keys_spec.split(",")
                    if camera_key.strip()
                ]
        if self.dataset.camera_keys is not None:
            if len(self.dataset.camera_keys) == 0:
                raise ValueError("`dataset.camera_keys` must not be empty when provided.")
            if not all(isinstance(camera_key, str) for camera_key in self.dataset.camera_keys):
                raise ValueError("`dataset.camera_keys` must contain strings only.")

        # HACK: We parse again the cli args here to get the pretrained paths if there was some.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            # Only load the policy config
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = Path(policy_path)
        elif self.resume:
            # The entire train config is already loaded, we just need to get the checkpoint dir
            config_path = parser.parse_arg("config_path")
            if not config_path:
                raise ValueError(
                    f"A config_path is expected when resuming a run. Please specify path to {TRAIN_CONFIG_NAME}"
                )

            if not Path(config_path).resolve().exists():
                raise NotADirectoryError(
                    f"{config_path=} is expected to be a local path. "
                    "Resuming from the hub is not supported for now."
                )

            policy_dir = Path(config_path).parent
            if self.policy is not None:
                self.policy.pretrained_path = policy_dir
            self.checkpoint_path = policy_dir.parent

        if self.policy is None:
            raise ValueError(
                "Policy is not configured. Please specify a pretrained policy with `--policy.path`."
            )
        if self.policy.type == "sarm" and isinstance(self.dataset.repo_id, list):
            raise ValueError("SARM policy currently supports only a single dataset.")

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type}"
            else:
                self.job_name = f"{self.env.type}_{self.policy.type}"

        if not self.resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory {self.output_dir} already exists and resume is {self.resume}. "
                f"Please change your output directory so that {self.output_dir} is not overwritten."
            )
        elif not self.output_dir:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/train") / train_dir

        if not self.use_policy_training_preset and (self.optimizer is None or self.scheduler is None):
            raise ValueError("Optimizer and Scheduler must be set when the policy presets are not used.")
        elif self.use_policy_training_preset and not self.resume:
            self.optimizer = self.policy.get_optimizer_preset()
            self.scheduler = self.policy.get_scheduler_preset()

        if self.policy.push_to_hub and not self.policy.repo_id:
            raise ValueError(
                "'policy.repo_id' argument missing. Please specify it to push the model to the hub."
            )

        if self.use_rabc and not self.rabc_progress_path:
            # Auto-detect from dataset path
            repo_id = self.dataset.repo_id
            if self.dataset.root:
                self.rabc_progress_path = str(Path(self.dataset.root) / "sarm_progress.parquet")
            else:
                if isinstance(repo_id, list):
                    raise ValueError(
                        "When using multiple datasets with RA-BC, please set `rabc_progress_path` explicitly."
                    )
                self.rabc_progress_path = f"hf://datasets/{repo_id}/sarm_progress.parquet"

        if not 0.0 <= self.language_mismatch_ratio <= 1.0:
            raise ValueError("`language_mismatch_ratio` must be in [0, 1].")
        if self.language_mismatch_mode not in {"action_rank", "task_id_not_correct"}:
            raise ValueError("`language_mismatch_mode` must be one of {'action_rank', 'task_id_not_correct'}.")
        if self.language_mismatch_margin < 0:
            raise ValueError("`language_mismatch_margin` must be >= 0.")
        if self.language_mismatch_weight < 0:
            raise ValueError("`language_mismatch_weight` must be >= 0.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]

    def to_dict(self) -> dict[str, Any]:
        return draccus.encode(self)  # type: ignore[no-any-return]  # because of the third-party library draccus uses Any as the return type

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / TRAIN_CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: builtins.type["TrainPipelineConfig"],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[Any, Any] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs: Any,
    ) -> "TrainPipelineConfig":
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if TRAIN_CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, TRAIN_CONFIG_NAME)
            else:
                print(f"{TRAIN_CONFIG_NAME} not found in {Path(model_id).resolve()}")
        elif Path(model_id).is_file():
            config_file = model_id
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=TRAIN_CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{TRAIN_CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        cli_args = kwargs.pop("cli_args", [])
        with draccus.config_type("json"):
            return draccus.parse(cls, config_file, args=cli_args)


@dataclass(kw_only=True)
class TrainRLServerPipelineConfig(TrainPipelineConfig):
    # NOTE: In RL, we don't need an offline dataset
    # TODO: Make `TrainPipelineConfig.dataset` optional
    dataset: DatasetConfig | None = None  # type: ignore[assignment] # because the parent class has made it's type non-optional
