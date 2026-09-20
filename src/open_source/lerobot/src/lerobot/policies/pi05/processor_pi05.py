#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import pad_vector
from lerobot.policies.pi05.revo2_relative import convert_revo2_relative
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_LANGUAGE_MISMATCH_ATTENTION_MASK,
    OBS_LANGUAGE_MISMATCH_TOKENS,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"
    mismatch_task_key: str = "task_mismatch"
    task_id_key: str = "task_id"
    dataset_index_key: str = "dataset_index"
    num_task_classes: int = 100
    task_pool: list[str] | None = None
    dataset_index_to_task_id: dict[int, int] | None = None
    include_state_in_prompt: bool = True

    def __post_init__(self):
        self._cleaned_task_pool: list[str] = []
        self._task_to_id: dict[str, int] = {}
        self._dataset_index_to_task_id: dict[int, int] = {}
        if self.dataset_index_to_task_id:
            self._dataset_index_to_task_id = {
                int(dataset_idx): int(task_id) for dataset_idx, task_id in self.dataset_index_to_task_id.items()
            }
        if self.task_pool:
            for task_name in self.task_pool:
                cleaned_task_name = task_name.strip().replace("_", " ").replace("\n", " ")
                self._cleaned_task_pool.append(cleaned_task_name)
                if cleaned_task_name not in self._task_to_id:
                    self._task_to_id[cleaned_task_name] = len(self._task_to_id)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # TODO: check if this necessary
        state = deepcopy(state)

        # Prepare state (pad to max_state_dim)
        state = pad_vector(state, self.max_state_dim)

        # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        full_mismatch_prompts = []
        task_ids: list[int] = []
        raw_task_indices = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get("task_index")
        raw_dataset_indices = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.dataset_index_key)

        if raw_dataset_indices is not None:
            if isinstance(raw_dataset_indices, torch.Tensor):
                raw_dataset_indices = raw_dataset_indices.tolist()
            task_ids = [
                int(self._dataset_index_to_task_id.get(int(dataset_idx), int(dataset_idx)))
                for dataset_idx in raw_dataset_indices
            ]
        elif raw_task_indices is not None:
            if isinstance(raw_task_indices, torch.Tensor):
                raw_task_indices = raw_task_indices.tolist()
            task_ids = [int(task_idx) for task_idx in raw_task_indices]
        else:
            task_ids = [0] * len(tasks)
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            if self.include_state_in_prompt:
                full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            else:
                full_prompt = f"Task: {cleaned_text}"
            full_prompts.append(full_prompt)

            mismatch_text = cleaned_text
            if self._cleaned_task_pool:
                if len(self._cleaned_task_pool) == 1:
                    mismatch_text = self._cleaned_task_pool[0]
                else:
                    choice_idx = int(np.random.randint(0, len(self._cleaned_task_pool)))
                    mismatch_text = self._cleaned_task_pool[choice_idx]
                    if mismatch_text == cleaned_text:
                        choice_idx = (choice_idx + 1) % len(self._cleaned_task_pool)
                        mismatch_text = self._cleaned_task_pool[choice_idx]
            if self.include_state_in_prompt:
                full_mismatch_prompt = f"Task: {mismatch_text}, State: {state_str};\nAction: "
            else:
                full_mismatch_prompt = f"Task: {mismatch_text}"
            full_mismatch_prompts.append(full_mismatch_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        transition[TransitionKey.COMPLEMENTARY_DATA][self.mismatch_task_key] = full_mismatch_prompts
        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_id_key] = torch.tensor(task_ids, dtype=torch.long)
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


@ProcessorStepRegistry.register(name="revo2_relative_pose_processor")
@dataclass
class Revo2RelativePoseProcessorStep(ProcessorStep):
    """Apply UMI's common-base relative transform before normalization."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        state = observation.get(OBS_STATE)
        action = transition.get(TransitionKey.ACTION)
        if state is None:
            raise ValueError("Revo2 relative pose conversion requires observation.state")
        state, action = convert_revo2_relative(state, action)
        transition[TransitionKey.OBSERVATION] = dict(observation)
        transition[TransitionKey.OBSERVATION][OBS_STATE] = state
        if action is not None:
            transition[TransitionKey.ACTION] = action
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta: Any | None = None,
    task_pool: list[str] | None = None,
    dataset_index_to_task_id: dict[int, int] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI0 policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    normalize_observation_keys = {
        key
        for key in config.input_features
        if key not in {config.box_aux_key, config.cross_center_aux_key}
    }

    # Add remaining processors
    if task_pool is None and dataset_meta is not None and getattr(dataset_meta, "tasks", None) is not None:
        task_pool = list(dataset_meta.tasks.index)
    tokenizer_src = (
        getattr(config, "paligemma_tokenizer_path", None)
        or os.environ.get("PALIGEMMA_TOKENIZER_PATH")
        or "google/paligemma-3b-pt-224"
    )

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
    ]
    if config.action_space == "revo2_eef_pose" and config.action_target_mode == "relative_pose":
        input_steps.append(Revo2RelativePoseProcessorStep())
    input_steps.extend([
        # NOTE: NormalizerProcessorStep MUST come before Pi05PrepareStateTokenizerProcessorStep
        # because the tokenizer step expects normalized state in [-1, 1] range for discretization
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            normalize_observation_keys=normalize_observation_keys,
        ),
        Pi05PrepareStateTokenizerProcessorStep(
            max_state_dim=config.max_state_dim,
            task_id_key=config.task_id_key,
            num_task_classes=config.task_aux_num_classes,
            task_pool=task_pool,
            dataset_index_to_task_id=dataset_index_to_task_id,
            include_state_in_prompt=config.include_state_in_language_prompt,
        ),
        TokenizerProcessorStep(
            tokenizer_name=tokenizer_src,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        TokenizerProcessorStep(
            tokenizer_name=tokenizer_src,
            max_length=config.tokenizer_max_length,
            task_key="task_mismatch",
            output_tokens_key=OBS_LANGUAGE_MISMATCH_TOKENS,
            output_attention_mask_key=OBS_LANGUAGE_MISMATCH_ATTENTION_MASK,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ])

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
