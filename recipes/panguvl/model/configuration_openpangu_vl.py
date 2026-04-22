# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The HuggingFace Inc. team.
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

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class OpenPanguVLVisionConfig(PretrainedConfig):
    model_type = "openpangu_vl"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=26,
        num_heads=16,
        rms_norm_eps=1e-06,
        hidden_size=1536,
        hidden_act="gelu",
        intermediate_size=4608,
        out_hidden_size=3584,
        in_chans=3,
        patch_size=14,
        spatial_merge_size=2,
        window_size=112,
        fullatt_block_indexes=[5, 12, 19, 25],
        tokens_per_second=2,
        temporal_patch_size=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.num_heads = num_heads
        self.rms_norm_eps = rms_norm_eps
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.out_hidden_size = out_hidden_size
        self.in_channels = in_chans
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.tokens_per_second = tokens_per_second
        self.temporal_patch_size = temporal_patch_size


class OpenPanguVLTextConfig(PretrainedConfig):
    model_type = "openpangu_vl_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        num_hidden_layers=34,
        num_attention_heads=32,
        num_key_value_heads=8,
        rms_norm_eps=1e-05,
        hidden_size=4096,
        hidden_act="silu",
        intermediate_size=12800,
        initializer_range=0.02,
        tie_word_embeddings=False,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=80,
        vocab_size=153376,
        max_position_embeddings=32768,
        use_cache=True,
        rope_theta=64000000.0,
        attention_dropout=0.0,
        rope_scaling=None,
        image_token_id=None,
        video_token_id=None,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.initializer_range = initializer_range
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            if self.rope_scaling["type"] == "mrope":
                self.rope_scaling["type"] = "default"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self, ignore_keys={"mrope_section", "mrope_interleaved"})
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class OpenPanguVLConfig(PretrainedConfig):
    model_type = "openpangu_vl"
    sub_configs = {"vision_config": OpenPanguVLVisionConfig, "text_config": OpenPanguVLTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=46005,
        video_token_id=144208,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"](**kwargs)

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        super().__init__(**kwargs)


__all__ = ["OpenPanguVLConfig", "OpenPanguVLTextConfig"]
