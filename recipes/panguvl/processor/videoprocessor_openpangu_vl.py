#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The HuggingFace Inc. team
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# Adapted from transformers/models/qwen2_vl/image_processing_qwen2_vl.py

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

from typing import List, Optional, Union

import torch
from torchvision import transforms
from torchvision.transforms.v2 import functional as F
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.processing_utils import Unpack, VideosKwargs
from transformers.utils import TensorType
from transformers.utils.import_utils import requires
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.video_utils import group_videos_by_shape, reorder_videos


class OpenPanguVLVideoProcessorInitKwargs(VideosKwargs):
    min_pixels: Optional[int]
    max_pixels: Optional[int]
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]
    any_res_dynamic_video_pixels: bool
    any_res_min_video_total_pixels: Optional[int]
    any_res_max_video_total_pixels: Optional[int]
    any_res_min_frame_pixels: Optional[int]
    any_res_max_frame_pixels: Optional[int]


@requires(backends=("torchvision",))
class OpenPanguVLVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    size = {"height": 448, "width": 448}
    do_resize = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    do_convert_rgb = True
    min_pixels = 56 * 56
    max_pixels = 28 * 28 * 1280
    patch_size = 14
    temporal_patch_size = 1
    merge_size = 2
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    any_res_dynamic_video_pixels = True
    any_res_min_video_total_pixels = 448 * 448 * 32
    any_res_max_video_total_pixels = 448 * 448 * 32
    any_res_min_frame_pixels = 56 * 56
    any_res_max_frame_pixels = 28 * 28 * 1280
    valid_kwargs = OpenPanguVLVideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]
    dtype = torch.bfloat16

    def __init__(self, **kwargs: Unpack[OpenPanguVLVideoProcessorInitKwargs]):
        super().__init__(**kwargs)

    def _preprocess(
        self,
        videos: List["torch.Tensor"],
        do_convert_rgb: bool,
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, List[float]]],
        image_std: Optional[Union[float, List[float]]],
        return_tensors: Optional[Union[str, TensorType]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        **kwargs,
    ):
        temporal_patch_size = OpenPanguVLVideoProcessor.temporal_patch_size
        # Recalculate the maximum and minimum resolution of a single frame
        num_frames = sum(video.shape[0] for video in videos)
        if not self.any_res_dynamic_video_pixels:
            self.min_pixels = self.any_res_min_frame_pixels
            self.max_pixels = self.any_res_max_frame_pixels
        else:
            # dynamic video pixels
            self.min_pixels = max(
                min(self.any_res_min_video_total_pixels // num_frames, self.any_res_max_frame_pixels),
                self.any_res_min_frame_pixels,
            )
            self.max_pixels = max(
                min(self.any_res_max_video_total_pixels // num_frames, self.any_res_max_frame_pixels),
                self.any_res_min_frame_pixels,
            )
        # Group videos by size for batched resizing
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            height, width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)
            resized_height, resized_width = height, width
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )
                stacked_videos = F.resize(
                    stacked_videos, size=(resized_height, resized_width), interpolation=interpolation
                )
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)
        # Group videos by size for further processing
        # Needed in case do_resize is False, or resize returns videos with different sizes
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_video_grid_thw = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)

            # rescale and normalize
            stacked_videos = torch.mul(stacked_videos, rescale_factor)
            stacked_videos = transforms.Normalize(mean=image_mean, std=image_std)(stacked_videos)

            # Need to fill frames to cope with temporal_patch_size, avoid time block sticking
            stacked_videos = torch.repeat_interleave(stacked_videos, repeats=temporal_patch_size, dim=1)

            batch_size, grid_t, channel = stacked_videos.shape[:3]
            grid_t, grid_h, grid_w = (
                grid_t // temporal_patch_size,
                resized_height // patch_size,
                resized_width // patch_size,
            )

            stacked_videos = stacked_videos.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            stacked_videos = stacked_videos.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            processed_stacked_videos = stacked_videos.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )

            processed_videos_grouped[shape] = processed_stacked_videos
            processed_video_grid_thw[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_video_grid_thw = reorder_videos(processed_video_grid_thw, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0).to(OpenPanguVLVideoProcessor.dtype)
        video_grid_thw = torch.tensor(processed_video_grid_thw)
        return BatchFeature(
            data={"pixel_values_videos": pixel_values_videos, "video_grid_thw": video_grid_thw},
            tensor_type=return_tensors,
        )


__all__ = ["OpenPanguVLVideoProcessor"]
