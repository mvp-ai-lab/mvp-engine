# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The HuggingFace Inc. team.
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# Adapted from transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py

#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

# This part will be removed in the future.
from collections import defaultdict
from functools import lru_cache, partial
from types import SimpleNamespace
from typing import Optional, Union

import torch
from torchvision.transforms.v2 import functional as F
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import (
    ChannelDimension,
    SizeDict,
    make_flat_list_of_images,
    pil_torch_interpolation_mapping,
    valid_images,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import (
    Qwen2VLImageProcessorFast,
)

# from transformers.image_processing_utils_fast import (
#     group_images_by_shape,
#     reorder_images,
# )


def rescale(image, scale):
    return image * scale


def normalize(image, mean, std):
    return F.normalize(image, mean, std)


@lru_cache(maxsize=10)
def _fuse_mean_std_and_rescale_factor(
    do_normalize: Optional[bool] = None,
    image_mean: Optional[Union[float, list[float]]] = None,
    image_std: Optional[Union[float, list[float]]] = None,
    do_rescale: Optional[bool] = None,
    rescale_factor: Optional[float] = None,
    device: Optional["torch.device"] = None,
) -> tuple:
    if do_rescale and do_normalize:
        # Fused rescale and normalize
        image_mean = torch.tensor(image_mean, device=device) * (1.0 / rescale_factor)
        image_std = torch.tensor(image_std, device=device) * (1.0 / rescale_factor)
        do_rescale = False
    return image_mean, image_std, do_rescale


def rescale_and_normalize(
    images: "torch.Tensor",
    do_rescale: bool,
    rescale_factor: float,
    do_normalize: bool,
    image_mean: Union[float, list[float]],
    image_std: Union[float, list[float]],
) -> "torch.Tensor":
    """
    Rescale and normalize images.
    """
    image_mean, image_std, do_rescale = _fuse_mean_std_and_rescale_factor(
        do_normalize=do_normalize,
        image_mean=image_mean,
        image_std=image_std,
        do_rescale=do_rescale,
        rescale_factor=rescale_factor,
        device=images.device,
    )
    # if/elif as we use fused rescale and normalize if both are set to True
    if do_normalize:
        images = normalize(images.to(dtype=torch.float32), image_mean, image_std)
    elif do_rescale:
        images = rescale(images, rescale_factor)
    images = images.to(OpenPanguVLImageProcessorFast.dtype)

    return images


def _group_images_by_shape(nested_images, is_nested: bool = False):
    """Helper function to flatten a single level of nested image structures and group by shape."""
    grouped_images = defaultdict(list)
    grouped_images_index = {}
    nested_images = [nested_images] if not is_nested else nested_images
    for i, sublist in enumerate(nested_images):
        for j, image in enumerate(sublist):
            key = (i, j) if is_nested else j
            shape = image.shape[1:]
            grouped_images[shape].append(image)
            grouped_images_index[key] = (shape, len(grouped_images[shape]) - 1)

    return grouped_images, grouped_images_index


def _reconstruct_nested_structure(indices, processed_images):
    """Helper function to reconstruct a single level nested structure."""
    # Find the maximum outer index
    max_outer_idx = max(idx[0] for idx in indices.keys())

    # Create the outer list
    result = [None] * (max_outer_idx + 1)

    # Group indices by outer index
    nested_indices = defaultdict(list)
    for i, j in indices.keys():
        nested_indices[i].append(j)

    for i in range(max_outer_idx + 1):
        if i in nested_indices:
            inner_max_idx = max(nested_indices[i])
            inner_list = [None] * (inner_max_idx + 1)
            for j in range(inner_max_idx + 1):
                if (i, j) in indices:
                    shape, idx = indices[(i, j)]
                    inner_list[j] = processed_images[shape][idx]
            result[i] = inner_list

    return result


def group_images_by_shape(
    images: Union[list["torch.Tensor"], "torch.Tensor"],
    disable_grouping: bool,
    is_nested: bool = False,
) -> tuple[dict[tuple[int, int], list["torch.Tensor"]], dict[Union[int, tuple[int, int]], tuple[tuple[int, int], int]]]:
    # If disable grouping is not explicitely provided, we favor disabling it if the images are on CPU, and enabling it otherwise.
    if disable_grouping is None:
        device = images[0][0].device if is_nested else images[0].device
        disable_grouping = device == "cpu"

    if disable_grouping:
        if is_nested:
            return {(i, j): images[i][j].unsqueeze(0) for i in range(len(images)) for j in range(len(images[i]))}, {
                (i, j): ((i, j), 0) for i in range(len(images)) for j in range(len(images[i]))
            }
        else:
            return {i: images[i].unsqueeze(0) for i in range(len(images))}, {i: (i, 0) for i in range(len(images))}

    # Handle single level nested structure
    grouped_images, grouped_images_index = _group_images_by_shape(images, is_nested)

    # Stack images with the same shape
    grouped_images = {shape: torch.stack(images_list, dim=0) for shape, images_list in grouped_images.items()}

    return grouped_images, grouped_images_index


def reorder_images(
    processed_images: dict[tuple[int, int], "torch.Tensor"],
    grouped_images_index: dict[Union[int, tuple[int, int]], tuple[tuple[int, int], int]],
    is_nested: bool = False,
) -> Union[list["torch.Tensor"], "torch.Tensor"]:
    if not is_nested:
        return [
            processed_images[grouped_images_index[i][0]][grouped_images_index[i][1]]
            for i in range(len(grouped_images_index))
        ]

    return _reconstruct_nested_structure(grouped_images_index, processed_images)


class OpenPanguVLImageProcessorFast(Qwen2VLImageProcessorFast):
    temporal_patch_size = 1
    min_pxl = 28
    min_edge = 56
    dtype = torch.bfloat16

    def _prepare_input_images(
        self,
        images,
        do_convert_rgb,
        input_data_format,
        device,
    ) -> list["torch.Tensor"]:
        """
        Prepare the input images for processing.
        """
        images = self._prepare_images_structure(images)
        process_image_fn = partial(
            self._process_image,
            do_convert_rgb=do_convert_rgb,
            input_data_format=input_data_format,
            device=device,
        )

        processed_images = []
        for image in images:
            if (
                image.size[0] <= OpenPanguVLImageProcessorFast.min_pxl
                or image.size[1] <= OpenPanguVLImageProcessorFast.min_pxl
            ):
                if image.size[0] >= image.size[1]:
                    aspect_ratio = OpenPanguVLImageProcessorFast.min_edge * 1.0 / image.size[1]
                    new_image_height = OpenPanguVLImageProcessorFast.min_edge
                    new_image_width = int(aspect_ratio * image.size[0])
                else:
                    aspect_ratio = OpenPanguVLImageProcessorFast.min_edge * 1.0 / image.size[0]
                    new_image_height = int(aspect_ratio * image.size[1])
                    new_image_width = OpenPanguVLImageProcessorFast.min_edge
                image = image.resize((new_image_width, new_image_height))

            processed_images.append(process_image_fn(image))
        return processed_images

    def preprocess(
        self,
        images=None,
        videos=None,
        do_resize=None,
        size=None,
        resample=None,
        do_rescale=None,
        rescale_factor=None,
        do_normalize=None,
        image_mean=None,
        image_std=None,
        min_pixels=None,
        max_pixels=None,
        patch_size=None,
        temporal_patch_size=None,
        merge_size=None,
        do_convert_rgb=None,
        return_tensors=None,
        data_format=ChannelDimension.FIRST,
        input_data_format=None,
        device=None,
        disable_grouping=False,
        **kwargs,
    ):
        temporal_patch_size = OpenPanguVLImageProcessorFast.temporal_patch_size
        params = self._resolve_preprocess_params(
            do_resize=do_resize,
            size=size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
            do_convert_rgb=do_convert_rgb,
        )

        data = self._process_images(images, params, input_data_format, device, disable_grouping, return_tensors)

        return data

    def _resolve_preprocess_params(self, **kwargs):
        params = SimpleNamespace()
        for key, value in kwargs.items():
            setattr(params, key, value if value is not None else getattr(self, key))
        if params.size is None:
            params.size = {"shortest_edge": params.min_pixels, "longest_edge": params.max_pixels}
        params.size = SizeDict(**params.size)
        params.image_mean = tuple(params.image_mean) if params.image_mean else None
        params.image_std = tuple(params.image_std) if params.image_std else None
        return params

    def _process_images(self, images, params, input_data_format, device, disable_grouping, return_tensors):
        images = make_flat_list_of_images(images)
        if not valid_images(images):
            raise ValueError("Invalid image type.")

        images = self._prepare_input_images(
            images=images,
            do_convert_rgb=params.do_convert_rgb,
            input_data_format=input_data_format,
            device=device,
        )

        data = self._preprocess(
            images=images,
            do_resize=params.do_resize,
            size=params.size,
            interpolation=pil_torch_interpolation_mapping.get(params.resample, params.resample),
            do_rescale=params.do_rescale,
            rescale_factor=params.rescale_factor,
            do_normalize=params.do_normalize,
            image_mean=params.image_mean,
            image_std=params.image_std,
            patch_size=params.patch_size,
            temporal_patch_size=params.temporal_patch_size,
            merge_size=params.merge_size,
            do_convert_rgb=params.do_convert_rgb,
            input_data_format=input_data_format,
            device=device,
            disable_grouping=disable_grouping,
            return_tensors=return_tensors,
        )

        return data

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        disable_grouping: Optional[bool],
        return_tensors,
        **kwargs,
    ):
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size["shortest_edge"],
                    max_pixels=size["longest_edge"],
                )
                stacked_images = self.resize(
                    image=stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        processed_grids = {}
        for shape, stacked_images in grouped_images.items():
            resized_height, resized_width = stacked_images.shape[-2:]
            # Fused rescale and normalize
            # patches = rescale_and_normalize(
            #     stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            # )
            patches = stacked_images
            if patches.ndim == 4:
                # add a temporal dimension if we have images
                patches = patches.unsqueeze(1)
            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)
            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
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
            # Reorder dimensions to group grid and patch information for subsequent flattening.
            # (batch, grid_t, grid_h, grid_w, merge_h, merge_w, channel, temp_patch_size, patch_h, patch_w)
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )

            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_grids = reorder_images(processed_grids, grouped_images_index)
        pixel_values = torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors
        )
